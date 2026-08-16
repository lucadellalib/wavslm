# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""WavSLM."""

import json
import math
import os
import re
import warnings
import wave
from array import array
from pathlib import Path
from sys import byteorder
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


__version__ = "0.0.1"

__all__ = ["WavSLM", "__version__"]


DEFAULT_CONFIGS = [
    "lucadellalib/wavslm_2k",
    "lucadellalib/wavslm_4k",
    "lucadellalib/wavslm_65k",
]


try:
    from torch.nn.attention.flex_attention import flex_attention

    HAS_FLEX_ATTENTION = True

    flex_attention = torch.compile(flex_attention)

    def build_bias_mod(bias: "Tensor") -> "Callable":
        def bias_mod(
            score: "Tensor",
            batch: "Tensor",
            head: "Tensor",
            q_idx: "Tensor",
            k_idx: "Tensor",
        ) -> "Tensor":
            return score + bias[batch, head, q_idx, k_idx]

        return bias_mod

except ImportError:
    HAS_FLEX_ATTENTION = False


class FeedForward(nn.Module):
    """Feed-forward neural network.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation layer.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.in_proj = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(ffn_dim, dim)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        output = self.in_proj(input)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional embeddings.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability for attention weights.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.head_dim = dim // num_heads

        # Modules
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        # Parameters
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "Optional[Tensor]" = None,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor]]":
        """Forward pass.

        This method applies relative positional embeddings and multi-head
        attention and handles key-value caching.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        position_bias:
            Precomputed relative positional embeddings for the current input sequence,
            corresponding to positions from `curr_pos` to `curr_pos + seq_length`.
            Shape (num_heads, tgt_seq_length, src_seq_length).
        mask:
            Float mask that is added to the attention scores,
            shape (..., tgt_seq_length, src_seq_length).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        kv_cache:
            Tensor to cache key-value pairs.
            If provided, it should be of shape (batch_size, curr_pos, num_heads, head_dim, 2).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated position `curr_pos + seq_length`;
            - updated key-value cache.

        """
        if curr_pos is None:
            curr_pos = torch.tensor(0, device=input.device)

        B, T, _ = input.shape
        if self.causal:
            next_pos = curr_pos + T
        else:
            next_pos = curr_pos

        if self.causal:
            if kv_cache is None:
                # TODO: avoid hard-coding (might cause issues with ONNX)
                min_cache_size = 512
                kv_cache = torch.zeros(
                    B,
                    max(min_cache_size, T),
                    self.num_heads,
                    self.head_dim,
                    2,
                    device=input.device,
                    dtype=input.dtype,
                )
            elif next_pos > kv_cache.shape[1]:
                # Expand along time dimension
                kv_cache = nn.functional.pad(
                    kv_cache, [0, 0, 0, 0, 0, 0, 0, int(next_pos) - kv_cache.shape[1]]
                )

        qkvs = self.qkv_proj(input).reshape(B, T, -1, self.head_dim)
        qs, ks, vs = qkvs.chunk(3, dim=-2)

        if self.causal and kv_cache is not None:
            kv_cache = kv_cache.type_as(qs)
            kv_cache[:, curr_pos:next_pos, :, :, 0] = ks
            kv_cache[:, curr_pos:next_pos, :, :, 1] = vs

            ks = kv_cache[..., :next_pos, :, :, 0]
            vs = kv_cache[..., :next_pos, :, :, 1]

        # Reshape for scaled_dot_product_attention
        qs = qs.permute(0, 2, 1, 3)  # [B, num_heads, T, head_dim]
        ks = ks.permute(0, 2, 1, 3)  # [B, num_heads, next_pos, head_dim]
        vs = vs.permute(0, 2, 1, 3)  # [B, num_heads, next_pos, head_dim]

        # Compute gated relative position bias
        gated_input = input.reshape(input.shape[:-1] + (self.num_heads, -1))
        gated_input = gated_input.permute(0, 2, 1, 3)

        relative_position_proj = self.gru_rel_pos_linear(gated_input)
        relative_position_proj = relative_position_proj.reshape(
            gated_input.shape[:-1] + (2, 4)
        ).sum(dim=-1)

        gate_a, gate_b = relative_position_proj.sigmoid().chunk(2, dim=-1)
        gate_input = gate_a * (gate_b * self.gru_rel_pos_const.type_as(qs) - 1.0) + 2.0
        gated_position_bias = gate_input * position_bias

        if mask is not None:
            # `mask` must be a float tensor
            gated_position_bias = gated_position_bias + mask

        gated_position_bias = gated_position_bias.type_as(qs)
        output = self._scaled_dot_product_attention(
            qs,
            ks,
            vs,
            gated_position_bias,
        )  # [B, num_heads, T, head_dim]

        # [B, T, num_heads * head_dim]
        output = output.permute(0, 2, 1, 3).reshape(B, T, -1)
        output = self.out_proj(output)  # [B, T, dim]

        return output, next_pos, kv_cache

    def _scaled_dot_product_attention(
        self,
        qs: "Tensor",
        ks: "Tensor",
        vs: "Tensor",
        gated_position_bias: "Tensor",
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=gated_position_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )


class MultiHeadFlexAttention(MultiHeadAttention):
    """See documentation of `MultiHeadAttention`."""

    def _scaled_dot_product_attention(
        self,
        qs: "Tensor",
        ks: "Tensor",
        vs: "Tensor",
        gated_position_bias: "Tensor",
    ) -> "Tensor":
        if not torch.is_grad_enabled() and (not self.training or self.dropout == 0.0):
            return self._flex_attention(
                qs,
                ks,
                vs,
                gated_position_bias,
            )
        return nn.functional.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=gated_position_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )

    @torch.jit.ignore
    def _flex_attention(
        self,
        qs: torch.Tensor,
        ks: torch.Tensor,
        vs: torch.Tensor,
        gated_position_bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.attention.flex_attention.flex_attention(
            qs,
            ks,
            vs,
            score_mod=build_bias_mod(gated_position_bias),
        )


class TransformerLayer(nn.Module):
    """Transformer layer comprising self-attention, feed-forward
    and normalization layers.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    num_heads:
        Number of attention heads in the self-attention mechanism.
    dropout:
        Dropout probability applied in the attention and feed-forward layers.
    causal:
        Whether the module should be causal.
    use_flex_attention:
        Whether to use FlexAttention (if available).
    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
        causal: "bool" = False,
        use_flex_attention: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_ = dropout
        self.causal = causal
        self.use_flex_attention = use_flex_attention
        if use_flex_attention and not HAS_FLEX_ATTENTION:
            warnings.warn(
                "FlexAttention is not available on this platform and/or PyTorch version"
            )

        # Modules
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = (
            MultiHeadFlexAttention
            if use_flex_attention and HAS_FLEX_ATTENTION
            else MultiHeadAttention
        )(dim, num_heads, dropout, causal)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
        curr_pos: "Optional[Tensor]" = None,
        kv_cache: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Tensor, Optional[Tensor]]":
        """See documentation of `MultiHeadAttention.forward`."""
        output = input
        residual = output

        output = self.attention_norm(output)
        output, curr_pos, kv_cache = self.attention(
            output,
            position_bias,
            mask,
            curr_pos,
            kv_cache,
        )
        output = self.dropout(output)
        output = residual + output

        residual = output
        output = self.feed_forward_norm(output)
        output = self.feed_forward(output)
        output = residual + output

        return output, curr_pos, kv_cache


class TransformerEncoderAbsolute(nn.Module):
    """Transformer encoder with relative positional embeddings but without
    convolutional positional embeddings.

    Parameters
    ----------
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    max_cached_steps:
        Maximum number of time steps for which relative positional
        embeddings are cached to avoid recomputation (improves
        runtime at the cost of increased memory usage).
    dropout:
        Dropout probability applied throughout the model.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    lookahead_size:
        Maximum number of future tokens each token can attend to
        (used only if causal=True).
    use_flex_attention:
        Whether to use FlexAttention (if available).

    """

    def __init__(
        self,
        num_layers: "int" = 18,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: "int" = 800,
        max_cached_steps: "int" = 2048,
        dropout: "float" = 0.0,
        causal: "bool" = True,
        window_size: "int" = 512,
        lookahead_size: "int" = 3,
        use_flex_attention: "bool" = False,
    ) -> "None":
        if max_cached_steps > 0 and max_cached_steps < window_size:
            raise ValueError(
                f"`max_cached_steps` ({max_cached_steps}) must be either zero "
                f"or at least `window_size` ({window_size})"
            )

        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.max_cached_steps = max_cached_steps
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.use_flex_attention = use_flex_attention
        self.chunk_size = 1 + lookahead_size

        # Needed to compute position bias
        self._num_buckets = num_buckets // 2
        self._num_buckets_minus_one = self._num_buckets - 1
        self._max_exact = self._num_buckets // 2
        self._num_buckets_minus_max_exact = self._num_buckets - self._max_exact
        self._log_max_distance_over_max_exact = math.log(
            self.max_distance / self._max_exact
        )

        # Modules
        self.relative_embedding = nn.Embedding(num_buckets, num_heads)
        self.layers = nn.ModuleList(
            TransformerLayer(
                dim,
                ffn_dim,
                num_heads,
                dropout,
                causal,
                use_flex_attention,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Non-persistent buffers
        @torch.no_grad()
        def _update_position_bias(
            this: "nn.Module", *_args: "Any", **_kwargs: "Any"
        ) -> "None":
            if max_cached_steps > 0:
                this.position_bias = this._compute_bias(
                    max_cached_steps,
                    max(max_cached_steps, window_size) if causal else max_cached_steps,
                )

        self.register_load_state_dict_post_hook(_update_position_bias)
        self.register_buffer(
            "position_bias",
            torch.as_tensor([[[float("nan")]]]),
            persistent=False,
        )
        _update_position_bias(self)

    def forward(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
        return_all: "bool" = False,
    ) -> "Tuple[Tensor, Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        kv_caches:
            Key-value caches for each layer.
            If provided, each tensor should be of shape
            (batch_size, min(curr_pos, window_size - chunk_size), num_heads, head_dim, 2).
        return_all:
            Whether to return all the hidden states.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim, [1 + num_layers]);
            - updated position `curr_pos + seq_length`;
            - updated key-value caches for each layer.

        """
        if self.causal:
            return self._forward_causal(input, curr_pos, kv_caches, return_all)
        return self._forward_bidirectional(input, return_all)

    def _forward_bidirectional(
        self,
        input: "Tensor",
        return_all: "bool" = False,
    ) -> "Tuple[Tensor, Tensor, List[Optional[Tensor]]]":
        T = input.shape[1]

        if self.training or T > self.max_cached_steps:
            position_bias = self._compute_bias(T, T)
        else:
            position_bias = self.position_bias[:, :T, :T]

        # Experimental: infer batch mask based on constant trailing regions
        B = input.shape[0]
        if B > 1:
            key_padding_mask = (input[..., 0] == input[:, -1:, 0]).int()
            key_padding_mask = key_padding_mask.flip(dims=(1,))
            key_padding_mask = key_padding_mask.cumprod(dim=1)
            key_padding_mask = key_padding_mask.flip(dims=(1,))
            key_padding_mask[key_padding_mask.sum(dim=-1) == 1] = 0
            key_padding_mask = ~key_padding_mask.type(torch.bool)
            input = input.clone()
            input[~key_padding_mask] = 0
            key_padding_mask = (
                (~key_padding_mask)
                .float()
                .masked_fill_(~key_padding_mask, -float("inf"))
            )
            key_padding_mask = key_padding_mask[:, None, None]
        else:
            key_padding_mask = None

        output = input
        output = self.dropout(output)
        position_bias = position_bias.type_as(output)

        outputs = []
        if return_all:
            outputs.append(output)
        new_kv_caches: List[Optional[Tensor]] = []
        for i, layer in enumerate(self.layers):
            output, _, _ = layer(output, position_bias, key_padding_mask)
            new_kv_caches.append(None)
            if return_all:
                outputs.append(output)

        if return_all:
            outputs[-1] = self.norm(outputs[-1])
            output = torch.stack(outputs, dim=-1)
        else:
            output = self.norm(output)

        return output, torch.tensor(0, device=input.device), new_kv_caches

    def _forward_causal(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
        return_all: "bool" = False,
    ) -> "Tuple[Tensor, Tensor, List[Optional[Tensor]]]":
        if curr_pos is None:
            curr_pos = torch.tensor(0, device=input.device)

        T = input.shape[1]
        device = input.device
        next_pos = curr_pos + T
        chunk_size = self.chunk_size

        if self.training or T > self.max_cached_steps:
            position_bias = self._compute_bias(next_pos, next_pos)
        else:
            position_bias = self.position_bias

        # Identify special cases where the rectangular causal mask simplifies to a non-causal mask
        if T <= chunk_size:
            mask = None
        else:
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (T, curr_pos + T), and the only non-masked entries are (i, j) for
            # j in [curr_pos + i - window_size, curr_pos + i + lookahead_size],
            # since row i corresponds to token curr_pos + i
            end_idxes = torch.arange(
                curr_pos + chunk_size,
                next_pos + chunk_size,
                device=device,
            )
            if chunk_size > 1:
                end_idxes = ((end_idxes // chunk_size) * chunk_size).clamp(max=next_pos)
            start_idxes = (end_idxes - self.window_size).clamp(min=0)
            idxes = torch.arange(start_idxes[0], next_pos, device=device)
            mask_ = (idxes[None, :] >= start_idxes[:, None]) & (
                idxes[None, :] < end_idxes[:, None]
            )
            mask = torch.full_like(mask_, fill_value=-float("inf"), dtype=input.dtype)
            mask.masked_fill_(mask_, 0.0)  # (T, min(next_pos, window_size))

        new_kv_caches: List[Optional[Tensor]] = []
        output = input
        output = self.dropout(output)
        end_idx = next_pos.clamp(
            max=torch.as_tensor(position_bias.shape[1], device=device)
        )
        start_idx = end_idx - T
        position_bias = position_bias[:, start_idx:end_idx, :next_pos]
        if position_bias.shape[1] > position_bias.shape[2] or (
            mask is not None and mask.shape[1] > position_bias.shape[2]
        ):
            position_bias = self._compute_bias(next_pos, next_pos)[:, start_idx:end_idx]
        position_bias = position_bias.type_as(output)

        curr_pos = curr_pos.clamp(max=self.window_size - chunk_size)
        kv_cache_start_idx = next_pos - self.window_size + chunk_size
        outputs = []
        if return_all:
            outputs.append(output)
        for i, layer in enumerate(self.layers):
            output, _, new_kv_cache = layer(
                output,
                position_bias,
                mask,
                curr_pos,
                None if kv_caches is None else kv_caches[i],  # JIT compilable
            )
            # Prune cache
            if new_kv_cache is not None:
                # new_kv_cache = new_kv_cache[:, kv_cache_start_idx.clamp(min=0):next_pos]
                # Roll cache
                shift = kv_cache_start_idx.clamp(min=0)
                new_kv_cache = torch.cat(
                    [new_kv_cache[:, shift:], new_kv_cache[:, :shift]], dim=1
                )
                new_kv_cache = new_kv_cache[:, : self.window_size]
            new_kv_caches.append(new_kv_cache)
            if return_all:
                outputs.append(output)
        next_pos = next_pos.clamp(max=self.window_size - chunk_size)
        if return_all:
            outputs[-1] = self.norm(outputs[-1])
            output = torch.stack(outputs, dim=-1)
        else:
            output = self.norm(output)
        return output, next_pos, new_kv_caches

    def _compute_bias(self, query_length: "int", key_length: "int") -> "Tensor":
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(
            self.relative_embedding.weight.device
        )
        values = self.relative_embedding(relative_position_bucket)
        values = values.permute(2, 0, 1)
        return values

    def _relative_positions_bucket(self, relative_positions: "Tensor") -> "Tensor":
        relative_buckets = (relative_positions > 0).to(torch.long) * self._num_buckets
        relative_positions = relative_positions.abs()
        is_small = relative_positions < self._max_exact
        relative_positions_if_large = (
            relative_positions.float() / self._max_exact
        ).log()
        relative_positions_if_large /= self._log_max_distance_over_max_exact
        relative_positions_if_large *= self._num_buckets_minus_max_exact
        relative_positions_if_large += self._max_exact
        relative_positions_if_large = relative_positions_if_large.to(torch.long)
        relative_positions_if_large = relative_positions_if_large.clamp(
            max=self._num_buckets_minus_one
        )
        relative_buckets += torch.where(
            is_small, relative_positions, relative_positions_if_large
        )
        return relative_buckets


class WavSLM(nn.Module):
    """WavSLM model.

    Parameters
    ----------
    vocab_size:
        Vocabulary size.
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of the input and output embeddings in the transformer.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    max_cached_steps:
        Maximum number of time steps for which relative positional
        embeddings are cached to avoid recomputation (improves
        runtime at the cost of increased memory usage).
    dropout:
        Dropout probability applied throughout the model.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to
        (used only if causal=True).
    lookahead_size:
        Maximum number of future tokens each token can attend to
        (used only if causal=True).
    use_flex_attention:
        Whether to use FlexAttention (if available).
    codec_config:
        FocalCodec configuration loaded through PyTorch Hub. Defaults to the
        causal 4k FocalCodec.
    toks_per_step:
        Number of codec tokens predicted at each autoregressive step.

    """

    def __init__(
        self,
        vocab_size: "int" = 4096,
        num_layers: "int" = 18,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: "int" = 800,
        max_cached_steps: "int" = 2048,
        dropout: "float" = 0.0,
        causal: "bool" = True,
        window_size: "int" = 512,
        lookahead_size: "int" = 3,
        use_flex_attention: "bool" = False,
        codec_config: "str" = "lucadellalib/focalcodec_50hz_4k_causal",
        toks_per_step: "int" = 4,
    ) -> "None":
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.max_cached_steps = max_cached_steps
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.use_flex_attention = use_flex_attention
        self.codec_config = codec_config
        self.toks_per_step = toks_per_step
        self.model_id = None

        # Modules
        self.encoder = TransformerEncoderAbsolute(
            num_layers,
            dim,
            ffn_dim,
            num_heads,
            num_buckets,
            max_distance,
            max_cached_steps,
            dropout,
            causal,
            window_size,
            lookahead_size,
            use_flex_attention,
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.chunk_size = 1 + lookahead_size
        self.codec = torch.hub.load(
            repo_or_dir="lucadellalib/focalcodec",
            model="focalcodec",
            config=codec_config,
            trust_repo=True,
        )
        self.codec.eval().requires_grad_(False)

    def jit(self) -> "WavSLM":
        """Return a copy with the codec and language-model modules JIT compiled."""
        from copy import deepcopy

        scripted = deepcopy(self)
        scripted.codec = scripted.codec.jit()
        scripted.encoder = torch.jit.script(scripted.encoder)
        scripted.head = torch.jit.script(scripted.head)
        return scripted

    @torch.jit.ignore
    def load_audio(self, path: "Union[str, Path]") -> "Tuple[Tensor, int]":
        """Load an integer PCM WAV file as a float tensor shaped (channels, time)."""
        with wave.open(str(path), "rb") as file:
            channels = file.getnchannels()
            sample_rate = file.getframerate()
            sample_width = file.getsampwidth()
            frames = file.readframes(file.getnframes())

        if sample_width == 1:
            waveform = torch.frombuffer(bytearray(frames), dtype=torch.uint8)
            waveform = (waveform.to(torch.float32) - 128.0) / 128.0
        elif sample_width == 2:
            waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
            waveform = waveform.to(torch.float32) / 32768.0
        elif sample_width == 3:
            values = torch.frombuffer(bytearray(frames), dtype=torch.uint8)
            values = values.reshape(-1, 3).to(torch.int32)
            waveform = values[:, 0] | values[:, 1].bitwise_left_shift(8)
            waveform |= values[:, 2].bitwise_left_shift(16)
            waveform = waveform - waveform.ge(1 << 23).to(torch.int32) * (1 << 24)
            waveform = waveform.to(torch.float32) / float(1 << 23)
        elif sample_width == 4:
            waveform = torch.frombuffer(bytearray(frames), dtype=torch.int32)
            waveform = waveform.to(torch.float32) / float(1 << 31)
        else:
            raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

        if waveform.numel() % channels:
            raise ValueError(f"Invalid WAV data in {path}")
        return waveform.reshape(-1, channels).t().contiguous(), sample_rate

    @torch.jit.ignore
    def save_audio(
        self,
        path: "Union[str, Path]",
        waveform: "Tensor",
        sample_rate: "int",
    ) -> "None":
        """Save a waveform tensor shaped (channels, time) as 16-bit PCM WAV."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("`waveform` must have shape (channels, time) or (time,)")
        if sample_rate <= 0:
            raise ValueError("`sample_rate` must be positive")

        pcm = waveform.detach().to(device="cpu", dtype=torch.float32)
        pcm = pcm.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16)
        samples = array("h", pcm.t().contiguous().view(-1).tolist())
        if byteorder == "big":
            samples.byteswap()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as file:
            file.setnchannels(waveform.shape[0])
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            file.writeframes(samples.tobytes())

    @torch.jit.ignore
    def resample_audio(
        self, waveform: "Tensor", orig_freq: "int", new_freq: "int"
    ) -> "Tensor":
        """Resample the last axis using Hann-windowed sinc interpolation."""
        if not isinstance(orig_freq, int) or not isinstance(new_freq, int):
            raise TypeError("Sample rates must be integers")
        if orig_freq <= 0 or new_freq <= 0:
            raise ValueError("Sample rates must be positive")
        if not waveform.is_floating_point():
            raise TypeError("`waveform` must have a floating-point dtype")
        if orig_freq == new_freq or waveform.shape[-1] == 0:
            return waveform

        common = math.gcd(orig_freq, new_freq)
        source_rate = orig_freq // common
        target_rate = new_freq // common
        lowpass_filter_width = 6
        base_rate = min(source_rate, target_rate) * 0.99
        width = math.ceil(lowpass_filter_width * source_rate / base_rate)

        output_dtype = waveform.dtype
        work_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        work = waveform.to(work_dtype)
        idx = torch.arange(
            -width,
            width + source_rate,
            dtype=work_dtype,
            device=waveform.device,
        )[None, None]
        idx = idx / source_rate
        offsets = torch.arange(
            0,
            -target_rate,
            -1,
            dtype=work_dtype,
            device=waveform.device,
        )[:, None, None]
        t = (offsets / target_rate + idx) * base_rate
        t = t.clamp(-lowpass_filter_width, lowpass_filter_width)
        window = torch.cos(t * math.pi / lowpass_filter_width / 2).square()
        t = t * math.pi
        sinc = torch.where(t == 0, torch.ones_like(t), t.sin() / t)
        kernel = sinc * window * (base_rate / source_rate)

        shape = work.shape
        length = shape[-1]
        packed = work.reshape(-1, length)
        packed = nn.functional.pad(packed, (width, width + source_rate))
        output = nn.functional.conv1d(packed[:, None], kernel, stride=source_rate)
        output = output.transpose(1, 2).reshape(packed.shape[0], -1)
        target_length = math.ceil(target_rate * length / source_rate)
        output = output[..., :target_length]
        output = output.reshape(shape[:-1] + (target_length,))
        return output.to(output_dtype)

    def forward(
        self,
        input: "Tensor",
        curr_pos: "Optional[Tensor]" = None,
        kv_caches: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        curr_pos:
            Starting position of the current input sequence.
            Default to 0.
        kv_caches:
            Key-value caches for each layer.
            If provided, each tensor should be of shape
            (batch_size, min(curr_pos, window_size - chunk_size), num_heads, head_dim, 2).

        Returns
        -------
            - Output logits of shape (batch_size, seq_length, vocab_size);
            - updated position `curr_pos + seq_length`;
            - updated key-value caches for each layer.

        """
        feats, curr_pos, kv_caches = self.encoder(
            input,
            curr_pos,
            kv_caches,
        )
        logits = self.head(feats)
        return logits, curr_pos, kv_caches

    @torch.jit.ignore
    def _generate_toks(
        self,
        embedding_fn: "Callable",
        bos_toks: "Tensor",
        eos_id: "int" = -1,
        toks_per_step: "int" = 1,
        max_gen_toks: "int" = 100,
        eos_threshold: "float" = float("inf"),
        top_p: "Optional[float]" = 0.3,
        top_k: "Optional[int]" = None,
        temp: "float" = 0.8,
    ) -> "List[Tensor]":
        """Autoregressively generate a sequence of tokens starting from the given prefix tokens.

        This method generates tokens from the given prefix tokens and continues until an EOS token
        is generated or the maximum number of tokens is reached. The generation can use different
        sampling strategies, including greedy search (for top_p=0.0), and top-p sampling with
        temperature scaling for more diverse outputs.

        Parameters
        ----------
        embedding_fn:
            Embedding function.
        bos_toks:
            Tensor containing the prefix tokens (the first token should be BOS). The tensor should
            have shape (batch_size, seq_length).
        eos_id:
            Token ID representing the EOS token. The generation process will stop when this token
            is generated.
        toks_per_step:
            Number of tokens to generate per autoregressive step.
        max_gen_toks:
            Maximum number of tokens to generate. Generation stops either when an EOS token
            is produced or when this number of tokens is generated.
        eos_threshold:
            Threshold that limits the probability of generating an EOS token at each step. When
            `eos_threshold` is finite, generation will avoid producing EOS tokens until their
            probability reaches the specified threshold relative to the maximum log probability.
        top_p:
            Cumulative probability threshold for nucleus (top-p) sampling.
            The model considers the smallest set of tokens whose cumulative
            probability mass exceeds `top_p`, and samples from this set.
                - Typical values: 0.8–0.95
                - If set to 0.0, this is equivalent to greedy decoding.
                - Mutually exclusive with `top_k`.
        top_k:
            Number of highest-probability tokens to keep for top-k sampling.
            The model samples only from the `top_k` most likely tokens at each step.
                - Typical values: 20–100
                - If set to 1, this is equivalent to greedy decoding.
                - Mutually exclusive with `top_p`.
        temp:
            Temperature parameter used to control randomness in the top-p sampling process.
            Values higher than 1.0 increase the randomness by flattening the probability distribution,
            values lower than 1.0 decrease the randomness by sharpening the probability distribution.

        Returns
        -------
            List of tensors, where each tensor is of shape (seq_length,) containing the
            generated sequence of tokens. The generation process will stop once the EOS
            token is generated or the maximum number of tokens is reached.

        Raises
        ------
        ValueError:
            If an invalid argument value is given.

        """
        if bos_toks.shape[1] < toks_per_step:
            raise ValueError(
                f"Need at least `toks_per_step` ({toks_per_step}) BOS tokens, got {bos_toks.shape[1]}"
            )
        if max_gen_toks % toks_per_step != 0:
            raise ValueError(
                f"`max_gen_toks` ({max_gen_toks}) must be a multiple of `toks_per_step` ({toks_per_step})"
            )
        # --- sampling strategy ---
        if top_p is not None and top_k is not None:
            raise ValueError("Specify only one of `top_p` or `top_k`, not both")
        if top_p is None and top_k is None:
            raise ValueError("You must specify either `top_p` or `top_k`")

        self.embed = embedding_fn
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        bos_toks = bos_toks.to(device)
        with torch.no_grad():
            hyp_toks = self._greedy_search(
                bos_toks,
                eos_id,
                toks_per_step,
                max_gen_toks,
                eos_threshold,
                top_p,
                top_k,
                temp,
            )

        if was_training:
            self.train()
        else:
            self.eval()
        self.embed = None

        return hyp_toks

    @torch.jit.ignore
    def _greedy_search(
        self,
        bos_toks: "Tensor",
        eos_id: "int" = -1,
        toks_per_step: "int" = 1,
        max_gen_toks: "int" = 100,
        eos_threshold: "float" = float("inf"),
        top_p: "Optional[float]" = 0.3,
        top_k: "Optional[int]" = None,
        temp: "float" = 0.8,
    ) -> "List[Tensor]":
        batch_size = bos_toks.shape[0]
        device = bos_toks.device

        hyp_toks = torch.full(
            (batch_size, max_gen_toks),
            eos_id,
            device=device,
        )
        lens = torch.zeros(batch_size, device=device)
        alive_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Autoregressive loop
        embedding_state, state = [], []
        num_gen_toks = 0
        embs, *embedding_state = self.embed(bos_toks, *embedding_state)
        while num_gen_toks < max_gen_toks:
            logits, *state = self.forward(embs, *state)
            logits = logits[:, -toks_per_step:].flatten(end_dim=1)
            log_probs = (logits / temp).log_softmax(dim=-1)

            if eos_threshold < float("inf"):
                max_log_probs, _ = log_probs.max(dim=-1)
                eos_log_probs = log_probs[:, eos_id]
                eos_mask = eos_log_probs <= (eos_threshold * max_log_probs)
                log_probs[:, eos_id][eos_mask] = -1e20

            if top_p is not None:
                if top_p == 0.0:
                    # greedy
                    next_tok = log_probs.argmax(dim=-1)
                else:
                    probs = log_probs.exp()
                    next_tok = self._sample_top_p(probs, top_p)

            elif top_k is not None:
                if top_k == 1:
                    # greedy
                    next_tok = log_probs.argmax(dim=-1)
                else:
                    probs = log_probs.exp()
                    next_tok = self._sample_top_k(probs, top_k)

            next_tok = next_tok.unflatten(0, (batch_size, toks_per_step))
            hyp_toks[:, num_gen_toks : num_gen_toks + toks_per_step] = next_tok
            eos_mask = next_tok[:, -1] == eos_id
            alive_mask = alive_mask & (~eos_mask)
            lens[alive_mask] += toks_per_step
            num_gen_toks += toks_per_step
            if not alive_mask.any():
                break
            embs, *embedding_state = self.embed(next_tok, *embedding_state)

        num_gen_toks = max(num_gen_toks, lens.max().item())
        hyp_toks = hyp_toks[:, :num_gen_toks]
        hyp_toks = [hyp_toks[i, : lens[i].long()] for i in range(batch_size)]

        return hyp_toks

    @torch.jit.export
    def _sample_top_p(self, probs: "Tensor", p: "float") -> "Tensor":
        # [B, C]
        probs, idx = probs.sort(dim=-1, descending=True)
        probs_sum = probs.cumsum(dim=-1)
        mask = probs_sum - probs > p
        probs[mask] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tok = torch.multinomial(probs, num_samples=1)
        next_tok = idx.gather(-1, next_tok)
        # [B]
        return next_tok[:, 0]

    @torch.jit.export
    def _sample_top_k(self, probs: "Tensor", k: "int") -> "Tensor":
        # probs: [B, C]
        B, C = probs.shape

        if k > C:
            k = C

        # Get only top-k values (no full sort)
        topk_probs, topk_idx = torch.topk(
            probs, k, dim=-1, largest=True, sorted=False
        )  # [B, k], [B, k]

        # Renormalize only over k
        denom = topk_probs.sum(dim=-1, keepdim=True)  # [B, 1]

        # Fallback if degenerate (rare)
        if (denom <= 0.0).any().item():
            topk_probs = torch.ones_like(topk_probs) / float(k)
        else:
            topk_probs = topk_probs / denom

        # Sample within k
        j = torch.multinomial(topk_probs, 1)  # [B, 1]

        # Map back to original vocab ids
        next_tok = topk_idx.gather(-1, j)  # [B, 1]

        return next_tok[:, 0]  # [B]

    @torch.no_grad()
    def generate(
        self,
        bos_sig: "Optional[Tensor]" = None,
        bos_toks: "Optional[Tensor]" = None,
        max_gen_secs: "Optional[float]" = None,
        max_gen_toks: "Optional[int]" = None,
        top_p: "Optional[float]" = 0.3,
        top_k: "Optional[int]" = None,
        temp: "float" = 0.8,
        sample_rate: "Optional[int]" = None,
    ) -> "Tuple[Tensor, Tensor]":
        """Return the generated waveform and its tokens."""
        codec = self.codec
        token_rate_ms = 20
        if (bos_sig is None) == (bos_toks is None):
            raise ValueError("Exactly one of `bos_sig` or `bos_toks` must be provided.")
        if max_gen_secs is not None and max_gen_toks is not None:
            raise ValueError("Specify only one of `max_gen_secs` or `max_gen_toks`.")

        device = next(self.head.parameters()).device
        bos_sig_orig = None
        if bos_sig is not None:
            if sample_rate is None:
                raise ValueError("`sample_rate` required when providing `bos_sig`")
            bos_sig_orig = bos_sig.to(device)
            bos_sig = self.resample_audio(
                bos_sig_orig, sample_rate, codec.sample_rate_input
            )
            bos_toks = codec.sig_to_toks(bos_sig)
        else:
            bos_toks = bos_toks.to(device)

        if max_gen_secs is None and max_gen_toks is None:
            if bos_sig_orig is not None:
                max_gen_secs = bos_sig_orig.shape[-1] / sample_rate
            else:
                max_gen_toks = bos_toks.shape[-1]
        if max_gen_secs is not None:
            max_gen_toks = int(max_gen_secs * 1000 // token_rate_ms)
        max_gen_toks = (max_gen_toks // self.toks_per_step) * self.toks_per_step
        generated = self._generate_toks(
            lambda *args: codec.toks_to_qfeats(*args, return_state=True),
            bos_toks,
            eos_id=-1,
            toks_per_step=self.toks_per_step,
            max_gen_toks=max_gen_toks,
            top_p=top_p,
            top_k=top_k,
            temp=temp,
        )
        gen_toks = torch.stack(generated)
        hyp_toks = torch.cat([bos_toks, gen_toks], dim=-1)
        hyp_sig = codec.toks_to_sig(hyp_toks)
        if bos_sig_orig is None:
            bos_sig_s = bos_toks.shape[1] * token_rate_ms / 1000
        else:
            bos_sig_s = bos_sig_orig.shape[1] / sample_rate
        gen_sig = hyp_sig[:, int(bos_sig_s * codec.sample_rate_output) :]

        if sample_rate is not None:
            gen_sig = self.resample_audio(
                gen_sig, codec.sample_rate_output, sample_rate
            )
        return gen_sig, gen_toks

    def _config(self) -> "Dict[str, Any]":
        return {
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_buckets": self.num_buckets,
            "max_distance": self.max_distance,
            "max_cached_steps": self.max_cached_steps,
            "dropout": self.dropout,
            "causal": self.causal,
            "window_size": self.window_size,
            "lookahead_size": self.lookahead_size,
            "use_flex_attention": self.use_flex_attention,
            "codec_config": self.codec_config,
            "toks_per_step": self.toks_per_step,
        }

    def info(self) -> "Dict[str, Any]":
        """Return the model information."""
        return {
            "model_id": self.model_id,
            "version": __version__,
            "num_total_params": sum([x.numel() for x in self.state_dict().values()]),
        }

    def to_config(self, config: "str", pretrained: "bool" = False) -> "None":
        """Save the complete model configuration and optionally its weights."""
        config_json = config if config.endswith(".json") else f"{config}.json"
        dirpath = os.path.dirname(config_json)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(config_json, "w") as file:
            json.dump(self._config(), file, indent=2)

        if pretrained:
            state_dict = {
                key: value.detach().cpu()
                for key, value in self.state_dict().items()
                if not key.startswith("codec.")
            }
            checkpoint = os.path.splitext(config_json)[0]
            try:
                from safetensors.torch import save_file

                save_file(state_dict, f"{checkpoint}.safetensors")
            except Exception:
                torch.save(state_dict, f"{checkpoint}.pt")

    def to_pretrained(self, config: "str") -> "None":
        """Save the model configuration and pretrained weights."""
        self.to_config(config, pretrained=True)

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
        overrides: "Optional[Dict[str, Any]]" = None,
        **download_kwargs: "Any",
    ) -> "WavSLM":
        """Load a local or Hugging Face configuration and optional weights."""

        def apply_overrides(values: "Dict[str, Any]") -> "None":
            if overrides is None:
                return
            for path, value in overrides.items():
                target = values
                keys = path.split(".")
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = value

        model_id = config
        config_json = config if config.endswith(".json") else f"{config}.json"
        is_local = os.path.exists(config_json)
        is_repo = bool(re.fullmatch(r"[\w\-]+/[\w\-.]+", config))
        if is_local:
            with open(config_json) as file:
                values = json.load(file)
            checkpoint_base = os.path.splitext(config_json)[0]
            repo_id = None
            checkpoint_name = None
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise ImportError(
                    "`pip install huggingface-hub` to load this model"
                ) from error
            repo_id = config if is_repo else os.path.dirname(config_json)
            filename = "config.json" if is_repo else os.path.basename(config_json)
            config_json = hf_hub_download(
                repo_id=repo_id, filename=filename, **download_kwargs
            )
            with open(config_json) as file:
                values = json.load(file)
            checkpoint_base = None
            checkpoint_name = "model" if is_repo else os.path.splitext(filename)[0]

        apply_overrides(values)
        model = cls(**values)
        if pretrained:
            try:
                from safetensors.torch import load_file

                if is_local:
                    state_dict = load_file(f"{checkpoint_base}.safetensors")
                else:
                    checkpoint = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{checkpoint_name}.safetensors",
                        **download_kwargs,
                    )
                    state_dict = load_file(checkpoint)
            except Exception:
                if is_local:
                    checkpoint = f"{checkpoint_base}.pt"
                else:
                    checkpoint = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{checkpoint_name}.pt",
                        **download_kwargs,
                    )
                state_dict = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            missing = [key for key in missing if not key.startswith("codec.")]
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}"
                )
        model.model_id = model_id
        return model

    @classmethod
    def from_pretrained(
        cls,
        config: "str",
        overrides: "Optional[Dict[str, Any]]" = None,
        **download_kwargs: "Any",
    ) -> "WavSLM":
        """Load a local or Hugging Face configuration and weights."""
        try:
            return cls.from_config(
                config, pretrained=True, overrides=overrides, **download_kwargs
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load the specified pretrained model. "
                f"Available default configurations: {DEFAULT_CONFIGS}"
            ) from error


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = WavSLM().to(device)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, 1024, device=device)
    output, curr_pos, kv_caches = model(input)
    model_jit = model.jit()
    output_jit, curr_pos_jit, kv_caches_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    output.sum().backward()

    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert curr_pos == curr_pos_jit, curr_pos - curr_pos_jit
    for x, y in zip(kv_caches, kv_caches_jit):
        assert torch.allclose(x, y, atol=1e-6), ((x - y) ** 2).mean().sqrt()

    bos_toks = torch.randint(0, model.vocab_size, (B, T), device=device)

    def embedding_fn(x: "Tensor", *_: "Any") -> "Tuple[Tensor, None]":
        return torch.ones(x.shape + (1024,), device=device), None

    output = model._generate_toks(
        embedding_fn,
        bos_toks,
        toks_per_step=4,
        max_gen_toks=100,
        top_p=0.0,
        top_k=None,
        temp=1.0,
    )
    output_jit = model_jit._generate_toks(
        embedding_fn,
        bos_toks,
        toks_per_step=4,
        max_gen_toks=100,
        top_p=0.0,
        top_k=None,
        temp=1.0,
    )
    for x, y in zip(output, output_jit):
        assert (x == y).all()

    model._generate_toks(
        embedding_fn,
        bos_toks,
        toks_per_step=4,
        max_gen_toks=100,
        top_p=0.3,
        top_k=None,
        temp=1.0,
    )
    model._generate_toks(
        embedding_fn,
        bos_toks,
        toks_per_step=4,
        max_gen_toks=100,
        top_p=None,
        top_k=30,
        temp=1.0,
    )

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    T = 50
    model = WavSLM().to(device)

    input = torch.randn(B, T, 1024, device=device)
    batch_output, batch_curr_pos, batch_kv_caches = model(input)
    single_output = []
    single_curr_pos = []
    single_kv_caches = []
    for i in range(B):
        output, curr_pos, kv_caches = model(input[i][None])
        single_output.append(output)
        single_curr_pos.append(curr_pos)
        single_kv_caches.append(kv_caches)
    single_output = torch.cat(single_output)
    assert all(x == single_curr_pos[0] for x in single_curr_pos), single_curr_pos
    single_curr_pos = single_curr_pos[0]
    single_kv_caches = [torch.cat(xs) for xs in zip(*single_kv_caches)]

    assert torch.allclose(batch_output, single_output, atol=1e-2), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )
    assert batch_curr_pos == single_curr_pos, batch_curr_pos - single_curr_pos
    for x, y in zip(batch_kv_caches, single_kv_caches):
        assert torch.allclose(x, y, atol=1e-2), ((x - y) ** 2).mean().sqrt()

    print("Batch invariance test passed")


@torch.no_grad()
def test_causality() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = WavSLM().to(device)

    input = torch.randn(B, T, 1024, device=device)
    model = model.jit()
    output, *_ = model(input)

    incremental_output = []
    state = []
    chunk_size = model.chunk_size
    i = 0
    while i < T:
        output_i, *state = model(input[:, i : i + chunk_size], *state)
        incremental_output.append(output_i)
        i += chunk_size
    incremental_output = torch.cat(incremental_output, dim=1)

    assert torch.allclose(output, incremental_output, atol=1e-2), (
        ((output - incremental_output) ** 2).mean().sqrt(),
    )

    print("Causality test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_causality()
