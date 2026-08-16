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

"""WavSLM speech-continuation demo."""

import argparse
import os

import torch


def main() -> "None":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lucadellalib/wavslm_4k")
    parser.add_argument(
        "--in-wav",
        "--in_wav",
        dest="in_wav",
        default="audios/prompt_8224-274381-0008.wav",
    )
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: CUDA when available, otherwise CPU)",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--top-p", type=float, default=0.3)
    sampling.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument(
        "--max-gen-secs",
        type=float,
        default=None,
        help="Generation duration in seconds (default: prompt duration)",
    )
    parser.add_argument(
        "--num-continuations",
        type=int,
        default=4,
        help="Number of continuations to generate in one batch (default: 4)",
    )
    args = parser.parse_args()
    if args.num_continuations <= 0:
        parser.error("--num-continuations must be positive")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = torch.hub.load(
        repo_or_dir="lucadellalib/wavslm",
        model="wavslm",
        config=args.config,
        trust_repo=True,
    )
    model = model.to(device).eval()
    prompt, sample_rate = model.load_audio(args.in_wav)
    total_size = sum(parameter.numel() for parameter in model.parameters())
    codec_size = sum(parameter.numel() for parameter in model.codec.parameters())
    lm_size = total_size - codec_size
    decompressor_size = sum(
        parameter.numel() for parameter in model.codec.decompressor.parameters()
    )
    print(f"Codec size: {codec_size / 1e6:.2f}M")
    print(f"LM size: {lm_size / 1e6:.2f}M")
    print(f"LM + decompressor size: {(lm_size + decompressor_size) / 1e6:.2f}M")
    print(f"Total size: {total_size / 1e6:.2f}M")

    prompt_batch = prompt.repeat(args.num_continuations, 1)
    gen_sigs, gen_toks = model.generate(
        bos_sig=prompt_batch,
        max_gen_secs=args.max_gen_secs,
        top_p=args.top_p if args.top_k is None else None,
        top_k=args.top_k,
        temp=args.temp,
        sample_rate=sample_rate,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    gen_sigs = gen_sigs.cpu()
    for index, gen_sig in enumerate(gen_sigs, start=1):
        model.save_audio(
            os.path.join(args.out_dir, f"gen_{index}_{sample_rate}.wav"),
            gen_sig,
            sample_rate,
        )
        model.save_audio(
            os.path.join(args.out_dir, f"hyp_{index}_{sample_rate}.wav"),
            torch.cat([prompt, gen_sig.unsqueeze(0)], dim=-1),
            sample_rate,
        )
    model.save_audio(
        os.path.join(args.out_dir, f"prompt_{sample_rate}.wav"),
        prompt,
        sample_rate,
    )
    print(
        f"Generated {args.num_continuations} continuations "
        f"of {gen_toks.shape[-1]} tokens each"
    )
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
