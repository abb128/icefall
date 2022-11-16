#!/usr/bin/env python3
# flake8: noqa
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script converts several saved checkpoints
# to a single one using model averaging.
"""

Usage:

(1) Export to torchscript model using torch.jit.trace()

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 35 \
  --avg 10 \
  --jit-trace 1

It will generate 3 files: `encoder_jit_trace.pt`,
`decoder_jit_trace.pt`, and `joiner_jit_trace.pt`.

(2) Export `model.state_dict()`

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 35 \
  --avg 10

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

To use the generated file with `lstm_transducer_stateless2/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./lstm_transducer_stateless2/decode.py \
        --exp-dir ./lstm_transducer_stateless2/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
    # You will find the pre-trained models in icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp

(3) Export to ONNX format

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --onnx 1

It will generate the following files in the given `exp_dir`.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - joiner_encoder_proj.onnx
    - joiner_decoder_proj.onnx

Please see ./streaming-onnx-decode.py for usage of the generated files

Check
https://github.com/k2-fsa/sherpa-onnx
for how to use the exported models outside of icefall.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace.
        It will generate 3 files:
         - encoder_jit_trace.pt
         - decoder_jit_trace.pt
         - joiner_jit_trace.pt

        Check ./jit_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--pnnx",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace for later
        converting to PNNX. It will generate 3 files:
         - encoder_jit_trace-pnnx.pt
         - decoder_jit_trace-pnnx.pt
         - joiner_jit_trace-pnnx.pt
        """,
    )

    parser.add_argument(
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit and --pnnx are ignored and it exports the model
        to onnx format. It will generate the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
            - joiner_encoder_proj.onnx
            - joiner_decoder_proj.onnx

        Refer to ./onnx_check.py and ./onnx_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def export_encoder_model_jit_trace(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.trace()

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)
    states = encoder_model.get_init_states()

    traced_model = torch.jit.trace(encoder_model, (x, x_lens, states))
    traced_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_trace(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.trace()

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = torch.tensor([False])

    traced_model = torch.jit.trace(decoder_model, (y, need_pad))
    traced_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_trace(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.

    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    traced_model = torch.jit.trace(joiner_model, (encoder_out, decoder_out))
    traced_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


class MergedEncoder(nn.Module):
    """
    This combines the encoder and joiner to provide a simplified model where
    the encoder_out is pre-projected according to the joiner.
    """

    def __init__(self, encoder: nn.Module, joiner: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = joiner.encoder_proj

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        warmup = 1.0
        x_lens = torch.tensor([9], dtype=torch.int64)

        x, _, new_states = self.encoder(x, x_lens, (h, c), warmup)
        x = self.encoder_proj(x)

        return x, new_states[0], new_states[1]


class MergedJoiner(nn.Module):
    """
    This combines the decoder and joiner to provide a simplified model that
    takes context and pre-projected encoder_out as input, and then runs
    decoder -> joiner.decoder_proj -> joiner sequentially.
    """

    def __init__(self, decoder: nn.Module, joiner: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.joiner = joiner

    def forward(
        self, context: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        need_pad = False  # Always False, so we can use torch.jit.trace() here
        project_input = False

        decoder_out = self.decoder(context, need_pad)
        decoder_out = self.joiner.decoder_proj(decoder_out)

        joiner_out = self.joiner(encoder_out, decoder_out, project_input)

        joiner_out = joiner_out.squeeze(0)

        return joiner_out


def export_model_onnx(
    model: nn.Module, out_path: str, opset_version: int = 11
) -> None:
    """Export the given model to ONNX format.
    This exports the model as two networks:
        - encoder.onnx, which combines the encoder and joiner's encoder_proj
        - joiner.onnx, which combines the decoder, decoder_proj and joiner.


    The encoder network has 3 inputs:
        - x: mel features, a tensor of shape (N, T, C); dtype is torch.float32
        - h: hidden state, a tensor of shape (num_layers, N, proj_size)
        - c: cell state, a tensor of shape (num_layers, N, hidden_size)
    and has 3 outputs:
        - encoder_out: a tensor of shape (N, T', joiner_dim)
        - next_h: a tensor of shape (num_layers, N, proj_size)
        - next_c: a tensor of shape (num_layers, N, hidden_size)

    h0 and c0 should be initialized to zeros in the beginning. The outputs
    next_h0 and next_c0 should be provided as h0 and c0 inputs in the
    subsequent call.

    Note: The warmup argument is fixed to 1.


    The joiner network has 2 inputs:
        - context: a torch.int64 tensor of shape (N, decoder_model.context_size)
        - encoder_out: a tensor of shape (N, joiner_dim)
    and has one output:
        - logit: a tensor of shape (N, vocab_size)


    Args:
      model:
        The input model
      out_path:
        The path to save the exported ONNX models.
      opset_version:
        The opset version to use.
    """
    encoder_filename = out_path / "encoder.onnx"
    joiner_filename = out_path / "joiner.onnx"

    onnx_encoder = MergedEncoder(model.encoder, model.joiner)
    onnx_joiner = MergedJoiner(model.decoder, model.joiner)
    onnx_encoder.eval()
    onnx_joiner.eval()

    N = 1
    SEGMENT_SIZE = 9
    MEL_FEATURES = 80

    x = torch.zeros(N, SEGMENT_SIZE, MEL_FEATURES, dtype=torch.float32)
    h = torch.rand(model.encoder.num_encoder_layers, N, model.encoder.d_model)
    c = torch.rand(
        model.encoder.num_encoder_layers, N, model.encoder.rnn_hidden_size
    )

    torch.onnx.export(
        onnx_encoder,  # use torch.jit.trace() internally
        (x, h, c),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "h", "c"],
        output_names=["encoder_out", "next_h", "next_c"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "h": {1: "N"},
            "c": {1: "N"},
            "encoder_out": {0: "N", 1: "T'"},
            "next_h": {1: "N"},
            "next_c": {1: "N"},
        },
    )
    logging.info(f"Saved to {encoder_filename}")

    context = torch.zeros(N, model.decoder.context_size, dtype=torch.int64)
    encoder_out, _, _ = onnx_encoder(x, h, c)
    encoder_out = encoder_out.squeeze(0)

    torch.onnx.export(
        onnx_joiner,  # use torch.jit.trace() internally
        (context, encoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["context", "encoder_out"],
        output_names=["logits"],
        dynamic_axes={
            "context": {0: "N"},
            "encoder_out": {0: "N"},
            "logits": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    if params.pnnx:
        params.is_pnnx = params.pnnx
        logging.info("For PNNX")

    logging.info("About to create model")
    model = get_transducer_model(params, enable_giga=False)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device),
                strict=False,
            )
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device),
                strict=False,
            )
    else:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg + 1]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )

    model.to("cpu")
    model.eval()

    if params.onnx:
        logging.info("Export model to ONNX format")
        convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

        opset_version = 11
        export_model_onnx(
            model,
            params.exp_dir,
            opset_version=opset_version,
        )

    elif params.pnnx:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace-pnnx.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace-pnnx.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace-pnnx.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    elif params.jit_trace is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    else:
        logging.info("Not using torchscript")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
