# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import run

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, default="01", help="Path to data")
parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
parser.add_argument(
    "--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number"
)
parser.add_argument(
    "--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet"
)
parser.add_argument(
    "--amp", action="store_true", help="Enable automatic mixed precision"
)
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument(
    "--deep_supervision", action="store_true", help="Enable deep supervision loss"
)
parser.add_argument(
    "--resume_training", action="store_true", help="Resume training from checkpoint"
)
parser.add_argument("--data", type=str, default="./data", help="Path to data directory")
parser.add_argument(
    "--results", type=str, default="./results", help="Path to results directory"
)
parser.add_argument("--logname", type=str, default="log", help="Name of dlloger output")
parser.add_argument(
    "--wandb_project", type=str, default=None, help="Project name for Weights & Biases"
)
parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to train")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--mde", action="store_true", help="Enable MD modules in encoder")
parser.add_argument("--mdd", action="store_true", help="Enable MD modules in decoder")
parser.add_argument("--swin", action="store_true", help="Use Swin-UNETR.")
parser.add_argument("--shape", action="store_true", help="Use shape term in loss")
parser.add_argument(
    "--scheduler", action="store_true", help="Use learning rate warmup scheduler"
)
parser.add_argument("--focal", action="store_true", help="Use focal term in loss")
parser.add_argument(
    "--tb_logs", action="store_true", help="Log training via tensorboard"
)
parser.add_argument(
    "--wandb_logs", action="store_true", help="Log training via weights and biases"
)
parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
parser.add_argument(
    "--paste", type=float, default=0.0, help="Probability to use lesion pasting."
)
parser.add_argument(
    "--skip_first_n_eval",
    type=int,
    default=1,
    help="Skip the evaluation for the first n epochs.",
)
parser.add_argument(
    "--val_epochs", type=int, default=0, help="Frequency of validation epochs."
)
parser.add_argument(
    "--weight_path", type=str, default=None, help="Path for loading model weights"
)
parser.add_argument(
    "--gradient_clip_val",
    type=float,
    default=12,
    help="Gradient clipping norm value",
)


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode train --task {args.task} --save_ckpt --scheduler "
    cmd += f"--results {args.results} "
    cmd += f"--ckpt_store_dir {args.results} "
    cmd += f"--weight_path {args.weight_path} "
    cmd += f"--data {args.data} "
    cmd += f"--logname {args.logname} "
    cmd += f"--dim {args.dim} "
    if args.batch_size is not None and args.batch_size > 0:
        cmd += f"--batch_size {args.batch_size} "
    else:
        cmd += f"--batch_size {2 if args.dim == 3 else 64} "
    cmd += f"--val_batch_size {4 if args.dim == 3 else 64} "
    cmd += f"--fold {args.fold} "
    cmd += f"--skip_first_n_eval {args.skip_first_n_eval} "
    cmd += f"--val_epochs {args.val_epochs} "
    cmd += f"--gpus {args.gpus} "
    cmd += f"--epochs {args.epochs} "
    cmd += f"--learning_rate {args.learning_rate} "
    cmd += f"--paste {args.paste} "
    # cmd += "--scheduler " if args.scheduler else ""
    cmd += "--amp " if args.amp else ""
    cmd += "--tta " if args.tta else ""
    cmd += "--resume_training " if args.resume_training else ""
    cmd += "--deep_supervision " if args.deep_supervision else ""
    cmd += "--md_encoder " if args.mde else ""
    cmd += "--md_decoder " if args.mdd else ""
    cmd += "--swin " if args.swin else ""
    cmd += "--shape " if args.shape else ""
    cmd += "--focal " if args.focal else ""
    cmd += "--tb_logs " if args.tb_logs else ""
    cmd += "--wandb_logs " if args.wandb_logs else ""
    cmd += f"--wandb_project {args.wandb_project}" if args.wandb_logs else ""
    cmd += f"--gradient_clip_val {args.gradient_clip_val}"
    run(cmd, shell=True)
