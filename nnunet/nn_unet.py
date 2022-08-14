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

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedSGD
from data_loading.data_module import get_data_path, get_test_fnames
from monai.inferers import sliding_window_inference

# from monai.networks.nets import DynUNet
from nnunet.model import MDUNet as DynUNet
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from pytorch_lightning.utilities import rank_zero_only
from scipy.special import expit, softmax
from skimage.transform import resize
from utils.logger import DLLogger
from utils.utils import get_config_file, print0

from nnunet.loss import Loss, LossBraTS
from nnunet.metrics import Dice


class NNUnet(pl.LightningModule):
    def __init__(self, args, triton=False, data_dir=None):
        super(NNUnet, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.triton = triton
        if data_dir is not None:
            self.args.data = data_dir
        self.build_nnunet()
        self.best_mean, self.best_epoch, self.test_idx = (0,) * 3
        self.start_benchmark = 0
        self.test_imgs = []
        if args.paste > 0.0:
            self.kernel = torch.zeros(3, 3, 3)
            self.kernel[1, 1, 1] = 0.4
            self.kernel[1, 1, 0] = 0.1
            self.kernel[0, 1, 1] = 0.1
            self.kernel[1, 1, 2] = 0.1
            self.kernel[2, 1, 1] = 0.1
            self.kernel[1, 2, 1] = 0.1
            self.kernel[1, 0, 1] = 0.1
            self.kernel = self.kernel[None][None]
        if not self.triton:
            self.learning_rate = args.learning_rate
            loss = LossBraTS if self.args.brats else Loss
            self.loss = loss(self.args.focal, self.args.shape)
            if self.args.dim == 2:
                self.tta_flips = [[2], [3], [2, 3]]
            else:
                self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
            self.dice = Dice(self.n_class, self.args.brats)
            if self.args.exec_mode in ["train", "evaluate"] and not self.args.benchmark:
                self.dllogger = DLLogger(args.results, args.logname)

    def forward(self, img):
        return torch.argmax(self.model(img), 1)

    def _forward(self, img):
        if self.args.benchmark:
            if self.args.dim == 2 and self.args.data2d_dim == 3:
                img = layout_2d(img, None)
            return self.model(img)
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def compute_loss(self, preds, label):
        if self.args.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(preds.shape[1]):
                loss += self.loss(preds[:, i], label) * 0.5**i
                weights += 0.5**i
            return loss / weights
        return self.loss(preds, label)

    def training_step(self, batch, batch_idx):
        img, lbl = self.get_train_data(batch)
        pred = self.model(img)
        loss = self.compute_loss(pred, lbl)
        self.log("train/loss", loss)
        return loss

    def lesion_paste(self, img, lbl):
        if torch.rand(1).item() < self.args.paste:
            lbl_type = lbl.dtype
            self.kernel = self.kernel.to(img.device)
            self.kernel = self.kernel.to(img.dtype)
            lbl = lbl.to(img.dtype)
            # add line to modify mask with soft boundaries
            mask = lbl
            mask = torch.nn.functional.conv3d(mask, self.kernel, padding=1)
            ix = torch.randperm(lbl.shape[0])
            img = img * (1 - mask[ix]) + img[ix] * mask[ix]
            lbl = ((lbl * (1 - mask[ix]) + lbl[ix] * mask[ix]) > 0.5).to(lbl.dtype)
            del mask, ix
            lbl = lbl.to(lbl_type)
        return img, lbl

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.args.skip_first_n_eval:
            return None
        img, lbl = batch["image"], batch["label"]
        pred = self._forward(img)
        loss = self.loss(pred, lbl)
        if self.args.invert_resampled_y:
            meta, lbl = batch["meta"][0].cpu().detach().numpy(), batch["orig_lbl"]
            pred = nn.functional.interpolate(
                pred, size=tuple(meta[3]), mode="trilinear", align_corners=True
            )
        self.dice.update(pred, lbl[:, 0], loss)

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        pred = self._forward(img).squeeze(0).cpu().detach().numpy()
        if self.args.save_preds:
            meta = batch["meta"][0].cpu().detach().numpy()
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]
            n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]
            if not all(cropped_shape == pred.shape[1:]):
                resized_pred = np.zeros((n_class, *cropped_shape))
                for i in range(n_class):
                    resized_pred[i] = resize(
                        pred[i],
                        cropped_shape,
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                pred = resized_pred
            final_pred = np.zeros((n_class, *original_shape))
            final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            if self.args.brats:
                final_pred = expit(final_pred)
            else:
                final_pred = softmax(final_pred, axis=0)

            self.save_mask(final_pred)

    def get_unet_params(self):
        config = get_config_file(self.args)
        patch_size, spacings = config["patch_size"], config["spacings"]
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [
                2 if ratio <= 2 and size >= 2 * self.args.min_fmap else 1
                for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == self.args.depth:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return config["in_channels"], config["n_class"], kernels, strides, patch_size

    def build_nnunet(self):
        (
            in_channels,
            out_channels,
            kernels,
            strides,
            self.patch_size,
        ) = self.get_unet_params()
        self.n_class = out_channels - 1
        if self.args.brats:
            out_channels = 3

        self.model = DynUNet(
            self.args.dim,
            in_channels,
            out_channels,
            kernels,
            strides,
            strides[1:],
            filters=self.args.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=self.args.deep_supervision,
            deep_supr_num=self.args.deep_supr_num,
            res_block=self.args.res_block,
            trans_bias=True,
            img_size=self.patch_size,
            num_units=self.args.num_units,
            md_encoder=self.args.md_encoder,
            md_decoder=self.args.md_decoder,
        )
        print0(
            f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}"
        )

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds = self.model(image)
        preds = torch.transpose(preds, 0, 1).unsqueeze(0)
        return preds

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    def round(self, tensor):
        return round(torch.mean(tensor).item(), 2)

    def validation_epoch_end(self, outputs):
        if self.current_epoch < self.args.skip_first_n_eval:
            self.log("dice", 0.001 * self.current_epoch)  # To prevent early stopping
            self.dice.reset()
            return None

        dice, loss = self.dice.compute()
        self.dice.reset()

        # Update metrics
        dice_mean = torch.mean(dice)
        if dice_mean >= self.best_mean:
            self.best_mean = dice_mean
            self.best_mean_dice = dice[:] if len(dice.shape) > 0 else dice
            self.best_epoch = self.current_epoch

        metrics = {}
        metrics["Dice"] = self.round(dice)
        metrics["Loss"] = self.round(loss)
        metrics["Max Dice"] = self.round(self.best_mean_dice)
        metrics["Best epoch"] = self.best_epoch
        if self.n_class > 1:
            metrics.update({f"D{i+1}": self.round(m) for i, m in enumerate(dice)})

        self.dllogger.log_metrics(step=self.current_epoch, metrics=metrics)
        self.dllogger.flush()
        if self.args.tb_logs or self.args.wandb_logs:
            self.logger.log_metrics(metrics, step=self.current_epoch)
        self.log("dice", metrics["Dice"])

    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice, _ = self.dice.compute()

    @rank_zero_only
    def on_fit_end(self):
        if not self.args.benchmark:
            metrics = {}
            metrics["dice_score"] = round(self.best_mean.item(), 2)
            metrics["Epoch"] = self.best_epoch
            self.dllogger.log_metrics(step=(), metrics=metrics)
            self.dllogger.flush()

    def configure_optimizers(self):
        optimizer = {
            "sgd": FusedSGD(
                self.parameters(), lr=self.learning_rate, momentum=self.args.momentum
            ),
            "adam": FusedAdam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.args.weight_decay,
            ),
        }[self.args.optimizer.lower()]

        if self.args.scheduler:
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=250,
                    t_total=self.args.epochs
                    * len(self.trainer.datamodule.train_dataloader()),
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {
                "optimizer": optimizer,
                "monitor": "val_loss",
                "lr_scheduler": scheduler,
            }
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_data_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        if self.args.paste > 0:
            img, lbl = self.lesion_paste(img, lbl)
        return img, lbl


def layout_2d(img, lbl):
    batch_size, depth, channels, height, weight = img.shape
    img = torch.reshape(img, (batch_size * depth, channels, height, weight))
    if lbl is not None:
        lbl = torch.reshape(lbl, (batch_size * depth, 1, height, weight))
        return img, lbl
    return img


def flip(data, axis):
    return torch.flip(data, dims=axis)
