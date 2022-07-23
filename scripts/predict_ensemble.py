from glob import glob
import nibabel as nib
import numpy as np
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tqdm import tqdm
from utils.args import get_main_args
from nnunet.nn_unet import NNUnet
from utils.utils import make_empty_dir, set_cuda_devices, set_granularity, verify_ckpt_path
from pytorch_lightning.callbacks import ModelSummary, RichProgressBar
from pytorch_lightning import Trainer
import monai as mn
import torch


parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--folder", type=str, required=True, help="Path to subjects to predict")
parser.add_argument("--best", action="store_true", help="Use 'best' or 'last' models")
parser.add_argument("--binarize", action="store_true", help="Binarize images with 0.5 cutoff")


MonaiTrans = mn.transforms.Compose([
    mn.transforms.LoadImage(),
    mn.transforms.NormalizeIntensity(),
    mn.transforms.ToTensor()
])


if __name__ == "__main__":
    local_args = parser.parse_args()
    files = glob(os.path.join(local_args.folder, '**', 'n4_stripped.nii.gz'))
    MonaiData = mn.data.Dataset(files, transform=MonaiTrans)
    odir = os.path.join(local_args.folder, 'ensemble', 'preds', 'best' if local_args.best else 'last', 'int' if local_args.binarize else 'soft')
    os.makedirs(odir, exist_ok=True)
    args = get_main_args()
    set_granularity()  # Increase maximum fetch granularity of L2 to 128 bytes
    set_cuda_devices(args)
    ckpt_path = verify_ckpt_path(args)

    callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]

    model = NNUnet(args)

    trainer = Trainer(
        default_root_dir=args.results,
        benchmark=True,
        deterministic=False,
        max_epochs=args.epochs,
        precision=16 if args.amp else 32,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=args.save_ckpt,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy="ddp" if args.gpus > 1 else None,
    )



    model.args = args
    trainer.predict(model, test_dataloaders=torch.utils.data.DataLoader(MonaiData), ckpt_path=ckpt_path)

    

    for ix, f in tqdm(enumerate(files)):

        

        preds = [np.load(f) for f in [f1,f2,f3,f4,f5]]
        pred = np.mean(preds, 0)[1]
        if local_args.binarize:
            pred = np.int(pred>0.5)
        img = np.load(os.path.join('/home/lchalcroft/mdunet/predict_data/16_3d/test', f1.split('/')[-1][:-4]+'_x.npy'))[0]
        # ref = nib.load(glob(os.path.join(args.refs, f1.split('/')[-1][:-4]+'*.nii.gz'))[0])
        # print(ref.affine.shape)
        # nib.save(
        #     nib.Nifti1Image(pred, ref.affine, header=ref.header),
        #     os.path.join(odir, f1.split('/')[-1][:-4]+'.nii.gz')
        # )
        nib.save(
            nib.Nifti1Image(img, np.eye(4)),
            os.path.join(odir, 'img_'+f1.split('/')[-1][:-4]+'.nii.gz')
        )
        nib.save(
            nib.Nifti1Image(pred, np.eye(4)),
            os.path.join(odir, f1.split('/')[-1][:-4]+'.nii.gz')
        )

        # reslice img to ref and then apply to label...

    print("Finished!")
