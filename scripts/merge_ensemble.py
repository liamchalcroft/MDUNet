from glob import glob
import nibabel as nib
import numpy as np
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--folder", type=str, required=True, help="Path to individual model predictions")
parser.add_argument("--refs", type=str, required=True, help="Path to original patient folder to use as ref in affine")
parser.add_argument("--best", action="store_true", help="Use 'best' or 'last' models")
parser.add_argument("--binarize", action="store_true", help="Binarize images with 0.5 cutoff")


if __name__ == "__main__":
    args = parser.parse_args()
    folders = [os.path.join(args.folder, str(i), 'preds', 'best' if args.best else 'last') for i in range(5)]
    odir = os.path.join(args.folder, 'ensemble', 'preds', 'best' if args.best else 'last', 'int' if args.binarize else 'soft')
    os.makedirs(odir, exist_ok=True)
    files = [sorted(glob(os.path.join(folder, '*', '*.npy'))) for folder in folders]

    for ix, (f1,f2,f3,f4,f5) in tqdm(enumerate(zip(*files))):
        assert f1.split('/')[-1]==f2.split('/')[-1]==f3.split('/')[-1]==f4.split('/')[-1]==f5.split('/')[-1]
        preds = [np.load(f) for f in [f1,f2,f3,f4,f5]]
        pred = np.mean(preds, 0)[1]
        if args.binarize:
            pred = pred>0.5
        # img = np.load(os.path.join('/home/lchalcroft/mdunet/predict_data/16_3d/test', f1.split('/')[-1][:-4]+'_x.npy'))[0]
        # print(img.shape)
        ref = nib.load(glob(os.path.join(args.refs, f1.split('/')[-1][:-4]+'*.nii.gz'))[0])
        pred = pred.transpose(2,1,0)
        nib.save(
            nib.Nifti1Image(pred, ref.affine, header=ref.header),
            os.path.join(odir, f1.split('/')[-1][:-4]+'.nii.gz')
        )
        nib.save(
            ref,
            os.path.join(odir, 'img_'+f1.split('/')[-1][:-4]+'.nii.gz')
        )
        # nib.save(
        #     nib.Nifti1Image(img, np.eye(4)),
        #     os.path.join(odir, 'img_'+f1.split('/')[-1][:-4]+'.nii.gz')
        # )
        # nib.save(
        #     nib.Nifti1Image(pred, np.eye(4)),
        #     os.path.join(odir, f1.split('/')[-1][:-4]+'.nii.gz')
        # )

        # reslice img to ref and then apply to label...

    print("Finished!")
