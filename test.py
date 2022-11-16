import torch
from nnunet.model import MDUNet
model = MDUNet(3, 1, 2, (3,3,3,3,3), (1,2,2,2,2), (2,2,2,2), md_encoder=True, md_decoder=True, img_size=32)
out = model(torch.rand(1,1,32,32,32))
print(out.shape)
