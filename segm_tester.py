import os
import torch
from PIL import Image
from model.segmentation import BaselineSegmentationTrainer
from utils.visualisation import SegmentationColourCarousel
import torchvision.transforms as T

trainer = BaselineSegmentationTrainer({}, {}, {}, {})
trainer.set_test()
trainer.load_checkpoint("./output/checkpoints/segmentation.pt")

split = 'test'
subtask = 'stack'
seq_number = 0
frame_number = 0

path = f"/media/m2data/NS_AP/NS_AP_v1_0_a"
path = os.path.join(path, subtask, split, 'sequences')
seq_name = sorted(os.listdir(path))[seq_number]
path = os.path.join(path, seq_name, f"frame_{frame_number:04d}.png")

img = Image.open(path).convert("RGB")

masks, labels = trainer.get_segmenation(img)

colours = SegmentationColourCarousel()
for i in range(len(masks)):
    mask = masks[i]
    # mask[mask > 0.5] = 1
    # mask[mask < 1] = 0
    mask_rgb = mask * torch.tensor(colours.get()).unsqueeze(1).unsqueeze(2).to(mask.device)
    mask_rgb = T.ToPILImage()(mask_rgb.float() / 255)
    mask = mask.float()
    mask[mask == 1] = 0.5
    mask[mask == 0] = 1
    mask = T.ToPILImage()(mask)
    img = Image.composite(img, mask_rgb, mask)

img.save('test/segtest.png')