import os
import json
import numpy as np
import torch
from model.visual_recognition import YCBAttributesTrainer, YCBAttributesTrainer2
from dataset.ycb_attributes import CLASS_TO_ID, MATERIAL_TO_ID, COLOUR_TO_ID, SHAPE_TO_ID
import robosuite.utils.transform_utils as T
from PIL import Image
import sys


img_path = '/home/michas/Desktop/codes/NS-AP/test/crop_test1.png'
name = 'foam brick'
name = 'potted meat can'
name = 'tomato soup can'

trainer = YCBAttributesTrainer({}, {}, {}, {})
# trainer = YCBAttributesTrainer2({}, {}, {}, {})
trainer.load_checkpoint(
    '/media/m2data/NS_AP_outputs/attributes/YCB_041122_144714/checkpoint_best.pt'
)
# trainer.load_checkpoint(
#     '/media/m2data/NS_AP_outputs/attributes/YCB_041122_143747/checkpoint_best.pt'
# )

ID_TO_CLASS = {v: k for (k, v) in CLASS_TO_ID.items()}
ID_TO_MATERIAL = {v: k for (k, v) in MATERIAL_TO_ID.items()}
ID_TO_COLOUR = {v: k for (k, v) in COLOUR_TO_ID.items()}
ID_TO_SHAPE = {v: k for (k, v) in SHAPE_TO_ID.items()}

img = Image.open(img_path).convert("RGB")

preds = trainer.get_single_pred(img)
# preds = trainer.get_single_pred(img, torch.tensor(CLASS_TO_ID[name]))
print(f'{ID_TO_CLASS[preds[0]]}\t{ID_TO_SHAPE[preds[1]]}\t{ID_TO_MATERIAL[preds[2]]}\t{ID_TO_COLOUR[preds[3]]}')