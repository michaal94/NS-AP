import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


CLASS_TO_ID = {
    'bleach cleanser': 1,
    'bowl': 2,
    'cracker box': 3,
    'foam brick': 4,
    'mug': 5,
    'mustard bottle': 6,
    'potted meat can': 7,
    'sugar box': 8,
    'tomato soup can': 9
}

SHAPE_TO_ID = {
    'cylindrical': 0,
    'hemisphere': 1,
    'irregularly shaped': 2,
    'prismatic': 3
}

MATERIAL_TO_ID = {
    'metal': 0,
    'paper': 1,
    'plastic': 2,
    'sponge': 3
}

COLOUR_TO_ID = {
    'blue': 0,
    'brown': 1,
    'red': 2,
    'white': 3,
    'yellow': 4
}

class AttributesYCB(Dataset):
    def __init__(self, params={}) -> None:
        super().__init__()

        assert 'path' in params
        assert 'split' in params

        self.transforms = []
        self.transforms.append(T.Resize((224, 224)))
        self.transforms.append(T.ToTensor())
        self.transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        if params['split'] != 'val':
            self.transforms.append(T.RandomHorizontalFlip(p=0.2))
        self.transforms = T.Compose(self.transforms)

        json_files = [
            p for p in os.listdir(params['path']) if p.endswith('.json')
        ]
        val_list = [
            'bl_015', 'bl_165',
            'bm_030', 'bm_300',
            'br_105', 'br_270',
            'ml_015', 'ml_165',
            'mm_030', 'mm_300',
            'mr_105', 'mr_270',
            'tl_015', 'tl_165',
            'tm_030', 'tm_300',
            'tr_105', 'tr_270',
        ]
        json_files_filtered = []
        for json_f in json_files:
            for val_suff in val_list:
                if val_suff in json_f:
                    json_files_filtered.append(json_f)
        if params['split'] != 'val':
            json_files_filtered = list(set(json_files) - set(json_files_filtered))
        json_files_filtered = sorted(json_files_filtered)

        if 'name_from_cosy' in params:
            self.include_name = params['name_from_cosy']
        else:
            self.include_name = False

        self.items = []
        for json_path in json_files_filtered:
            with open(os.path.join(params['path'], json_path), 'r') as f:
                attr = json.load(f)
            img_path = os.path.join(
                params['path'],
                json_path.replace('.json', '_crop.png')
            )
            name = CLASS_TO_ID[attr['name']]
            shape = SHAPE_TO_ID[attr['shape']]
            material = MATERIAL_TO_ID[attr['material']]
            colour = COLOUR_TO_ID[attr['colour']]
            self.items.append(
                (
                    img_path,
                    name,
                    shape,
                    material,
                    colour
                )
            )

    def __getitem__(self, idx):
        img_path, name, shape, material, colour = self.items[idx]
        img = Image.open(img_path).convert("RGB")

        target = (
            torch.tensor(name),
            torch.tensor(shape),
            torch.tensor(material),
            torch.tensor(colour),
        )

        img = self.transforms(img)

        if self.include_name:
            return img, torch.tensor(name), target
        else:
            return img, target

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return "YCB attributes"
