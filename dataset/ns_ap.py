import os
import json
import torch
import numpy as np
from pycocotools import coco
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# 0 reserved for background in Mask R-CNN
CLASS_TO_ID = {
    'baking tray': 1,
    'bowl': 2,
    'chopping board': 3,
    'food box': 4,
    'glass': 5,
    'mug': 6,
    'plate': 7,
    'soda can': 8,
    'thermos': 9,
    'wine glass': 10,
    'robot': 11,
    'table': 12
}

SHAPE_TO_ID = {
    'flat': 0,
    'cuboid': 1,
    'cylindrical': 2,
    'hemisphere': 3,
    'irregularly shaped': 4,
}

MATERIAL_TO_ID = {
    'ceramic': 0,
    'metal': 1,
    'wooden': 2,
    'glass': 3,
    'plastic': 4,
    'rubber': 5
}

COLOUR_TO_ID = {
    'white': 0,
    'blue': 1,
    'green': 2,
    'yellow': 3,
    'red': 4,
    'brown': 5,
    'black': 6,
    'purple': 7,
    'cyan': 8,
    'gray': 9,
    'metallic': 10,
    'transparent': 11
}

GOAL_VOCAB = {
    'measure_weight': 0,
    'stack': 1,
    'move_to[right]' : 2,
    'move_to[left]' : 3,
    'pick_up': 4
}

ACTION_VOCAB = {
    'move': 0,
    'approach_grasp': 1,
    'grasp': 2,
    'release': 3,
    'pick_up': 4,
    'put_down': 5,
    'move_right': 6,
    'move_left': 7,
}

class GeneralNSAP(Dataset):
    def __init__(self, params={}) -> None:
        super().__init__()
        assert 'path' in params, "Please provide path to dataset location"
        self.path = params['path']
        subtasks = sorted(os.listdir(self.path)) 

        if 'subtasks' in params:
            subtasks = list(set(subtasks).intersection(set(params['subtasks'])))

        self.split = 'train'
        if 'split' in params:
            self.split = params['split']
        
        self.subtask_split_paths = [
            os.path.join(self.path, subtask, self.split) for subtask in subtasks
        ]

    def __str__(self):
        return "Generic NS-AP dataset"


