import os
import json
import torch
import numpy as np
from pycocotools import coco
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .ns_ap import GeneralNSAP, CLASS_TO_ID, SHAPE_TO_ID, MATERIAL_TO_ID, COLOUR_TO_ID
from utils.utils import ItemCarousel

class AttributesNSAP(GeneralNSAP):
    def __init__(self, params={}) -> None:
        super().__init__(params)

        self.sequence_dirs = [
            os.path.join(p, 'sequences') for p in self.subtask_split_paths
        ]

        self.sequence_paths = []
        for d in self.sequence_dirs:
            self.sequence_paths += [
                os.path.join(d, p, 'sequence.json') for p in sorted(os.listdir(d))
            ]

        if self.split == 'train':
            split_id = 0
        elif self.split == 'val':
            split_id = 100000
        elif self.split == 'test':
            split_id = 200000
        else:
            raise NotImplementedError()

        self.transforms = []
        self.transforms.append(T.ToTensor())
        self.transforms.append(T.Resize((448, 448)))
        self.transforms = T.Compose(self.transforms)

        only_selected = False
        if 'only_selected' in params:
            only_selected = params['only_selected']
        if only_selected:
            selected_percentages = ItemCarousel([0, 0.25, 0.5, 0.75, 1.0])

        self.items = []
        for json_path in tqdm(self.sequence_paths):
            seq_id = int(json_path[-20:-14])
            with open(json_path, 'r') as f:
                sequence_struct = json.load(f)
            observations_gt = sequence_struct['observations_gt']
            image_paths = sequence_struct['image_paths']
            assert len(observations_gt) == len(image_paths), "Sequence error"
            # print(len(observations_gt))
            if only_selected:
                chosen_idx = (len(observations_gt) - 1) * selected_percentages.get()
                chosen_idx = int(chosen_idx)
                observations_gt = [observations_gt[chosen_idx]]
            for i, obs in enumerate(observations_gt):
                img_path = os.path.join(self.path, image_paths[i])
                frame_id = int(img_path[-8:-4])
                image_id = split_id + 100 * seq_id + frame_id
                # print(obs['objects'][0].keys())
                # exit()
                for obj in obs['objects']:
                    mask = obj['mask']
                    if coco.maskUtils.area(mask) < 50:
                        continue
                    name = CLASS_TO_ID[obj['name']]
                    shape = SHAPE_TO_ID[obj['shape']]
                    size = obj['scale_factor']
                    if size > 0.7:
                        size = 1
                    else:
                        size = 0
                    material = MATERIAL_TO_ID[obj['material']]
                    pos = obj['3d_coords']
                    ori = obj['orientation']
                    colour = COLOUR_TO_ID[obj['colour']]
                    # in_hand = int(obj['in_hand'])
                    # raised = int(obj['raised'])
                    # approached = int(obj['approached'])
                    # gripper_over = int(obj['gripper_over'])
                    self.items.append(
                        (
                            img_path,
                            mask,
                            name,
                            shape,
                            material,
                            colour,
                            size,
                            # in_hand,
                            # raised,
                            # approached,
                            # gripper_over,
                            pos,
                            ori
                        )
                    )
                # self.items.append((img_path, masks, image_id))

    def __getitem__(self, idx):
        # img_path, mask, name, shape, material, colour, size, in_hand, raised, approached, gripper_over, pos, ori = self.items[idx]
        img_path, mask, name, shape, material, colour, size, pos, ori = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        mask_arr = coco.maskUtils.decode(mask)
        img = np.concatenate((img, img * np.expand_dims(mask_arr, 2)), axis=0)
        
        target = (
            torch.tensor(name),
            torch.tensor(shape),
            torch.tensor(material),
            torch.tensor(colour),
            torch.tensor(size),
            # torch.tensor(in_hand),
            # torch.tensor(raised),
            # torch.tensor(approached),
            # torch.tensor(gripper_over),
            torch.tensor(pos),
            torch.tensor(ori)
        )

        img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return "NS-AP attributes"

    def visualise(self, idx, save_path=None):
        img, _ = self.__getitem__(idx)
        transform = T.ToPILImage()
        img = transform(img)
        if save_path:
            img.save(save_path)
        else:
            img.show()