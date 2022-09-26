import os
import json
import torch
import numpy as np
from pycocotools import coco
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .ns_ap import GeneralNSAP, CLASS_TO_ID

class SegmentationNSAP(GeneralNSAP):
    def __init__(self, params={}) -> None:
        super().__init__(params)

        include_table_mask = False
        if 'table_mask' in params:
            include_table_mask = params['table_mask']

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
        self.transforms = T.Compose(self.transforms)

        only_first = False
        if 'only_first' in params:
            only_first = params['only_first']

        self.items = []
        for json_path in tqdm(self.sequence_paths):
            seq_id = int(json_path[-20:-14])
            with open(json_path, 'r') as f:
                sequence_struct = json.load(f)
            observations_gt = sequence_struct['observations_gt']
            image_paths = sequence_struct['image_paths']
            assert len(observations_gt) == len(image_paths), "Sequence error"
            # print(len(observations_gt))
            for i, obs in enumerate(observations_gt):
                img_path = os.path.join(self.path, image_paths[i])
                frame_id = int(img_path[-8:-4])
                image_id = split_id + 100 * seq_id + frame_id
                masks = []
                masks.append(('robot', obs['robot']['robot_mask']))
                if include_table_mask:
                    masks.append(('table', obs['robot']['table_mask']))
                for obj in obs['objects']:
                    masks.append((obj['name'], obj['mask']))
                self.items.append((img_path, masks, image_id))
                if only_first:
                    break

    def __getitem__(self, idx):
        img_path, coco_masks, img_id = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        image_id = torch.tensor(img_id, dtype=torch.int64)
        target = {}
        target['image_id'] = image_id
        boxes = []
        masks = []
        labels = []
        area = []
        iscrowd = []
        # print(len(coco_masks))
        for class_name, coco_mask in coco_masks:
            class_id = CLASS_TO_ID[class_name]
            # print(coco_mask['counts'])
            # print(coco_mask['counts'].decode("utf-8"))
            # # print(bytes(coco_mask['counts'], "utf-8"))
            # coco_mask['counts'] = coco_mask['counts'].decode("utf-8")
            mask_arr = coco.maskUtils.decode(coco_mask)
            mask_tensor = torch.tensor(mask_arr, dtype=torch.uint8)
            pos = torch.nonzero(mask_tensor)
            if pos.shape[0] == 0:
                continue
            yxmin = torch.min(pos, dim=0)[0]
            yxmax = torch.max(pos, dim=0)[0]
            if (yxmax - yxmin)[0] < 1:
                yxmax[0] += 1
            if (yxmax - yxmin)[1] < 1:
                yxmax[1] += 1
            bbox = torch.flip(
                torch.cat((yxmax, yxmin), dim=0), [0]
            ).float()
            labels.append(class_id)
            masks.append(mask_tensor)
            boxes.append(bbox)

        labels = torch.tensor(labels, dtype=torch.int64)
        target['labels'] = labels

        boxes = torch.stack(boxes)
        target['boxes'] = boxes

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = area

        iscrowd = torch.zeros((area.shape[0],), dtype=torch.int64)
        target['iscrowd'] = iscrowd

        masks = torch.stack(masks)
        target['masks'] = masks

        img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return "NS-AP segmentation"

    def visualise(self, idx, segmentation=True, bbox=True, save_path=None):
        img, target = self.__getitem__(idx)
        transform = T.ToPILImage()
        img = transform(img)
        from utils.visualisation import SegmentationColourCarousel

        if segmentation:
            colours = SegmentationColourCarousel()
            for i in range(target['masks'].shape[0]):
                mask = target['masks'][i, :, :]
                mask_rgb = mask.unsqueeze(2) * torch.tensor(colours.get())
                # print(mask_rgb.shape)
                # print(mask_rgb.shape)
                mask_rgb = transform(mask_rgb.permute((2, 0, 1)).float() / 255)
                mask = mask.float()
                mask[mask == 1] = 0.5
                mask[mask == 0] = 1
                mask = transform(mask)
                img = Image.composite(img, mask_rgb, mask)

        if bbox:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            colours = SegmentationColourCarousel()
            for i in range(target['boxes'].shape[0]):
                box = target['boxes'][i, :]
                draw.rectangle(box.tolist(), outline=tuple(colours.get()), width=3)

        if save_path:
            img.save(save_path)
        else:
            img.show()