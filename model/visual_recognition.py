import abc
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from pycocotools import coco
import torchvision.transforms as T


class VisualRecognitionLoader:
    @abc.abstractmethod
    def get_scene(self, image, segmentation, scene_gt):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class VisualGTLoader(VisualRecognitionLoader):
    def get_scene(self, image, segmentation, scene_gt):
        assert 'objects' in scene_gt, "Provide GT scene"
        return scene_gt['objects']


class AttributesBaselineTrainer(VisualRecognitionLoader):
    def __init__(self, model_params, loss_params, optimiser_params, scheduler_params) -> None:
        self.model = AttributesBaselineModel(model_params)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = 0.005
        weight_decay = 0
        if 'lr' in optimiser_params:
            lr = scheduler_params['lr']
        if 'weight_decay' in optimiser_params:
            weight_decay = scheduler_params['weight_decay']
        self.optimiser = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

        self.epoch = 0
        self.new_epoch_trigger = False
        if 'starting_epoch' in scheduler_params:
            self.epoch = scheduler_params['starting_epoch']
        scheduler_step = 1
        if 'scheduler_step' in scheduler_params:
            scheduler_step = scheduler_params['scheduler_step']
        gamma = 0.7
        if 'gamma' in scheduler_params:
            gamma = scheduler_params['gamma']

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=scheduler_step,
            gamma=gamma
        )

        self.loss_weight = 1.0
        if 'loss_weight' is loss_params:
            self.loss_weight = loss_params['loss_weight']
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.loss_function = self.loss_fn

        self.program_idx_to_token = None
        self.ce_num = 5
        if 'ce_num' in model_params:
            self.ce_num = model_params['ce_num']

        self.transforms = None


    def loss_fn(self, pred, target):
        # print()
        # print(pred, target)
        loss = self.ce_loss(pred[0], target[0])
        loss +=  self.ce_loss(pred[1], target[1])
        loss +=  self.ce_loss(pred[2], target[2])
        loss +=  self.ce_loss(pred[3], target[3])
        loss +=  self.ce_loss(pred[4], target[4])
        loss /= self.ce_num
        # loss +=  self.ce_loss(pred[5], target[5])
        # loss +=  self.ce_loss(pred[6], target[6])
        # loss +=  self.ce_loss(pred[7], target[7])
        # loss +=  self.ce_loss(pred[8], target[8])
        loss_ce = loss.item()
        loss_l1 =  self.loss_weight * self.l1_loss(pred[self.ce_num], target[self.ce_num])
        ori_err1 = torch.abs(pred[self.ce_num + 1] - target[self.ce_num + 1]).mean(dim=1)
        ori_err2 = torch.abs(pred[self.ce_num + 1] + target[self.ce_num + 1]).mean(dim=1)
        ori_loss = torch.where(ori_err1 < ori_err2, ori_err1, ori_err2)
        loss_l1 +=  self.loss_weight * ori_loss.mean()
        loss += loss_l1
        loss_l1_val = loss_l1.item()
        return loss, {'CE': loss_ce, 'L1': loss_l1_val}

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()

        if self.new_epoch_trigger:
            self.lr_scheduler.step()
            self.new_epoch_trigger = False

        img, target = self.data
        img = img.to(self.device)
        target = [t.to(self.device) for t in target]

        preds = self.model(img)

        loss, stats = self.loss_function(preds, target)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), stats

    def validate(self, loader):
        self.set_test()
        stats = {
            'err_plus_pose_err': 0,
            'acc_name': 0,
            'acc_shape': 0,
            'acc_material': 0,
            'acc_colour': 0,
            'acc_size': 0,
            # 'acc_in_hand': 0,
            # 'acc_raised': 0,
            # 'acc_approached': 0,
            # 'acc_gripper_over': 0,
            'pos_err': 0,
            'ori_err': 0
        }
        with torch.no_grad():
            for img, target in tqdm(loader):
                img = img.to(self.device)
                target = [t.to(self.device) for t in target]
                preds = self.model(img)
                # print(preds, target)

                for i, p in enumerate(preds[0:self.ce_num]):
                    _, pred_token = torch.max(p, 1)
                    # print(pred_token)
                    correct = (pred_token == target[i]).sum()
                    stats[list(stats.keys())[i + 1]] += correct.item()

                pos_pred = preds[self.ce_num]
                pos_err = torch.abs(pos_pred - target[self.ce_num]).mean(dim=1)
                stats['pos_err'] += pos_err.sum().item()
                ori_pred = preds[self.ce_num + 1]
                ori_err1 = torch.abs(ori_pred - target[self.ce_num + 1]).mean(dim=1)
                ori_err2 = torch.abs(ori_pred + target[self.ce_num + 1]).mean(dim=1)
                ori_err = torch.where(ori_err1 < ori_err2, ori_err1, ori_err2)
                stats['ori_err'] += ori_err.sum().item()

        acc_avg = 0
        for k in stats.keys():
            if 'acc' in k:
                stats[k] = stats[k] / len(loader.dataset)
                acc_avg += stats[k]
        acc_avg = acc_avg / self.ce_num

        stats['pos_err'] = stats['pos_err'] / len(loader.dataset)
        stats['ori_err'] = stats['ori_err'] / len(loader.dataset)
        
        stats['err_plus_pose_err'] = 0.5 - 0.5 * acc_avg + 0.25 * stats['pos_err'] + 0.25 * stats['ori_err']

        return stats

    def get_scene(self, image, segmentation, scene_gt):
        return None

    def get_scene_pose(self, image, segmentation, scene_gt, return_bboxes=False):
        self.set_test()
        if self.transforms is None:
            import robosuite.utils.transform_utils as transf
            from dataset.ns_ap import CLASS_TO_ID, SHAPE_TO_ID, MATERIAL_TO_ID, COLOUR_TO_ID
            self.transforms = []
            self.id_to_class = {v: k for k, v in CLASS_TO_ID.items()}
            self.id_to_shape = {v: k for k, v in SHAPE_TO_ID.items()}
            self.id_to_material = {v: k for k, v in MATERIAL_TO_ID.items()}
            self.id_to_colour = {v: k for k, v in COLOUR_TO_ID.items()}
            self.transforms.append(T.ToTensor())
            self.transforms.append(T.Resize((448, 448)))
            self.transforms = T.Compose(self.transforms)
        with torch.no_grad():
            img = np.asarray(image)
            # T.ToPILImage()(self.transforms(img)).save('test/asd.png')
            objects = []
            a = 0
            for mask in segmentation:
                # print(mask.sum())
                # print(img.shape)
                mask_arr = mask.squeeze().to(torch.uint8).cpu().numpy()
                # print(mask_arr.shape)
                # print((img * np.expand_dims(mask_arr, 2)).shape)
                # img_inp = np.concatenate((img, img), axis=0)
                img_inp = np.concatenate((img, img * np.expand_dims(mask_arr, 2)), axis=0)
                # print(img_inp.shape)
                # T.ToPILImage()(self.transforms(img * np.expand_dims(mask_arr, 2))).save('test/asd.png')
                img_inp = self.transforms(img_inp).to(self.device)
                # print(img_inp.shape)
                # exit()
                # T.ToPILImage()(img_inp).save(f'test/asd_{a}.png')
                a += 1
                preds = self.model(img_inp.unsqueeze(0))
                # print(preds[0])
                # print(preds[0].shape)
                _, class_id = torch.max(preds[0], 1)
                class_id = class_id.item()
                # print(class_id)
                if class_id > 0:
                    class_name = self.id_to_class[class_id]
                else:
                    class_name = "<NULL>"

                _, shape_id = torch.max(preds[1], 1)
                shape_id = shape_id.item()
                shape_name = self.id_to_shape[shape_id]

                _, material_id = torch.max(preds[2], 1)
                material_id = material_id.item()
                material_name = self.id_to_material[material_id]

                _, colour_id = torch.max(preds[3], 1)
                colour_id = colour_id.item()
                colour_name = self.id_to_colour[colour_id]

                _, scale_id = torch.max(preds[4], 1)
                scale_id = scale_id.item()
                if scale_id == 1:
                    scale_factor = 0.6
                else:
                    scale_factor = 0.45

                position = preds[5].squeeze().cpu().numpy()
                # print(position)
                orientation = preds[6].squeeze().cpu().numpy()

                objects.append(
                    {
                        'name': class_name,
                        'shape': shape_name,
                        'material': material_name,
                        'colour': colour_name,
                        '3d_coords': position,
                        'orientation': orientation
                    }
                )

        if return_bboxes:
            bboxes = []
            positions = [np.array(o['3d_coords']) for o in scene_gt['objects']]
            for obj in objects:
                diffs = [np.abs(obj['3d_coords'] - p).sum() for p in positions]
                min_idx = diffs.index(min(diffs))
                match = scene_gt['objects'][min_idx]
                bbox = match['bbox']
                bbox = self._get_local_bounding_box(bbox)
                bboxes.append(bbox)
        else:
            bboxes = None

        poses = [
            (o['3d_coords'], o['orientation']) for o in objects
        ]

        scene = objects

        return poses, bboxes, scene

    def _get_local_bounding_box(self, bbox):
        bbox_wh = bbox['x'] / 2
        bbox_dh = bbox['y'] / 2
        bbox_h = bbox['z']

        # Local bounding box w.r.t to local (0,0,0) - mid bottom
        return np.array(
            [
                [ bbox_wh,  bbox_dh, 0],
                [-bbox_wh,  bbox_dh, 0],
                [-bbox_wh, -bbox_dh, 0],
                [ bbox_wh, -bbox_dh, 0],
                [ bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh, -bbox_dh, bbox_h],
                [ bbox_wh, -bbox_dh, bbox_h]
            ]
        )

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        #TODO it saves in validation
        if self.optimiser is not None:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, path)
        self.model = self.model.to(self.device)

    def load_checkpoint(self, path, zero_train=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimiser is not None:
            if self.model.training and not zero_train:
                self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        if self.lr_scheduler is not None:
            if self.model.training and not zero_train:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        if epoch is not None:
            self.epoch = epoch
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def set_train(self):
        if not self.model.training:
            self.model.train()

    def set_test(self):
        if self.model.training:
            self.model.eval()

    def set_input(self, data):
        self.data = data

    def unset_input(self):
        self.data = None

    def set_epoch_number(self, epoch):
        if self.epoch != epoch:
            self.new_epoch_trigger = True
            self.epoch = epoch

    def __str__(self):
        return "Baseline Attributes ResNet"


class AttributesBaselineModel(nn.Module):
    def __init__(self, params={}) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.linear_head = nn.Sequential(
            nn.PReLU(),
            nn.Linear(1000, 250),
            nn.PReLU()
        )
        if 'ce_num' in params and params['ce_num'] == 5:
                self.heads = nn.ModuleList(
                    [
                        nn.Linear(250, 13),
                        nn.Linear(250, 5),
                        nn.Linear(250, 6),
                        nn.Linear(250, 12),
                        nn.Linear(250, 2),
                        nn.Linear(250, 3),
                        nn.Linear(250, 4),
                    ]
                )
        else:
            self.heads = nn.ModuleList(
                [
                    nn.Linear(250, 13),
                    nn.Linear(250, 5),
                    nn.Linear(250, 6),
                    nn.Linear(250, 12),
                    nn.Linear(250, 2),
                    nn.Linear(250, 2),
                    nn.Linear(250, 2),
                    nn.Linear(250, 2),
                    nn.Linear(250, 2),
                    nn.Linear(250, 3),
                    nn.Linear(250, 4),
                ]
            )

    def forward(self, inp):
        body = self.model(inp)
        lin_head = self.linear_head(body)
        preds = [
            head(lin_head) for head in self.heads
        ]
        return preds


class YCBAttributesTrainer(VisualRecognitionLoader):
    def __init__(self, model_params, loss_params, optimiser_params, scheduler_params) -> None:
        self.model = AttributesBaselineModel(model_params)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = 0.005
        weight_decay = 0
        if 'lr' in optimiser_params:
            lr = scheduler_params['lr']
        if 'weight_decay' in optimiser_params:
            weight_decay = scheduler_params['weight_decay']
        self.optimiser = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

        self.epoch = 0
        self.new_epoch_trigger = False
        if 'starting_epoch' in scheduler_params:
            self.epoch = scheduler_params['starting_epoch']
        scheduler_step = 1
        if 'scheduler_step' in scheduler_params:
            scheduler_step = scheduler_params['scheduler_step']
        gamma = 0.7
        if 'gamma' in scheduler_params:
            gamma = scheduler_params['gamma']

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=scheduler_step,
            gamma=gamma
        )

        self.loss_weight = 1.0
        if 'loss_weight' is loss_params:
            self.loss_weight = loss_params['loss_weight']
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.loss_function = self.loss_fn

        self.program_idx_to_token = None
        self.ce_num = 5
        if 'ce_num' in model_params:
            self.ce_num = model_params['ce_num']

        self.transforms = None


    def loss_fn(self, pred, target):
        # print()
        # print(pred, target)
        loss = self.ce_loss(pred[0], target[0])
        loss +=  self.ce_loss(pred[1], target[1])
        loss +=  self.ce_loss(pred[2], target[2])
        loss +=  self.ce_loss(pred[3], target[3])
        loss +=  self.ce_loss(pred[4], target[4])
        loss /= self.ce_num
        # loss +=  self.ce_loss(pred[5], target[5])
        # loss +=  self.ce_loss(pred[6], target[6])
        # loss +=  self.ce_loss(pred[7], target[7])
        # loss +=  self.ce_loss(pred[8], target[8])
        loss_ce = loss.item()
        loss_l1 =  self.loss_weight * self.l1_loss(pred[self.ce_num], target[self.ce_num])
        ori_err1 = torch.abs(pred[self.ce_num + 1] - target[self.ce_num + 1]).mean(dim=1)
        ori_err2 = torch.abs(pred[self.ce_num + 1] + target[self.ce_num + 1]).mean(dim=1)
        ori_loss = torch.where(ori_err1 < ori_err2, ori_err1, ori_err2)
        loss_l1 +=  self.loss_weight * ori_loss.mean()
        loss += loss_l1
        loss_l1_val = loss_l1.item()
        return loss, {'CE': loss_ce, 'L1': loss_l1_val}

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()

        if self.new_epoch_trigger:
            self.lr_scheduler.step()
            self.new_epoch_trigger = False

        img, target = self.data
        img = img.to(self.device)
        target = [t.to(self.device) for t in target]

        preds = self.model(img)

        loss, stats = self.loss_function(preds, target)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), stats

    def validate(self, loader):
        self.set_test()
        stats = {
            'err_plus_pose_err': 0,
            'acc_name': 0,
            'acc_shape': 0,
            'acc_material': 0,
            'acc_colour': 0,
            'acc_size': 0,
            # 'acc_in_hand': 0,
            # 'acc_raised': 0,
            # 'acc_approached': 0,
            # 'acc_gripper_over': 0,
            'pos_err': 0,
            'ori_err': 0
        }
        with torch.no_grad():
            for img, target in tqdm(loader):
                img = img.to(self.device)
                target = [t.to(self.device) for t in target]
                preds = self.model(img)
                # print(preds, target)

                for i, p in enumerate(preds[0:self.ce_num]):
                    _, pred_token = torch.max(p, 1)
                    # print(pred_token)
                    correct = (pred_token == target[i]).sum()
                    stats[list(stats.keys())[i + 1]] += correct.item()

                pos_pred = preds[self.ce_num]
                pos_err = torch.abs(pos_pred - target[self.ce_num]).mean(dim=1)
                stats['pos_err'] += pos_err.sum().item()
                ori_pred = preds[self.ce_num + 1]
                ori_err1 = torch.abs(ori_pred - target[self.ce_num + 1]).mean(dim=1)
                ori_err2 = torch.abs(ori_pred + target[self.ce_num + 1]).mean(dim=1)
                ori_err = torch.where(ori_err1 < ori_err2, ori_err1, ori_err2)
                stats['ori_err'] += ori_err.sum().item()

        acc_avg = 0
        for k in stats.keys():
            if 'acc' in k:
                stats[k] = stats[k] / len(loader.dataset)
                acc_avg += stats[k]
        acc_avg = acc_avg / self.ce_num

        stats['pos_err'] = stats['pos_err'] / len(loader.dataset)
        stats['ori_err'] = stats['ori_err'] / len(loader.dataset)
        
        stats['err_plus_pose_err'] = 0.5 - 0.5 * acc_avg + 0.25 * stats['pos_err'] + 0.25 * stats['ori_err']

        return stats

    def get_scene(self, image, segmentation, scene_gt):
        return None

    def get_scene_pose(self, image, segmentation, scene_gt, return_bboxes=False):
        self.set_test()
        if self.transforms is None:
            import robosuite.utils.transform_utils as transf
            from dataset.ns_ap import CLASS_TO_ID, SHAPE_TO_ID, MATERIAL_TO_ID, COLOUR_TO_ID
            self.transforms = []
            self.id_to_class = {v: k for k, v in CLASS_TO_ID.items()}
            self.id_to_shape = {v: k for k, v in SHAPE_TO_ID.items()}
            self.id_to_material = {v: k for k, v in MATERIAL_TO_ID.items()}
            self.id_to_colour = {v: k for k, v in COLOUR_TO_ID.items()}
            self.transforms.append(T.ToTensor())
            self.transforms.append(T.Resize((448, 448)))
            self.transforms = T.Compose(self.transforms)
        with torch.no_grad():
            img = np.asarray(image)
            # T.ToPILImage()(self.transforms(img)).save('test/asd.png')
            objects = []
            a = 0
            for mask in segmentation:
                # print(mask.sum())
                # print(img.shape)
                mask_arr = mask.squeeze().to(torch.uint8).cpu().numpy()
                # print(mask_arr.shape)
                # print((img * np.expand_dims(mask_arr, 2)).shape)
                # img_inp = np.concatenate((img, img), axis=0)
                img_inp = np.concatenate((img, img * np.expand_dims(mask_arr, 2)), axis=0)
                # print(img_inp.shape)
                # T.ToPILImage()(self.transforms(img * np.expand_dims(mask_arr, 2))).save('test/asd.png')
                img_inp = self.transforms(img_inp).to(self.device)
                # print(img_inp.shape)
                # exit()
                # T.ToPILImage()(img_inp).save(f'test/asd_{a}.png')
                a += 1
                preds = self.model(img_inp.unsqueeze(0))
                # print(preds[0])
                # print(preds[0].shape)
                _, class_id = torch.max(preds[0], 1)
                class_id = class_id.item()
                # print(class_id)
                if class_id > 0:
                    class_name = self.id_to_class[class_id]
                else:
                    class_name = "<NULL>"

                _, shape_id = torch.max(preds[1], 1)
                shape_id = shape_id.item()
                shape_name = self.id_to_shape[shape_id]

                _, material_id = torch.max(preds[2], 1)
                material_id = material_id.item()
                material_name = self.id_to_material[material_id]

                _, colour_id = torch.max(preds[3], 1)
                colour_id = colour_id.item()
                colour_name = self.id_to_colour[colour_id]

                _, scale_id = torch.max(preds[4], 1)
                scale_id = scale_id.item()
                if scale_id == 1:
                    scale_factor = 0.6
                else:
                    scale_factor = 0.45

                position = preds[5].squeeze().cpu().numpy()
                # print(position)
                orientation = preds[6].squeeze().cpu().numpy()

                objects.append(
                    {
                        'name': class_name,
                        'shape': shape_name,
                        'material': material_name,
                        'colour': colour_name,
                        '3d_coords': position,
                        'orientation': orientation
                    }
                )

        if return_bboxes:
            bboxes = []
            positions = [np.array(o['3d_coords']) for o in scene_gt['objects']]
            for obj in objects:
                diffs = [np.abs(obj['3d_coords'] - p).sum() for p in positions]
                min_idx = diffs.index(min(diffs))
                match = scene_gt['objects'][min_idx]
                bbox = match['bbox']
                bbox = self._get_local_bounding_box(bbox)
                bboxes.append(bbox)
        else:
            bboxes = None

        poses = [
            (o['3d_coords'], o['orientation']) for o in objects
        ]

        scene = objects

        return poses, bboxes, scene

    def _get_local_bounding_box(self, bbox):
        bbox_wh = bbox['x'] / 2
        bbox_dh = bbox['y'] / 2
        bbox_h = bbox['z']

        # Local bounding box w.r.t to local (0,0,0) - mid bottom
        return np.array(
            [
                [ bbox_wh,  bbox_dh, 0],
                [-bbox_wh,  bbox_dh, 0],
                [-bbox_wh, -bbox_dh, 0],
                [ bbox_wh, -bbox_dh, 0],
                [ bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh, -bbox_dh, bbox_h],
                [ bbox_wh, -bbox_dh, bbox_h]
            ]
        )

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        #TODO it saves in validation
        if self.optimiser is not None:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, path)
        self.model = self.model.to(self.device)

    def load_checkpoint(self, path, zero_train=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimiser is not None:
            if self.model.training and not zero_train:
                self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        if self.lr_scheduler is not None:
            if self.model.training and not zero_train:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        if epoch is not None:
            self.epoch = epoch
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def set_train(self):
        if not self.model.training:
            self.model.train()

    def set_test(self):
        if self.model.training:
            self.model.eval()

    def set_input(self, data):
        self.data = data

    def unset_input(self):
        self.data = None

    def set_epoch_number(self, epoch):
        if self.epoch != epoch:
            self.new_epoch_trigger = True
            self.epoch = epoch

    def __str__(self):
        return "Baseline Attributes ResNet"