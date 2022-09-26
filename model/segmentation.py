import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils.coco as coco
from tqdm import tqdm
import torchvision.transforms as T
from dataset.ns_ap import CLASS_TO_ID


class BaselineSegmentationTrainer():
    def __init__(self, model_params, loss_params, optimiser_params, scheduler_params) -> None:
        # Get model from torchvision
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 13)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            13
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = 0.005
        momentum = 0.9
        weight_decay = 0.0005
        if 'lr' in optimiser_params:
            lr = scheduler_params['lr']
        if 'momentum' in optimiser_params:
            momentum = scheduler_params['momentum']
        if 'weight_decay' in optimiser_params:
            weight_decay = scheduler_params['weight_decay']
        self.optimiser = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )


        self.epoch = 0
        self.new_epoch_trigger = False
        if 'starting_epoch' in scheduler_params:
            self.epoch = scheduler_params['starting_epoch']
        scheduler_step = 3
        if 'scheduler_step' in scheduler_params:
            scheduler_step = scheduler_params['scheduler_step']
        gamma = 0.1
        if 'gamma' in scheduler_params:
            gamma = scheduler_params['gamma']

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=scheduler_step,
            gamma=gamma
        )

        self.warmup_scheduler = None

        self.dataloader_len = 1000

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()
        if self.epoch == 0:
            if self.warmup_scheduler is None:
                self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimiser, 
                    start_factor=1e-3,
                    total_iters=min(1000, self.dataloader_len)
                )
        else:
            self.warmup_scheduler = None

        if self.new_epoch_trigger:
            self.lr_scheduler.step()
            self.new_epoch_trigger = False

        images, targets = self.data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_sum_val = loss.item()
        loss_dict_val = {k: v.item() for k, v in loss_dict.items()}

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if self.warmup_scheduler is not None:
            self.warmup_scheduler.step()

        return loss_sum_val, loss_dict_val

    def validate(self, loader):
        aps_per_img = []
        with torch.no_grad():
            for imgs, targets in tqdm(loader):
                imgs = list(img.to(self.device) for img in imgs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(imgs)
                for i in range(len(targets)):
                    gt = targets[i]
                    pred = outputs[i]
                    gt_boxes = gt['boxes']
                    gt_masks = gt['masks']
                    gt_labels = gt['labels']
                    pred_boxes = pred['boxes']
                    pred_labels = pred['labels']
                    pred_scores = pred['scores']
                    pred_masks = pred['masks']

                    # fake_scores = torch.ones((gt_boxes.shape[0]))
                    # for i in range(gt_boxes.shape[0]):
                    #     fake_scores[i] -= (0.1 - i*0.01)

                    mAP, _, _, _ = coco.compute_ap(
                        gt_boxes, gt_labels, gt_masks,
                        pred_boxes, pred_labels, pred_masks, pred_scores               
                    )
                    aps_per_img.append(mAP)
        
        return torch.tensor(aps_per_img).mean().item()

    def get_segmenation(self, image):
        self.set_test()
        with torch.no_grad():
            image = T.ToTensor()(image)
            img_list = [image.to(self.device)]
            outputs = self.model(img_list)
            masks = outputs[0]['masks']
            labels = outputs[0]['labels'].tolist()
            if CLASS_TO_ID['robot'] in labels or CLASS_TO_ID['table'] in labels:
                mask_items = []
                labels_items = []
                for i, lab in enumerate(labels):
                    if lab != CLASS_TO_ID['robot'] and lab != CLASS_TO_ID['table']:
                        mask_items.append(masks[i])
                        labels_items.append(lab)
                masks = mask_items
                labels = labels_items
            for mask in masks:
                mask[mask > 0.5] = 1
                mask[mask < 1] = 0
                mask = mask.int()

        return masks, labels

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        if self.model.training:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
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
        return "Baseline Segmentation Mask R-CNN"

