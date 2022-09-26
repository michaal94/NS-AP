'''
Functions adapted from https://github.com/pytorch/vision
'''

import numpy as np
import torch

def compute_ap_thresholds(gt_box, gt_class, gt_mask,
                          pred_box, pred_class, pred_mask, pred_score,
                          iou_min=0.5, iou_max=0.95, iou_step=0.05):
    iou_thresholds = np.arange(iou_min, iou_max + iou_step, iou_step)
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = compute_ap(
            gt_box, gt_class, gt_mask,
            pred_box, pred_class, pred_mask, pred_score,
            iou_threshold=iou_threshold
        )
        AP.append(ap)
    AP = torch.tensor(AP).mean()
    return AP


def compute_ap(gt_box, gt_class, gt_mask,
               pred_box, pred_class, pred_mask, pred_score,
               iou_threshold=0.5):
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_box, gt_class, gt_mask,
        pred_box, pred_class, pred_mask, pred_score,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = torch.cumsum(pred_match > -1, dim=0) / (torch.arange(len(pred_match)) + 1)
    recalls = torch.cumsum(pred_match > -1, dim=0).float() / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = torch.cat((torch.tensor([0]), precisions, torch.tensor([0])))
    recalls = torch.cat((torch.tensor([0]), recalls, torch.tensor([1])))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = torch.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = torch.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_masks, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # gt_boxes = trim_zeros(gt_boxes)
    # gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    # pred_boxes = trim_zeros(pred_boxes)
    # pred_scores = pred_scores[:pred_boxes.shape[0]]

    # Sort predictions by score from high to low
    indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[indices, ...]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * torch.ones([pred_boxes.shape[0]])
    gt_match = -1 * torch.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = torch.argsort(overlaps[i], descending=True)
        # 2. Remove low scores
        low_score_idx = torch.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.shape[0] > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [instances, Height, Width]
    """
    
    # If either set of masks is empty return empty result
    if len(masks1.shape) == 0 or len(masks2.shape) == 0 or masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return torch.zeros((masks1.shape[0], masks2.shape[0]))
    # flatten masks and compute their areas
    masks1 = torch.reshape(masks1 > .5, (masks1.shape[0], -1)).float()
    masks2 = torch.reshape(masks2 > .5, (masks2.shape[0], -1)).float()
    area1 = masks1.sum(dim=1)
    area2 = masks2.sum(dim=1)

    # intersections and union
    intersections = torch.matmul(masks1, masks2.T)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[x.abs().sum(dim=1).bool()]