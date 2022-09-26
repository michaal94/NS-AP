from dataset.ns_ap_segmentation import SegmentationNSAP
from torch.utils.data import DataLoader

def segmentation_collate(batch):
    return tuple(zip(*batch))

params = {
    # "path": "/media/m2data/NS_AP/NS_AP_v1_0_a",
    "path": "/home/michal/datasets/NS_AP_v1_0_a",
    "subtasks": ['stack'],
    'split': 'val',
    "coco_annotation": True
}

ds = SegmentationNSAP(params)
# print(len(ds))
# # ds.__getitem__(0)
# print(ds.__getitem__(0))
# print(ds.__getitem__(100))
# for i in range(len(ds)):
# for i in range(788, 792):
#     print(i)
#     ds.__getitem__(i)
loader = DataLoader(
            dataset=ds,
            batch_size=2,
            shuffle=False,
            num_workers=8,
            collate_fn=segmentation_collate
        )

batch = next(iter(loader))
# print(batch[0].shape)