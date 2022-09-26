from torch.utils.data import DataLoader

from .ns_ap_segmentation import SegmentationNSAP
from .ns_ap_instructions import InstructionsNSAP
from .ns_ap_attributes import AttributesNSAP
from .ns_ap_actions import ActionPlanNSAP

NAME2CLASS = {
    "SegmentationNSAP": SegmentationNSAP,
    "InstructionsNSAP": InstructionsNSAP,
    "AttributesNSAP": AttributesNSAP,
    "ActionPlanNSAP": ActionPlanNSAP
}

def get_dataset(params):
    return NAME2CLASS[params['name']](params)

def get_dataloader(params):
    dataset = get_dataset(params)
    print(len(dataset))
    shuffle = False
    assert 'split' in params
    if 'shuffle' in params:
        shuffle = params['shuffle']
    else:
        if params['split'] == 'train':
            shuffle = True
    assert 'batch' in params
    batch = params['batch']
    num_workers = 0
    if 'num_workers' in params:
        num_workers = params['num_workers']
    if 'Segmentation' in params['name']:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=segmentation_collate
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=num_workers
        )
    print("Loaded {} dataset, split: {} number of samples: {}".format(
        str(dataset),
        params['split'],
        len(dataset)
    ))

    return loader

def segmentation_collate(batch):
    return tuple(zip(*batch))