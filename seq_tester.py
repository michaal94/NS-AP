from model.rnn_blocks import BaselineLSTMEncoder, BaselineLSTMDecoder
from dataset.ns_ap_instructions import InstructionsNSAP
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


params = {
    "path": "/media/m2data/NS_AP/NS_AP_v1_0_a",
    # "subtasks": ['move_weight'],
    'preprocess': False,
    'split': 'train'
}

ds = InstructionsNSAP(params)

loader = DataLoader(
    dataset=ds,
    batch_size=2,
    shuffle=True,
    num_workers=4
)

params = {
    'encoder_vocab_size': 80,
    'decoder_vocab_size': 50,
    'start_token': 0,
    'end_token': 2
}

enc = BaselineLSTMEncoder(params)
dec = BaselineLSTMDecoder(params)

data = next(iter(loader))

eo, eh = enc(data[0])

do, dh = dec(data[1], eo, eh)
print(data[1].shape, do.shape)

print(dec.sample(torch.zeros((2, 1), dtype=torch.long), eo, eh, 2))

# maxcurr = 0

# for data in loader:
#     match_matrix = (data[1] == 2)
#     if torch.any(match_matrix.sum(dim=1) != 1):
#         print('Incorrect end token')
#     indices = match_matrix.nonzero()[:, 1]
#     lengths = indices + 1
#     if torch.max(lengths).item() > maxcurr:
#         maxcurr = torch.max(lengths).item()


# print(maxcurr)