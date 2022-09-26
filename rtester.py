from model.visual_recognition import AttributesBaselineModel
# from dataset.ns_ap_instructions import InstructionsNSAP
# from torch.utils.data import DataLoader
# from tqdm import tqdm
import torch


m = AttributesBaselineModel()

out = m(torch.randn(2, 3, 448, 448))

print([o.shape for o in out])

