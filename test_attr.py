from dataset.ns_ap_attributes import AttributesNSAP
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from model.visual_recognition import AttributesBaselineModel
from tqdm import tqdm 
import torchvision.transforms as T

params = {
    "path": "/media/m2data/NS_AP/NS_AP_v1_0_a",
    "split": 'test',
    "only_selected": True,
}

device = 'cuda:0'

ds = AttributesNSAP(params)
model = AttributesBaselineModel({'ce_num': 5}).to(device)
checkpoint = torch.load('/home/michas/Desktop/codes/NS-AP/output/checkpoint_058000.pt')
model.load_state_dict(checkpoint['state_dict'])

loader = DataLoader(
    dataset=ds,
    batch_size=1,
    shuffle=False,
    num_workers=8
)

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

ce_num = 5

model.eval()
with torch.no_grad():
    for data in tqdm(loader):
        img, target = data
        T.ToPILImage()(img[0, :, :, :]).save('test/asd.png')
        exit()
        img = img.to(device)
        # img = torch.cat([img, img], dim=0)
        target = [t.to(device) for t in target]
        # target = [torch.cat([t, t], dim=0) for t in target]
        preds = model(img)
        # for p in preds:
        #     print(p.shape)
        for i, p in enumerate(preds[0:ce_num]):
            _, pred_token = torch.max(p, 1)
            # print(pred_token, target[i])
            # print(pred_token == target[i])
            correct = (pred_token == target[i]).sum()
            stats[list(stats.keys())[i + 1]] += correct.item()

        pos_pred = preds[ce_num]
        pos_err = torch.abs(pos_pred - target[ce_num]).mean(dim=1)
        stats['pos_err'] += pos_err.sum().item()
        ori_pred = preds[ce_num + 1]
        ori_err1 = torch.abs(ori_pred - target[ce_num + 1]).mean(dim=1)
        ori_err2 = torch.abs(ori_pred + target[ce_num + 1]).mean(dim=1)
        ori_err = torch.where(ori_err1 < ori_err2, ori_err1, ori_err2)
        stats['ori_err'] += ori_err.sum().item()
    # exit()

acc_avg = 0
for k in stats.keys():
    if 'acc' in k:
        stats[k] = stats[k] / len(loader.dataset)
        acc_avg += stats[k]
acc_avg = acc_avg / ce_num

stats['pos_err'] = stats['pos_err'] / len(loader.dataset)
stats['ori_err'] = stats['ori_err'] / len(loader.dataset)

stats['err_plus_pose_err'] = 0.5 - 0.5 * acc_avg + 0.25 * stats['pos_err'] + 0.25 * stats['ori_err']

print(stats)
