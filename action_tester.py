from model.action_planner import BaselineActionTrainer
from dataset.ns_ap import GOAL_VOCAB, ACTION_VOCAB
from dataset.ns_ap_actions import ActionPlanNSAP
from torch.utils.data import DataLoader

params = {
    "path": "/media/m2data/NS_AP/NS_AP_v1_0_a",
    "split": 'test',
    "preprocess": False,
}

token_to_goal = {v: k for k, v in GOAL_VOCAB.items()}
token_to_action = {v: k for k, v in ACTION_VOCAB.items()}

trainer = BaselineActionTrainer()
trainer.set_test()
trainer.load_checkpoint("./output/checkpoints/action.pt")

ds = ActionPlanNSAP(params)

loader = DataLoader(
    dataset=ds,
    batch_size=1,
    shuffle=True,
    num_workers=4
)
print(len(ds))
for data in loader:
    # print(ds.__getitem__(i))
    # data = ds.__getitem__(i)
    # print(data[0].shape)
    print(data)