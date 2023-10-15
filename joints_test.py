import os
import json
import numpy as np
np.set_printoptions(linewidth=np.inf)

with open('/home/michas/Desktop/codes/NS-AP/output_ycb/NS_AP_demo_000093/sequence.json', 'r') as f:
    seq = json.load(f)['observations_gt']

adjust = np.array([-90, 0, 0, 0, 0, 0, 0])
adjust = np.array([0, 0, 0, 0, 0, 0, 0])
adjust_mul = np.array([-1, 1, 1, 1, 1, 1, 1])
adjust_mul = np.array([1, 1, 1, -1, 1, -1, -1])
for s in seq:
    new = (np.array(s['robot']['robot_joints']) + adjust) * adjust_mul
    # print(np.array(s['robot']['robot_joints']) + adjust)
    print(new)
