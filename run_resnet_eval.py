import os
import json
import numpy as np
from model.visual_recognition import YCBAttributesTrainer
from dataset.ycb_attributes import CLASS_TO_ID, MATERIAL_TO_ID, COLOUR_TO_ID, SHAPE_TO_ID
import robosuite.utils.transform_utils as T
from PIL import Image
import sys

task = int(sys.argv[1])
query = int(sys.argv[2])
source = 'sim'
main_dir = './output_ycb_blender'
exp_name = f'NS_AP_demo_{(task * 10 + query):06d}'

Kmatrix = np.array([
    [624.2282702957912, 0.0, 640.0],
    [0.0, 624.2282702957912, 360.0],
    [0.0, 0.0, 1.0]
])
cam_pos = np.array([1.1398999691009521, 0.04690000042319298, 0.2775000035762787])
cam_ori = np.array([0.5590000152587891, 0.4250999987125397, 0.4602999985218048, 0.5430999994277954])
# Rot around x 180d (robosuite is xyzw)
blender_to_cosy = np.array([1.0, 0, 0, 0])
cam_ori = T.quat2mat(T.quat_multiply(T.convert_quat(cam_ori, to='xyzw'), blender_to_cosy))
cam_in_base = T.make_pose(cam_pos, cam_ori)
Pmatrix = T.pose_inv(cam_in_base)

if source == 'sim':
    img_path = os.path.join(main_dir, exp_name, 'frame_0000.png')
else:
    raise NotImplementedError()
json_path = os.path.join(main_dir, exp_name, 'sequence.json')

trainer = YCBAttributesTrainer({}, {}, {}, {})
trainer.load_checkpoint(
    '/media/m2data/NS_AP_outputs/attributes/YCB_031122_161857/checkpoint_best.pt'
)

ID_TO_CLASS = {v: k for (k, v) in CLASS_TO_ID.items()}
ID_TO_MATERIAL = {v: k for (k, v) in MATERIAL_TO_ID.items()}
ID_TO_COLOUR = {v: k for (k, v) in COLOUR_TO_ID.items()}
ID_TO_SHAPE = {v: k for (k, v) in SHAPE_TO_ID.items()}

with open(json_path, 'r') as f:
    if source == 'sim':
        observations = json.load(f)['observations']
    else:
        raise NotImplementedError()

img = Image.open(img_path).convert("RGB")

for obj in observations[0]['objects']:
    min_x = 2000
    min_y = 2000
    max_x = 0
    max_y = 0
    for p in obj['bbox']:
        bbox_corner = np.array(p + [1.0])
        corner_in_cam = np.matmul(Pmatrix, bbox_corner)[0:3]
        img_point = np.matmul(Kmatrix, corner_in_cam)
        img_point = [img_point[0] / img_point[2], img_point[1] / img_point[2]]
        # draw_point = [int(img_point[0]) - 2, int(img_point[1]) - 2, int(img_point[0]) + 2, int(img_point[1]) + 2,]
        # d.ellipse(draw_point, fill = 'blue', outline ='blue')
        min_x = min(min_x, img_point[0])
        min_y = min(min_y, img_point[1])
        max_x = max(max_x, img_point[0])
        max_y = max(max_y, img_point[1])
    crop = img.crop((min_x, min_y, max_x, max_y))
    preds = trainer.get_single_pred(crop)
    print(f'{ID_TO_CLASS[preds[0]]}\t{ID_TO_SHAPE[preds[1]]}\t{ID_TO_MATERIAL[preds[2]]}\t{ID_TO_COLOUR[preds[3]]}')
img.show()