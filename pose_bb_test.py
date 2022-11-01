import os
import json
import numpy as np
from PIL import Image, ImageDraw
from model.ycb_data import CAMERA_DATA
import robosuite.utils.transform_utils as T

sequence = [3, 2]
frame = 0

Kmatrix = CAMERA_DATA['intrinsic']
cam_pos = CAMERA_DATA['extrinsic']['pos']
cam_ori = CAMERA_DATA['extrinsic']['ori']
# Rot around x 180d (robosuite is xyzw)
blender_to_cosy = np.array([1.0, 0, 0, 0])
cam_ori = T.quat2mat(T.quat_multiply(T.convert_quat(cam_ori, to='xyzw'), blender_to_cosy))
cam_in_base = T.make_pose(cam_pos, cam_ori)
Pmatrix = T.pose_inv(cam_in_base)


dir_name = os.path.join(f'./output_ycb_blender/NS_AP_demo_{(10 * sequence[0] + sequence[1]):06d}')

json_path = os.path.join(dir_name, 'sequence.json')
img_path = os.path.join(dir_name, f'frame_{frame:04d}.png')

with open(json_path, 'r') as f:
    obs = json.load(f)['observations'][frame]


img = Image.open(img_path)
d = ImageDraw.Draw(img)

for obj in obs['objects']:
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
    # print([min_x, min_y, max_x, max_y])
    d.rectangle([min_x, min_y, max_x, max_y], outline ='blue')

    # exit()

img.save('./test/test_bbox.png')