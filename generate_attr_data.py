import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from model.ycb_data import CAMERA_DATA
import robosuite.utils.transform_utils as T

# out_dir = './'
in_dir = './output_ycb_images'

def get_local_bounding_box(bbox):
    bbox_wh = bbox['x'] / 2
    bbox_dh = bbox['y'] / 2
    bbox_h = bbox['z']

    # Local bounding box w.r.t to local (0,0,0) - mid bottom
    return np.array(
        [
            [ bbox_wh,  bbox_dh, 0],
            [-bbox_wh,  bbox_dh, 0],
            [-bbox_wh, -bbox_dh, 0],
            [ bbox_wh, -bbox_dh, 0],
            [ bbox_wh,  bbox_dh, bbox_h],
            [-bbox_wh,  bbox_dh, bbox_h],
            [-bbox_wh, -bbox_dh, bbox_h],
            [ bbox_wh, -bbox_dh, bbox_h]
        ]
    )

Kmatrix = CAMERA_DATA['intrinsic']
cam_pos = CAMERA_DATA['extrinsic']['pos']
cam_ori = CAMERA_DATA['extrinsic']['ori']
# Rot around x 180d (robosuite is xyzw)
blender_to_cosy = np.array([1.0, 0, 0, 0])
cam_ori = T.quat2mat(T.quat_multiply(T.convert_quat(cam_ori, to='xyzw'), blender_to_cosy))
cam_in_base = T.make_pose(cam_pos, cam_ori)
Pmatrix = T.pose_inv(cam_in_base)

bname_list = sorted(os.listdir(in_dir))
bname_list = [os.path.splitext(p)[0] for p in bname_list if p.endswith('.png')]

for f_bname in tqdm(bname_list):
    with open(os.path.join(in_dir, f_bname + '.json'), 'r') as f:
        attr = json.load(f)
    img = Image.open(os.path.join(in_dir, f_bname + '.png'))
    d = ImageDraw.Draw(img)
    min_x = 2000
    min_y = 2000
    max_x = 0
    max_y = 0
    bbox = get_local_bounding_box(attr['bbox'])
    for p in bbox:
        print(p)
        bbox_corner = np.ones(4)
        bbox_corner[0:3] = p
        # print(bbox_corner)
        corner_in_cam = np.matmul(Pmatrix, bbox_corner)[0:3]
        img_point = np.matmul(Kmatrix, corner_in_cam)
        img_point = [img_point[0] / img_point[2], img_point[1] / img_point[2]]
        # draw_point = [int(img_point[0]) - 2, int(img_point[1]) - 2, int(img_point[0]) + 2, int(img_point[1]) + 2,]
        # d.ellipse(draw_point, fill = 'blue', outline ='blue')
        min_x = min(min_x, img_point[0])
        min_y = min(min_y, img_point[1])
        max_x = max(max_x, img_point[0])
        max_y = max(max_y, img_point[1])
    d.rectangle([min_x, min_y, max_x, max_y], outline ='blue')
    input()