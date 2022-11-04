import numpy as np

def check_eef_in_bbox(bbox, eef_pos):
    x_vec = bbox[0] - bbox[1]
    y_vec = bbox[0] - bbox[3]
    z_vec = bbox[4] - bbox[0]
    x_size = np.linalg.norm(x_vec)
    y_size = np.linalg.norm(y_vec)
    z_size = np.linalg.norm(z_vec)
    # print(bbox.shape)
    bbox_mid = np.mean(bbox, axis=0)
    mid_to_eef_vec = eef_pos - bbox_mid
    axes_in = []
    x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
    if x_proj < x_size / 2:
        axes_in.append('x')
    y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
    if y_proj < y_size / 2:
        axes_in.append('y')
    z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
    if z_proj < z_size / 2:
        axes_in.append('z')
    return axes_in 

bbox = np.array(
    [[ 0.51780815, -0.04068139,  0.11670223],
       [ 0.56214407, -0.01338788,  0.10959042],
       [ 0.57069771, -0.00717521,  0.18675762],
       [ 0.52636179, -0.03446872,  0.19386944],
       [ 0.54471126, -0.08424627,  0.11722752],
       [ 0.58904718, -0.05695276,  0.1101157 ],
       [ 0.59760082, -0.05074009,  0.18728291],
       [ 0.5532649 , -0.0780336 ,  0.19439472]]
)
eef = np.array([ 0.51291183, -0.05929338, 0.11359484])

print(check_eef_in_bbox(bbox, eef))