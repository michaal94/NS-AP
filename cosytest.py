import zmq
import pickle
import numpy as np
from model.ycb_data import COSYPOSE2NAME, COSYPOSE_BBOX, COSYPOSE_TRANSFORM_DEFAULT, CAMERA_DATA
import robosuite.utils.transform_utils as T

class Tester:
    def __init__(self) -> None:
        pass

    def _request_cosypose_detection(self, img_path):
        request = {
            'img_path': img_path,
            'camera_k': CAMERA_DATA['intrinsic']
        }
        request_msg = pickle.dumps(request)
        self._socket_cosypose.send(request_msg)
        print(f"Cosypose detection requested for {img_path}")
        msg = self._socket_cosypose.recv()
        msg = pickle.loads(msg)
        print("Cosypose detection received")
        # print(msg)
        # exit()
        names, poses, bboxes = [], [], []
        for (label, pose) in msg:
            # print(label)
            # print(pose)
            # continue
            name = COSYPOSE2NAME[label]
            names.append(name)
            pose_in_base = np.matmul(self.camera_pose, pose)
            # print(name)
            # print(pose_in_base)
            # print(T.make_pose(COSYPOSE_TRANSFORM[label][0],
            #         T.quat2mat(COSYPOSE_TRANSFORM[label][1])
            #     ))
            pose_in_base = np.matmul(
                pose_in_base,
                T.make_pose(COSYPOSE_TRANSFORM_DEFAULT[label][0],
                    T.quat2mat(COSYPOSE_TRANSFORM_DEFAULT[label][1])
                )
            )
            pose_in_base[2, 3] = max(pose_in_base[2, 3], 0.0)


            bbox_xyz = COSYPOSE_BBOX[label]
            bbox_local = self._get_local_bounding_box(bbox_xyz)
            bbox_local = np.concatenate(
                (
                    bbox_local.T,
                    np.ones((bbox_local.shape[0], 1)).T
                )
            )
            # print(pose_mat)
            bbox_world = np.matmul(pose_in_base, bbox_local)
            poses.append(T.mat2pose(pose_in_base))
            bboxes.append(bbox_world[:-1, :].T)
            print(name)
            print(poses[-1])
            # print(pos_cosy, pos)
            # print(ori_cosy, ori)

    def _get_local_bounding_box(self, bbox):
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

    def _setup_communication(self):
        context = zmq.Context()
        self._socket_cosypose = context.socket(zmq.REQ)
        self._socket_cosypose.connect("tcp://127.0.0.1:5554")
        cam_pos = CAMERA_DATA['extrinsic']['pos']
        cam_ori = CAMERA_DATA['extrinsic']['ori']
        # Rot around x 180d (robosuite is xyzw)
        blender_to_cosy = np.array([1.0, 0, 0, 0])
        cam_ori = T.quat2mat(T.quat_multiply(T.convert_quat(cam_ori, to='xyzw'), blender_to_cosy))
        self.camera_pose = T.make_pose(cam_pos, cam_ori)
        print(self.camera_pose)

test = Tester()
test._setup_communication()
test._request_cosypose_detection("/home/michas/Desktop/codes/NS-AP/output_ycb/NS_AP_demo_000000/frame_0000.png")


