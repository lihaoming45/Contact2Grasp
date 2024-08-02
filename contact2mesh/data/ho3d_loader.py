import torch
import time
import pickle
import cv2
import numpy as np
import torch.nn.functional as F
from pytorch3d import ops, transforms
from torch.utils.data import Dataset

class HO3DDataset(Dataset):
    def __init__(self, data, d_step=None, is_rot=False, is_trans=False, is_scale=False, is_train=False):
        self.is_rot = is_rot
        self.is_trans = is_trans

        self.is_scale = is_scale
        self.is_train = is_train

        start_time = time.time()
        if isinstance(data, str):
            dataset = pickle.load(open(data, 'rb'))  # Load pickle, can take many seconds
        else:
            dataset = data
        if d_step:
            d_idx = np.array(range(0, len(dataset), d_step))
            dataset = dataset[d_idx]
        self.dataset = dataset


        print('Dataset loaded in {:.2f} sec, {} samples'.format(time.time() - start_time, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        out = dict()
        obj_sampled_verts_gt = sample['obj_sampled_verts_gt']
        obj_verts_ori = sample['obj_verts_ori']

        if self.is_rot:
            if 'objRot' not in sample.keys():
                objRot, objTrans = self.get_tform(add_trans=self.is_trans)
                obj_sampled_verts_gt = np.matmul(obj_sampled_verts_gt, objRot.clone().T)+objTrans.clone()
                obj_verts_ori = np.matmul(obj_verts_ori, objRot.clone().T)+objTrans.clone()
            else:
                obj_sampled_verts_gt = np.matmul(obj_sampled_verts_gt, cv2.Rodrigues(sample['objRot'])[0].T)

        if self.is_trans:
            obj_sampled_verts_gt = self.aug_trans(obj_sampled_verts_gt)

        if self.is_scale:
            scale = self.get_diameter(obj_sampled_verts_gt)
            obj_sampled_verts_gt*=scale
            obj_verts_ori*=scale

        # sample['obj_sampled_verts_gt'] = obj_sampled_verts_gt
        if 'gen_contact' in sample.keys():
            out['gen_contact'] = torch.Tensor(sample['gen_contact'])

        if self.is_train is False:
            out['obj_verts_ori'] = obj_verts_ori
            out['obj_faces_ori'] = sample['obj_faces_ori']

        out['obj_sampled_verts_gt'] = torch.Tensor(obj_sampled_verts_gt)

        return out


    # def aug_trans(self, obj_xyz, hand_xyz=None):
    #     N = obj_xyz.size(0)
    #     random_trans = 0.04 * (torch.rand((1, 3)) - 0.5)  # range [-2cm, 2cm]
    #
    #     random_trans_obj = random_trans.repeat(N, 1)
    #     if hand_xyz is not None:
    #         random_trans_hand = random_trans.repeat(778, 1)
    #         return obj_xyz + random_trans_obj, hand_xyz + random_trans_hand
    #     return obj_xyz + random_trans

    def aug_trans(self, obj_xyz, hand_xyz=None):
        N = obj_xyz.shape[0]
        random_trans = 0.01 * (np.random.random((1, 3)) - 0.5)  # range [-1cm, 1cm]

        # random_trans_obj = random_trans.repeat(N, 1)
        if hand_xyz is not None:
            # random_trans_hand = random_trans.repeat(778, 1)
            return obj_xyz + random_trans, hand_xyz + random_trans
        return obj_xyz + random_trans

    def get_diameter(self,vp):
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
        diameter_x = abs(x_max - x_min)
        diameter_y = abs(y_max - y_min)
        diameter_z = abs(z_max - z_min)
        diameter = np.sqrt(diameter_x ** 2 + diameter_y ** 2 + diameter_z ** 2)
        return diameter


    @staticmethod
    def get_tform(add_trans=False):
        """
        Find a 4x4 rigid transform to normalize the pointcloud. We choose the object center of mass to be the origin,
        the hand center of mass to be along the +X direction, and the rotation around this axis to be random.
        :param hand_verts: (batch, 778, 3)
        :param obj_verts: (batch, 2048, 3)
        :return: tform: (batch, 4, 4)
        """
        ttrans = torch.tensor([0.])

        rot_angles = np.random.random(3) * np.pi * 2
        theta_x, theta_y, theta_z = rot_angles[0], rot_angles[1], rot_angles[2]
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        rot = Rx @ Ry @ Rz  # [3, 3]

        # generate random translation

        # trans = torch.tensor([-0.0793, 0.0208, -0.6924]) + torch.rand(3) * 0.1
        trans = 0.04 * (torch.rand((1, 3)) - 0.5)  # range [-2cm, 2cm]
        trans = trans.reshape((1,3))

        # trot = ops.eyes(3)
        trot = rot

        if add_trans:
            ttrans= trans

        return trot.float(),ttrans
