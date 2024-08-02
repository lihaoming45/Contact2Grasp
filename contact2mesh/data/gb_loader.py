
from torch.utils import data
import os
import time
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import contact2mesh.utils.util as util

class GrabNetDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 load_params = False,
                 load_on_ram = False,
                 load_contact=True
                 ):

        super().__init__()

        self.load_params = load_params
        self.load_contact = load_contact
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))

        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()# for object "mesh file"
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item() # for right hand "mesh file"
        ## bps_torch data

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)
        ## Hand vtemps and betas

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch
    def load_disk(self,idx):

        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):# idx:0-31026
        # out = dict()
        data_out = self.load_disk(idx)

        if self.load_params:
            params_out = {k: self.ds[k][idx] for k in self.ds.keys()}
            data_out.update(params_out)
            data_out['pose_rhand_rotmat'] = data_out['fpose_rhand_rotmat']

        data_out['obj_sampled_verts_gt'] = data_out['verts_object']
        if self.load_contact:
            data_out['obj_sampled_contact_gt'] = data_out['contact_sample'].unsqueeze(-1) / 55


        # out['hand_verts_gt'] = data_out['verts_rhand']

        return data_out

    # @staticmethod
    # def collate_fn(batch):
    #     out = dict()
    #     batch_keys = batch[0].keys()
    #     use_keys = ['obj_sampled_contact_gt']
    #     for key in [k for k in batch_keys if k in use_keys]:
    #         out[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])
    #
    #     out['obj_sampled_contact_gt'] = out['obj_sampled_contact_gt'].unsqueeze(-1)
    #
    #
    #     # out['obj_sampled_verts_gt'] = data_out
    #     return batch


if __name__ == '__main__':
    data_path = '/home/dataset/haoming/grabnet/new_data/GRAB_V00'
    # work_dir = "D:\PycharmProjects\GrabNet\grabnet\\tests\\tests\gt"
    rhm_path = '/home/haoming/GrabNet/mano_v1_2/models/MANO_RIGHT.pkl'
    mesh_base = '/home/haoming/GrabNet/tools/object_meshes/contact_meshes'

    ds = GrabNetDataset(data_path, ds_name='test', load_params=False)
    dataloader = data.DataLoader(ds, batch_size=100, shuffle=False, num_workers=0, drop_last=True)

    dl = iter(dataloader)
    for i in range(10):
        a = next(dl)
