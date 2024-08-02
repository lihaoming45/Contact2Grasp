import torch
import time
import pickle
import trimesh
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes
import contact2mesh.utils.util as util
from pytorch3d import ops, transforms
from bps_torch.bps import bps_torch
import torch.nn.functional as F
from contact2mesh.utils.util import soft_to_binary,TRAIN_OBJ_NAMES, TEST_OBJ_NAMES,OBJ_NAMES
from contact2mesh.utils.contact_metrics import dbscan_cluster
from contact2mesh.utils.mano_util import batch_rodrigues,aa2rotmat_new,rotmat2aa_new
from visual import pcd_instance, mesh_instance
from pytorch3d import transforms as p3dt

class ContactDBDataset(Dataset):
    """PyTorch Dataset object which allows batched fetching of hand/object pairs from a dataset.
    PyTorch3D Meshes are used to handle batches of variable-size meshes"""
    def __init__(self, data, bps_dir=None, train=False, is_aug=False, is_param=True, is_rot=False):
        start_time = time.time()
        self.train = train
        self.aug_vert_jitter = 0.0005
        self.random_trans=0.
        self.is_aug = is_aug
        self.is_param = is_param
        self.is_rot=is_rot

        self.bps_dir=bps_dir
        if bps_dir:
            self.bps = bps_torch(custom_basis=torch.from_numpy(np.load(bps_dir)['basis']).to(torch.float32))

        if isinstance(data, str):
            self.dataset = pickle.load(open(data, 'rb'))    # Load pickle, can take many seconds
        else:
            self.dataset = data

        # if 'num_verts_in_contact' in self.dataset[0]:
        #     print('Cutting samples with less than {} points in contact. Was size {}'.format(min_num_cont, len(self.dataset)))
        #     self.dataset = [s for s in self.dataset if s['num_verts_in_contact'] >= min_num_cont]
        self.root = torch.tensor([[0.0957, 0.0064, 0.0062]])
        self.joint9 = torch.tensor([0.0010, 0.0049, 0.0028])
        self.root2joint9 = torch.tensor([[-0.0947, -0.0015, -0.0034]])

        print('Dataset loaded in {:.2f} sec, {} samples'.format(time.time() - start_time, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        out = dict()
        if not self.train:
            out['obj_faces'] = torch.Tensor(sample['ho_gt'].obj_faces)
            out['obj_verts_gt'] = torch.Tensor(sample['ho_gt'].obj_verts)
            out['obj_contact_gt'] = torch.Tensor(sample['ho_gt'].obj_contact)

        out['hand_verts_gt'] = torch.Tensor(sample['ho_gt'].hand_verts)

        out['obj_sampled_idx'] = torch.Tensor(sample['obj_sampled_idx']).long()

        out['obj_sampled_verts_gt'] =torch.Tensor(sample['ho_gt'].obj_verts)[out['obj_sampled_idx'], :]
        out['obj_sampled_contact_gt'] =torch.Tensor(sample['ho_gt'].obj_contact)[out['obj_sampled_idx'], :]
        # out['obj_sampled_contact_rec'] = torch.Tensor(sample['obj_sampled_contact_rec'].squeeze())
        # out['obj_sem_contact_gt'] = torch.Tensor(sample['ho_gt'].obj_sem_contact)
        # out['obj_sampled_sem_contact_gt'] = out['obj_sem_contact_gt'][out['obj_sampled_idx'], :]
        # out['obj_feats_gt'] = torch.Tensor(sample['obj_feats_gt'])
        out['obj_class'] = torch.tensor(sample['ho_gt'].obj_class)
        # out['p_num'] = sample['ho_gt'].p_num
        # out['intent'] = sample['ho_gt'].intent
        out['object_name'] = sample['ho_gt'].obj_name


        # mesh = mesh_instance(sample['ho_gt'].obj_verts,sample['ho_gt'].obj_faces)
        # mesh.compute_vertex_normals()
        # pcd = pcd_instance(sample['ho_gt'].obj_verts)
        # pcd.estimate_normals()

        out['obj_sampled_normals_gt'] = torch.tensor(sample['ho_gt'].obj_normals)[out['obj_sampled_idx'], :]
        out['obj_sampled_normals_gt'] = out['obj_sampled_normals_gt']/out['obj_sampled_normals_gt'].norm(dim=-1,keepdim=True)
        # tgt_mesh = trimesh.Trimesh(vertices = sample['ho_gt'].obj_verts, faces = sample['ho_gt'].obj_faces)
        # gt_surface_pts, face_index = trimesh.sample.sample_surface_even(tgt_mesh, 2048)
        # out['obj_sampled_verts_gt'] = torch.tensor(np.array(gt_surface_pts))

        if 'contact_latent' in sample.keys():
            out['contact_latent'] = torch.Tensor(sample['contact_latent'])

        if self.is_rot:
            out['obj_sampled_verts_gt'], out['hand_verts_gt'], rot_matrix = pcd_rotation(out['obj_sampled_verts_gt'],  out['hand_verts_gt'])


        if self.is_param:
            out['hand_fpose_gt'] = torch.Tensor(sample['hand_pose45_gt'])  # (48)
            out['trans_rhand'] = torch.Tensor(sample['trans_rhand'])   # (3)

            out['global_orient'] = out['hand_fpose_gt'][:3]
            out['global_orient_rhand_rotmat'] = batch_rodrigues(out['hand_fpose_gt'][:3].view(1, -1))  # (1, 3, 3)

            out['hand_pose_gt'] = out['hand_fpose_gt'][3:]
            out['pose_rhand_rotmat'] = batch_rodrigues(out['hand_fpose_gt'][3:].view(-1, 3))  # (15,3,3)
            if self.is_rot:
                out = self.aug_param_rot(rot_matrix, out)

        if self.bps_dir:
            out['obj_bps'] =self.bps.encode(out['obj_sampled_verts_gt'], feature_type='dists')['dists'].squeeze()

        if self.is_aug:
            out['obj_sampled_verts_gt'], out['hand_verts_gt'] = self.aug_trans(out['obj_sampled_verts_gt'],
                                                                               out['hand_verts_gt'])
            if self.is_param:
                out['trans_rhand']+=self.random_trans#(3)

        if 'contacts_gen' in sample.keys():
            out['gen_contact'] = torch.Tensor(sample['contacts_gen'])

        if 'obj_sampled_contact_rec' in sample.keys():
            out['obj_sampled_contact_rec'] = torch.Tensor(sample['obj_sampled_contact_rec'])
            out['obj_sampled_contact_rec2'] = torch.Tensor(sample['obj_sampled_contact_rec2'])

        # out['hand_feats_gt'] = torch.Tensor(sample['hand_feats_gt'])
        out['verts_rhand'] = out['hand_verts_gt']

        # --------- For Param2Mesh ----------
        # out['hand_fpose_gt'] = torch.Tensor(sample['hand_pose45_gt']) #(48)
        # out['trans_rhand'] = torch.Tensor(sample['trans_rhand'])+self.random_trans #(3)
        #
        # out['global_orient'] = out['hand_fpose_gt'][:3]
        # out['global_orient_rhand_rotmat'] = batch_rodrigues(out['hand_fpose_gt'][:3].view(1, -1))  # (1, 3, 3)
        #
        # out['hand_pose_gt'] = out['hand_fpose_gt'][3:]
        # out['pose_rhand_rotmat'] = batch_rodrigues(out['hand_fpose_gt'][3:].view(-1,3))  # (15,3,3)


        # if self.train:
        #     out['obj_sampled_verts_aug'] += torch.randn(out['obj_sampled_verts_aug'].shape) * self.aug_vert_jitter

        return out

    def aug_trans(self, obj_xyz, hand_xyz=None):
        N = obj_xyz.size(0)
        random_trans = 0.02* (torch.rand((1, 3))*2-1)  # range [-2cm, 2cm]
        self.random_trans=torch.squeeze(random_trans)

        random_trans_obj = random_trans.repeat(N, 1)
        if hand_xyz is not None:
            random_trans_hand = random_trans.repeat(778, 1)
            return obj_xyz + random_trans_obj, hand_xyz + random_trans_hand
        return obj_xyz + random_trans

    def aug_param_rot(self,rot_matrix, out):
        transl = out['trans_rhand'].unsqueeze(0)
        trans_root = self.root+transl
        center2root = trans_root - 0.
        center2root_rot = torch.mm(center2root, rot_matrix)

        new_transl = center2root_rot+self.root2joint9
        new_transl -= self.joint9

        glob_rotmat = aa2rotmat_new(out['global_orient'].unsqueeze(0))
        new_glob_rotmat = torch.mm(rot_matrix.T, glob_rotmat.squeeze())
        new_glob_orient = rotmat2aa_new(new_glob_rotmat)

        out['trans_rhand'] = new_transl.squeeze()
        out['global_orient'] = new_glob_orient.squeeze()
        out['global_orient_rhand_rotmat']=new_glob_rotmat.view(1,3,3)
        return out

    @staticmethod
    def get_tform(hand_verts, obj_verts, random_rot=True, norm_trans=False):
        """
        Find a 4x4 rigid transform to normalize the pointcloud. We choose the object center of mass to be the origin,
        the hand center of mass to be along the +X direction, and the rotation around this axis to be random.
        :param hand_verts: (batch, 778, 3)
        :param obj_verts: (batch, 2048, 3)
        :return: tform: (batch, 4, 4)
        """
        with torch.no_grad():
            obj_centroid = torch.mean(obj_verts, dim=1)  # (batch, 3)
            hand_centroid = torch.mean(hand_verts, dim=1)

            x_vec = F.normalize(hand_centroid - obj_centroid, dim=1)  # From object to hand
            if random_rot:
                rand_vec = transforms.random_rotations(hand_verts.shape[0], device=hand_verts.device)   # Generate random rot matrix
                y_vec = F.normalize(torch.cross(x_vec, rand_vec[:, :3, 0]), dim=1)  # Make orthogonal
            else:
                ref_pt = hand_verts[:, 80, :]
                y_vec = F.normalize(torch.cross(x_vec, ref_pt - obj_centroid), dim=1)  # From object to hand ref point

            z_vec = F.normalize(torch.cross(x_vec, y_vec), dim=1)  # Z axis

            tform = ops.eyes(4, hand_verts.shape[0], device=hand_verts.device)
            tform[:, :3, 0] = x_vec
            tform[:, :3, 1] = y_vec
            tform[:, :3, 2] = z_vec
            if norm_trans:
                tform[:, :3, 3] = obj_centroid

            return torch.inverse(tform)


    @staticmethod
    def collate_fn(batch):
        out = dict()
        batch_keys = batch[0].keys()
        skip_keys = ['obj_normals_aug', 'obj_verts_aug','obj_sem_contact_gt']   # These will be manually collated

        # For each not in skip_keys, use default torch collator
        for key in [k for k in batch_keys if k not in skip_keys]:
            out[key] = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])

        verts_gt_all = [sample['obj_verts_gt'] for sample in batch]
        faces_all = [sample['obj_faces'] for sample in batch]

        # verts_aug_all = [sample['obj_verts_aug'] for sample in batch]
        # contact_all = [sample['obj_contact_gt'] for sample in batch]
        # # obj_normals_aug_all = [sample['obj_normals_aug'] for sample in batch]
        # # sem_contact_all = [sample['obj_sem_contact_gt'] for sample in batch]
        #
        #
        # out['obj_contact_gt'] = pytorch3d.structures.utils.list_to_padded(contact_all, pad_value=-1)
        # # out['obj_normals_aug'] = pytorch3d.structures.utils.list_to_padded(obj_normals_aug_all, pad_value=-1)
        # # out['obj_sem_contact_gt'] = pytorch3d.structures.utils.list_to_padded(sem_contact_all, pad_value=-1)
        #
        # out['obj_verts_gt'] = pytorch3d.structures.utils.list_to_padded(verts_gt_all, pad_value=-1)
        # # out['obj_verts_aug'] = pytorch3d.structures.utils.list_to_padded(verts_aug_all, pad_value=-1)
        # out['obj_faces'] = pytorch3d.structures.utils.list_to_padded(faces_all, pad_value=-1)
        out['mesh_gt'] = Meshes(verts=verts_gt_all, faces=faces_all)    # This is slower than the above, but probably fast enough
        # # out['mesh_aug'] = Meshes(verts=verts_aug_all, faces=faces_all)

        return out


class CPDataset(ContactDBDataset):
    def __init__(self, data, bps_dir=None, train=False, is_aug=False, is_param=True, is_rot=False, c_enhance=False):
        super(CPDataset, self).__init__(data, bps_dir, train, is_aug, is_param, is_rot)
        self.c_enhance=c_enhance
    def __getitem__(self, idx):
        data = self.dataset[idx]
        sample = {}
        if 'ho_gt' in data.keys():
            sample['obj_verts_ori'] = torch.Tensor(data['ho_gt'].obj_verts)
            sample['obj_faces_ori'] = torch.Tensor(data['ho_gt'].obj_faces)
            sample['obj_sampled_verts_gt'] = torch.Tensor(data['ho_gt'].obj_verts)[data['obj_sampled_idx'], :]

        else:
            # sample['gen_contact'] = torch.Tensor(data['gen_contact'])
            sample['obj_verts_ori'] = data['obj_verts_gt']
            sample['obj_faces_ori'] = data['obj_faces']
            sample['obj_sampled_verts_gt'] = data['obj_sampled_verts_gt']

        if 'diff_contacts' in data.keys():
            sample['gen_contact'] = torch.Tensor(data['diff_contacts']) #(T, 2048, 1)
        else:
            sample['gen_contact'] = torch.Tensor(data['gen_contact'])

        if self.c_enhance:
            sample['gen_contact'] = util.contact_enhance(sample['gen_contact'])
            a=1

        return sample




def pcd_rotation(obj_xyz, rh_xyz=None):
    rot_matrix = p3dt.random_rotation() #(3,3)
    rot_obj_xyz = torch.mm(obj_xyz,rot_matrix)

    if rh_xyz is not None:
        rot_rh_xyz = torch.mm(rh_xyz, rot_matrix)
    return rot_obj_xyz, rot_rh_xyz, rot_matrix

def cp_dataset_sample_idx_split(dataloader):

    contact_dct = {}
    idx_dct = {}


    for idx, sample in enumerate(tqdm(dataloader)):
        if sample['object_name'][0] not in contact_dct.keys():
            contact_dct[sample['object_name'][0]]=[sample['obj_sampled_contact_gt'].squeeze(-1)]
            idx_dct[sample['object_name'][0]] = [idx]
        else:
            contact_dct[sample['object_name'][0]].append(sample['obj_sampled_contact_gt'].squeeze(-1))
            idx_dct[sample['object_name'][0]].append(idx)

    save_sample_idx = np.random.rand(dataloader.__len__())
    contact_all = [{k:soft_to_binary(torch.cat(v,dim=0))} for k,v in contact_dct.items()]
    for s_i in contact_all:
        obj_name_i, contacts_i = s_i.popitem()
        obj_i_idx = idx_dct[obj_name_i]
        labels_, labels_count = dbscan_cluster(contacts_i, min_s=1)

        labels_copy = np.zeros_like(labels_)
        assert labels_.shape[0]==len(obj_i_idx)
        for i in range(len(labels_count)):
            for k, v in labels_count[i].items():
                if v>=15:
                    labels_copy[labels_==int(k)]=0
                else:
                    labels_copy[labels_ == int(k)] = 1

        save_sample_idx[obj_i_idx]=labels_copy


    primary_idxs = np.where(save_sample_idx==0)[0]
    secondary_idxs = np.where(save_sample_idx==1)[0]
    a=1
    np.save('contactpose_fixed_all_primary_idxs.npy',primary_idxs)
    np.save('contactpose_fixed_all_secondary_idxs.npy',secondary_idxs)



def cp_dataset_obj_split():
    primary_idx = np.load('/home/dataset/haoming/ContactPose/pkl/all/contactpose_fixed_all_primary_idxs.npy')
    new_sample_idx = []
    train_data_list = []
    test_data_list = []

    data_dir ='/home/dataset/haoming/ContactPose/pkl/all/contactpose_fixed_all_replace.pkl'
    data_np_list = pickle.load(open(data_dir, 'rb'))

    for idx, sample_i in enumerate(tqdm(data_np_list)):
        obj_name = sample_i['ho_gt'].obj_name
        if obj_name in TRAIN_OBJ_NAMES:
            train_data_list.append(sample_i)
            if idx in primary_idx:
                new_sample_idx.append(0)
            else:
                new_sample_idx.append(1)

        else:
            test_data_list.append(sample_i)

    new_sample_idx = np.array(new_sample_idx).astype(primary_idx.dtype)
    new_primary_idx = np.where(new_sample_idx==0)[0]
    new_second_idx = np.where(new_sample_idx==1)[0]


    output_path = '/home/dataset/haoming/ContactPose/pkl/split'

    np.save(os.path.join(output_path, 'contactpose_fixed_train_primary_idxs.npy'),new_primary_idx)
    np.save(os.path.join(output_path, 'contactpose_fixed_train_secondary_idxs.npy'),new_second_idx)

    pickle.dump(train_data_list, open(os.path.join(output_path, 'contactpose_fixed_train_replace.pkl'), 'wb'))
    pickle.dump(test_data_list, open(os.path.join(output_path, 'contactpose_fixed_test_replace.pkl'), 'wb'))


def obj_verts_bps_save(obj_names, output_path):
    data_dir ='/home/dataset/haoming/ContactPose/pkl/all/contactpose_fixed_all_replace.pkl'
    data_np_list = pickle.load(open(data_dir, 'rb'))

    bps = bps_torch(custom_basis=torch.from_numpy(np.load('../configs/bps.npz')['basis']).to(torch.float32))

    obj_info_dict = {}
    for idx, sample_i in enumerate(tqdm(data_np_list)):
        obj_name = sample_i['ho_gt'].obj_name
        if obj_name in obj_names:
            obj_bps = bps.encode(torch.tensor(sample_i['ho_gt'].obj_verts), feature_type='dists')['dists'].squeeze().cpu()
            obj_info_dict[obj_name + '_bps'] =obj_bps.numpy()

            obj_sample_idx = sample_i['obj_sampled_idx']
            obj_sampled_verts_gt = sample_i['ho_gt'].obj_verts[obj_sample_idx, :]

            obj_info_dict[obj_name + '_verts'] =obj_sampled_verts_gt
            obj_names.remove(obj_name)

    pickle.dump(obj_info_dict, open(os.path.join(output_path, 'obj_info_dict.pkl'), 'wb'))



if __name__ == '__main__':
    from contactopt.manopth.manopth.manolayer import ManoLayer
    from open3d import visualization as o3dv
    from open3d import geometry as o3dg
    from open3d import io
    from contact2mesh.utils.util import makepath

    from torch.utils.data import DataLoader
    from visual import hand_mesh_instance, obj_mesh_instance2
    rh_m = ManoLayer(mano_root='/remote-home/lihaoming/haoming/Contact2Mesh/contactopt/manopth/mano/models', use_pca=False,
                          ncomps=45, side='right', flat_hand_mean=False)
    hand_face = rh_m.th_faces.int().view(-1, 3).numpy()
    test_dataset = ContactDBDataset('/remote-home/share/datasets/haoming/ContactPose/pkl/split/contactpose_fixed_train_replace.pkl', bps_dir=None, train=False)
    # test_loader = DataLoader(test_dataset, batch_size=4, drop_last=True,collate_fn=ContactDBDataset.collate_fn)


    # cp_data = pickle.load(open(os.path.join('/home/dataset/haoming/ContactPose/pkl', 'obj_info_dict.pkl'),'rb'))
    # sp_data = pickle.load(open(os.path.join('/home/dataset/haoming/ShapeNet_v2_processed', 'shapenetv2_sample_verts_mscale.pkl'),'rb'))
    i=0
    for data in iter(test_dataset):
        obj_name = data['object_name']
        save_path = makepath('./gt_save/{}'.format(obj_name),isfile=False)
        if obj_name=='mouse':
            sample_verts = data['obj_sampled_verts_gt'].numpy()
            obj_faces = data['obj_faces'].numpy()
            obj_verts = data['obj_verts_gt'].numpy()
            obj_contact = data['obj_contact_gt'].squeeze().numpy()

            hand_verts = data['hand_verts_gt'].numpy()
            hand_mesh = hand_mesh_instance(hand_verts, hand_face)
            obj_mesh = obj_mesh_instance2(obj_verts,obj_faces,obj_contact,colors='jet')

            io.write_triangle_mesh('{}/hand_{}.obj'.format(save_path, i), hand_mesh)
            io.write_triangle_mesh('{}/{}_{}_contact.obj'.format(save_path, data['object_name'], i), obj_mesh)
            a=1
            print('finish_{}'.format(i))
            i+=1


        # obj_face = data['obj_faces'].numpy()
        # obj_verts = data['obj_verts_gt'].numpy()
        # break


    # sample_pcd = pcd_instance(sample_verts)
    # sample_pcd.estimate_normals(search_param=o3dg.KDTreeSearchParamKNN(knn=10))
    # o3dv.draw_geometries([sample_pcd], point_show_normal=True)

    a=1


    # print('Epoch dataload time: ', time.time() - start_time)
