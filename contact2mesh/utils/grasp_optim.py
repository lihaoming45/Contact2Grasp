# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import chamfer_distance as chd

from contact2mesh.utils.util import makepath, to_tensor, to_cpu, remove_noise
from contact2mesh.utils.mano_util import pose_decompose
from contact2mesh.utils.mano_util import forward_mano2
from contactopt.manopth.manopth.manolayer import ManoLayer
from bps_torch.bps import bps_torch
from data.obman_data.data_pkl import get_NN, get_pseudo_cmap
from contact2mesh.xinzhuo.refine_utils import Contact_loss, get_interior, FTCloss, Physcial_loss, Attractive_loss

from pytorch3d.structures import Meshes

from contact2mesh.utils.MANO_indices import bigfinger_vertices as bf_v
from contact2mesh.utils.MANO_indices import indexfinger_vertices as if_v
from contact2mesh.utils.MANO_indices import fourthfinger_vertices as ff_v
from contact2mesh.utils.MANO_indices import smallfinger_vertices as sf_v
from contact2mesh.utils.MANO_indices import middlefinger_vertices as mf_v

from contact2mesh.utils.MANO_indices import mano_fingers_idxs
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.colors import name_to_rgb
from contact2mesh.utils.vis_utils import sp_animation
from contact2mesh.utils.MANO_indices import prior_idx

import trimesh
from visual import hand_mesh_instance, mesh_instance
from open3d import visualization as o3dv
import open3d as o3d
from open3d import geometry as o3dg
from open3d import utility as o3du

class HandOptim(nn.Module):

    def __init__(self,
                 cfg,
                 verbose=False,
                 verhtml=False,
                 mode='pgt'
                 ):
        super(HandOptim, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg
        self.mode = mode

        # flat_hand_mean = False if 'ContactPose' in cfg.desc else True
        self.rh_m = ManoLayer(mano_root='/remote-home/lihaoming/haoming/GrabNet/contactopt/manopth/mano/models', use_pca=False,
                              ncomps=45, side='right', flat_hand_mean=cfg.flat_hand_mean).to(self.device)

        self.flat_hand_pose = torch.zeros(45).to(self.device) if cfg.flat_hand_mean else -self.rh_m.th_hands_mean

        self.config_optimizers()

        rh_f = self.rh_m.th_faces.int().view(1, -1, 3)
        self.rh_f_np = rh_f.cpu().view(-1, 3).numpy()
        self.rh_f = rh_f.repeat(self.bs, 1, 1).to(self.device).to(torch.long)

        # self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        # self.rhand_idx = to_tensor(np.load(self.cfg.losses.rh2smplx_idx), dtype=torch.long)
        self.finger_ids = to_tensor(np.concatenate([bf_v, if_v, mf_v, ff_v, sf_v])).long()
        self.finger_dct = {k: torch.tensor(v) for k, v in mano_fingers_idxs.items()}

        self.prior_idx = prior_idx

        palm_ids = torch.tensor(list(range(778)))
        palm_ids = torch.cat([palm_ids, self.finger_ids])
        uniset, count = palm_ids.unique(return_counts=True)
        self.palm_ids = uniset.masked_select(mask=(count == 1))

        self.z_normal = torch.tensor([[0., 0., 1.]]).to(self.device)  # (1,3)

        h, w, sw = pow(3, 0.5) / 2, 0.5, pow(2, 0.5) / 4
        self.dec_znorm = torch.tensor(
            [[-w, 0, h], [w, 0, h], [0, -w, h], [0, w, h], [sw, sw, h], [-sw, -sw, h], [-sw, sw, h],
             [sw, -sw, h]]).to(self.device)  # (8,3)

        self.I = torch.eye(3).to(self.device)
        self.R_init = torch.zeros(3, 3).to(self.device)
        self.pt = torch.zeros(1, 3).to(self.device)
        # self.rhand_tri = to_tensor(np.load(self.cfg.losses.rh_faces).astype(np.int32))
        # self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)
        self.verbose = verbose
        self.verhtml = verhtml

        self.bps_torch = bps_torch()
        # self.ch_dist = chd.ChamferDistance()

        self.grasp_anim = sp_animation()

    def config_optimizers(self):
        bs = 1
        self.bs = bs
        device = self.device
        dtype = self.dtype
        lr = self.cfg.lr  # 5e-3

        self.optim_parts = []
        if self.mode == 'pgt':
            self.opt_params = {
                'global_orient': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=True),
                'hand_pose': torch.randn(bs, 15 * 3, device=device, dtype=dtype, requires_grad=True),
                'transl': torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
            }
            self.opt_key_word = {'global_orient': 0, 'hand_pose': 1, 'transl': 2}

            self.opt = optim.Adam(
                [self.opt_params[k] for k in self.opt_key_word.keys()], lr=lr)
            self.optimizers = [self.opt]

            self.num_iters = [self.cfg.steps1]


        elif self.mode == 'local_pgt':
            self.opt_params = {
                'index': torch.randn(bs, 1 * 9, device=device, dtype=dtype, requires_grad=False),
                'middle': torch.randn(bs, 1 * 9, device=device, dtype=dtype, requires_grad=False),
                'small': torch.randn(bs, 1 * 9, device=device, dtype=dtype, requires_grad=False),
                'fourth': torch.randn(bs, 1 * 9, device=device, dtype=dtype, requires_grad=False),
                'big': torch.randn(bs, 1 * 9, device=device, dtype=dtype, requires_grad=False),
                'global_orient': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=False),
                'transl': torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=False),
            }
            self.opt_key_word = {'index': 0, 'middle': 1, 'small': 2, 'fourth': 3, 'big': 4, 'global_orient': 5,
                                 'transl': 6}
            # lr = self.cfg.get('mano_opt_lr', 5e-3)
            self.opt_s1 = optim.Adam(
                [self.opt_params[k] for k in ['global_orient', 'transl', 'index', 'middle', 'small', 'fourth', 'big']],
                lr=lr)
            # self.opt_s2 = optim.Adam([self.opt_params[k] for k in ['index', 'middle', 'small', 'fourth', 'big']], lr=lr)

            self.optimizers = [self.opt_s1]
            self.num_iters = [self.cfg.steps1]  # 201
            self.w_steps = [self.cfg.s1w]  # 201

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')

    def init_optim(self, set_grad=False, optim_part=None):
        if optim_part is None:
            for k in list(self.opt_key_word.keys()):
                self.opt_params[k].requires_grad = set_grad

    def init_params(self, start_params):
        """
        :param start_params:  containing: hand_pose, transl, global_orient
        :return:
        """
        self.fit_s1 = False
        self.fit_s2 = False

        if self.mode == 'local_pgt':
            start_params = pose_decompose(start_params)

        self.rh_params = start_params.copy()

        # self.start_params = start_params
        start_params_aa = start_params

        for k in self.opt_params.keys():
            self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k], self.bs, dim=0)

    def get_mano_verts(self, output):
        with torch.no_grad():
            verts_rhand, _ = forward_mano2(self.rh_m, output['hand_fpose'], None, output['transl'])

            v = verts_rhand.reshape(-1, 778, 3)
            verts_fingers = v[:, self.finger_ids]

        return v, verts_fingers

    def calc_loss(self, batch, sw):
        B = batch['obj_sampled_verts_gt'].size(0)

        # opt_params = {k: aa2rotmat_new(v) for k, v in self.opt_params.items() if k != 'transl'}
        opt_params = self.opt_params.copy()
        # opt_params['transl'] = self.opt_params['transl']

        if 'big' in opt_params.keys():
            opt_params['hand_pose'] = torch.cat([opt_params[k] for k in list(self.opt_key_word.keys())[:5]], dim=1)
            a = 1

        opt_params['hand_fpose'] = torch.cat([opt_params['global_orient'], opt_params['hand_pose']], dim=1)
        verts_opt, _ = forward_mano2(self.rh_m, opt_params['hand_fpose'], None, opt_params['transl'])

        # cmap consistency loss
        # verts_opt_prior = verts_opt[:,self.prior_idx,:]
        nn_dist, nn_idx = get_NN(batch['obj_sampled_verts_gt'], verts_opt)
        obj_contact_pred = remove_noise(get_pseudo_cmap(nn_dist).view(B, -1), 0.05)
        if 'contact_gen' in batch.keys():
            loss_consistency = sw['consist'] * torch.nn.functional.mse_loss(obj_contact_pred, batch['contact_gen'],
                                                                            reduction='none').sum() / B

        # inter-penetration loss
        mesh = Meshes(verts=verts_opt.cuda(), faces=self.rh_f.cuda())
        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
        interior = get_interior(hand_normal, verts_opt, batch['obj_sampled_verts_gt'], nn_idx).type(torch.bool)
        penetr_dist = sw['pentr'] * nn_dist[interior].sum() / B

        losses = {
            # "consistency": loss_consistency,
            # "hand_centric": loss_contact,
            "penetration": penetr_dist,
            # 'penet': 1  *torch.pow(rh2obj_penet[is_penet], 2).mean()
        }

        # hand-centric loss
        if 'hc' in self.cfg.folder:
            loss_contact = Contact_loss(batch['obj_sampled_verts_gt'], verts_opt, cmap=nn_dist < 0.03 ** 2) * sw['hc']
            losses['hand_centric'] = loss_contact
        if 'ftc' in self.cfg.folder:
            loss_ftc = FTCloss(batch['obj_sampled_verts_gt'], verts_opt) * sw['ftc']
            losses['tip_close'] = loss_ftc

        if 'fc' in self.cfg.folder:
            loss_fc = Physcial_loss(batch['obj_sampled_verts_gt'], batch['obj_sampled_normals_gt'], verts_opt,
                                    self.z_normal, self.dec_znorm, self.I, self.R_init, self.pt) * 8
            losses['force'] = loss_fc


        if 'att' in  self.cfg.folder:
            locc_att = Attractive_loss(verts_opt, hand_normal, batch['contact_gen'], batch['obj_sampled_verts_gt'])
            losses['attractive'] = locc_att

        if self.mode == 'pgt':
            rhand_loss = {k: self.LossL2(self.rh_params[k].detach().reshape(-1), self.opt_params[k].reshape(-1))
                          for k in ['global_orient', 'hand_pose']}
        elif self.mode == 'gt':
            rhand_loss = {k: self.LossL2(self.rh_params[k].detach().reshape(-1), self.opt_params[k].reshape(-1))
                          for k in ['global_orient']}



        elif self.mode == 'local_pgt':
            rhand_loss = {k: self.LossL2(self.rh_params[k].detach().reshape(-1), self.opt_params[k].reshape(-1))
                          for k in self.optim_parts if k not in ['transl']}

        rhand_loss['transl'] = self.LossL1(self.opt_params['transl'], self.rh_params['transl'].detach()) * 2.5

        losses.update(rhand_loss)

        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts_opt, opt_params

    def get_peneteration(self, source_mesh, target_mesh):

        source_verts = source_mesh.verts_packed()
        source_normals = source_mesh.verts_normals_packed()

        target_verts = target_mesh.verts_packed()
        target_normals = target_mesh.verts_normals_packed()

        src2trgt, trgt2src, src2trgt_idx, trgt2src_idx = chd.ChamferDistance(source_verts.reshape(1, -1, 3).to(self.device),
                                                                      target_verts.reshape(1, -1, 3).to(self.device))

        source2target_correspond = target_verts[src2trgt_idx.data.view(-1).long()]

        distance_vector = source_verts - source2target_correspond

        in_out = torch.bmm(source_normals.view(-1, 1, 3), distance_vector.view(-1, 3, 1)).view(-1).sign()

        src2trgt_signed = src2trgt * in_out

        return src2trgt_signed


    def fitting(self, batch, net_output, idx, thres_vol=5.0):
        self.thres_vol = thres_vol
        self.w_steps = [self.cfg.s1w.copy()]

        gnet_verts, _ = self.get_mano_verts(batch, net_output)

        if self.mode == 'local_pgt':
            self.optimizers = [self.opt_s1]
            self.num_iters = [self.cfg.steps1]  # 201

            self.init_optim(False)
            self.local_detect(gnet_verts, batch)
            skip = True if (1 - bool(self.optim_parts)) else False

        else:
            inter_vol, _ = self.penetration_detect(batch, gnet_verts)
            skip = True if inter_vol < self.thres_vol and inter_vol>=0.5  else False

        opt_results = {}
        if skip:
            opt_results['gnet_verts'] = gnet_verts
            opt_results['opt_verts'] = gnet_verts
            return opt_results

        if self.verhtml:
            makepath('./{}/'.format(self.cfg.folder), isfile=False)
            f = open(r'./{}/{}_losses_print.txt'.format(self.cfg.folder, idx), 'w')
        for stg, optimizer in enumerate(self.optimizers):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses, opt_verts,_ = self.calc_loss(batch, net_output, self.w_steps[stg])
                losses['loss_total'].backward(retain_graph=True)
                optimizer.step()
                if self.verbose and itr % 50 == 0:
                    print(self.create_loss_message(losses, idx, itr))

                if self.verhtml and itr % 100 == 0:
                    self.add_grasp_frame(opt_verts, batch, itr)
                    print(self.create_loss_message(losses, idx, itr), file=f)

        if self.verhtml:

            self.grasp_anim.save_animation('./{}/optim_process_{}.html'.format(self.cfg.folder, idx))
            print('save in :./{}/optim_process_{}.html'.format(self.cfg.folder, idx))
            f.close()



        opt_results['gnet_verts'] = gnet_verts
        opt_results['opt_verts'] = opt_verts

        return opt_results

    def local_detect(self, hand_verts, batch):
        self.ftc,self.att=True,True

        # N = batch['obj_verts_ori'].shape[0]
        # vol, insert_idx = self.penetration_detect(batch, hand_verts, return_vol=True)
        ptr_start_time=time.time()
        insert_idx = self.penetration_detect_quick(batch, hand_verts)
        ptr_end_time=time.time()
        print("penetration_detect_quick time ={}.s".format(ptr_end_time-ptr_start_time))

        # if vol < self.thres_vol:
        if len(insert_idx) < 15 and len(insert_idx)>2: #<50
            self.optim_parts = []
        else:

            if len(insert_idx)>50.0:
                self.w_steps[0]['pentr']=self.cfg.s1w['pentr']*2.0
            elif len(insert_idx) <=50 and  len(insert_idx) >30:
                self.w_steps[0]['pentr']=self.cfg.s1w['pentr']*2.0
            else:
                self.w_steps[0]['pentr']=self.cfg.s1w['pentr']
                a=1


            optim_parts = []
            superset = torch.cat([insert_idx, self.palm_ids])
            uniset, count = superset.unique(return_counts=True)

            in_count = (count > 1).sum()
            if in_count >= 70 or in_count==0.: #>=15
                self.init_optim(True)
                # optim_parts = ['transl', 'global_orient']
                optim_parts = [key for key in self.opt_params.keys()]
                self.fit_s1 = True
                if in_count==0:
                    self.w_steps[0]['ftc'] = self.cfg.s1w['ftc']*2.0
                    self.w_steps[0]['hc'] = self.cfg.s1w['hc']*2.0
                else:
                    self.w_steps[0]['ftc'] = self.cfg.s1w['ftc']
                    self.w_steps[0]['hc'] = self.cfg.s1w['hc']
                    self.ftc = False
                    self.att=False




            else:

                for k, v in self.finger_dct.items():
                    superset = torch.cat([insert_idx, v])
                    uniset, count = superset.unique(return_counts=True)
                    mask = (count > 1)
                    if mask.sum() >= 10.: #10.
                        optim_parts.append(k)
                        self.fit_s2 = True

                for part in optim_parts:
                    self.opt_params[part].requires_grad = True
                    a = 1

            self.optim_parts = optim_parts


    def penetration_detect(self, data, hand_verts, return_vol=True):
        self.optim_parts = []
        if torch.is_tensor(hand_verts):
            hand_verts = hand_verts.detach().squeeze().cpu().numpy()
        obj_mesh = trimesh.Trimesh(vertices=data["obj_verts_ori"],
                                   faces=data["obj_faces_ori"])


        inside_tri = obj_mesh.contains(hand_verts)
        insert_idx = torch.tensor(inside_tri).nonzero().view(-1)

        if return_vol:
            hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=self.rh_f_np)
            volume = self.intersect_vox(obj_mesh, hand_mesh, pitch=0.005)

            return volume * 1e6, insert_idx
        else:
            return insert_idx

    def penetration_detect_quick(self, data, hand_verts):
        if torch.is_tensor(hand_verts):
            hand_verts = hand_verts.detach().squeeze().cpu().numpy()

        mesh = o3dg.TriangleMesh()
        mesh.vertices = o3du.Vector3dVector(data["obj_verts_ori"])

        obj_faces = data["obj_faces_ori"].astype(np.int)
        mesh.triangles = o3du.Vector3iVector(obj_faces)

        t_mesh =o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(t_mesh)  # we do not need the geometry ID for mesh


        signed_distance = scene.compute_signed_distance(hand_verts)
        signed_distance=signed_distance.numpy()
        insert_mask = (signed_distance<0)*1
        # insert_idx = np.where(insert_mask==1)
        insert_idx = torch.tensor(insert_mask).nonzero().view(-1)

        return insert_idx

    @staticmethod
    def create_loss_message(loss_dict, idx=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Idx:{idx:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'

    @staticmethod
    def intersect_vox(obj_mesh, hand_mesh, pitch=0.005):
        obj_vox = obj_mesh.voxelized(pitch=pitch)
        obj_points = obj_vox.points
        inside_idx = hand_mesh.contains(obj_points)
        volume = inside_idx.sum() * np.power(pitch, 3)

        return volume

    def add_grasp_frame(self, hand_verts, batch, idx):
        hand_mesh = Mesh(v=hand_verts.detach().cpu().squeeze().numpy(), f=self.rh_f_np, vc=name_to_rgb['pink'])
        obj_mesh = Mesh(v=batch['obj_verts_ori'], f=batch['obj_faces_ori'], vc=name_to_rgb['yellow'])
        self.grasp_anim.add_frame([hand_mesh, obj_mesh], ['opt_grasp_{}'.format(idx), 'obj'])
        a = 1


