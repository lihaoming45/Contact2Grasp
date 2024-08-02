import torch
import numpy as np
from pytorch3d.structures import Meshes
from contact2mesh.utils.util import class_to_onehot
from contact2mesh.utils.mano_util import forward_mano2
from contactopt.manopth.manopth.manolayer import ManoLayer
from contact2mesh.utils.train_util import Trainer_base, model_select,point2point_signed, geodesic_loss
# from contact2mesh.utils.grasp_metric import cord_mse
import contact2mesh.utils.util as util
# from pytorch3d.loss import chamfer_distance
from contact2mesh.utils.contact_metrics import distDICE
import mano

class Train_Param2Mesh(Trainer_base):
    def  __init__(self,args, train_loader, test_loader=None):
        model = model_select(args.use_model, args)
        super(Train_Param2Mesh, self).__init__(args, model, train_loader, test_loader)
        self.cfg = args
        self.eval_gap = 5
        self.save_step = 10
        self.save_epo = 50

        flat_hand_mean=False if 'ContactPose' in args.desc else True
        self.rhm_model = ManoLayer(mano_root='/remote-home/lihaoming/haoming/Contact2Mesh/contactopt/manopth/mano/models', use_pca=False,
                                   ncomps=45, side='right', flat_hand_mean=flat_hand_mean).to(self.device)

        rh_f = self.rhm_model.th_faces.int().view(1, -1, 3)
        self.rh_f = rh_f.repeat(args.batch_size, 1, 1).to(self.device).to(torch.long)

        if isinstance(args.c_weights_path, str):
            v_weights = torch.from_numpy(np.load(args.c_weights_path)).to(torch.float32).to(self.device)
            v_weights2 = torch.pow(v_weights, 1.0 / 2.5)
        else:
            v_weights, v_weights2 = None, None

        self.vpe = None
        if isinstance(args.vpe_path, str):
            self.vpe = torch.from_numpy(np.load(args.vpe_path)).to(self.device).to(torch.long)

        self.v_weights = v_weights
        self.v_weights2 = v_weights2
        self.w_dist = torch.ones([args.batch_size, 2048]).to(self.device)

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')

        self.do_eval=True
        self.do_metric=False
        self.save_epo=2

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def model_forward(self, data):
        out = self.model(data['obj_sampled_verts_gt'].transpose(1,2), data['trans_rhand'], data['global_orient_rhand_rotmat'],data['pose_rhand_rotmat'])
        return out

    def cal_metrics(self, out, data, nsample=1):
        if type(self.rhm_model).__name__ == 'ManoLayer':
            verts_rhand, _ = forward_mano2(self.rhm_model, out['hand_fpose'], None, out['transl'])
        else:
            verts_rhand = self.rhm_model(**out).vertices

        verts_rhand_gt = data['verts_rhand']

        rhand_verts_mse = torch.square(verts_rhand - verts_rhand_gt).sum(dim=2).sqrt().mean()

        eval_dict = {
            'rhand_verts_mse':rhand_verts_mse
        }

        return eval_dict

    def cal_loss(self, out, data, dis_out=None):
        dtype = data['trans_rhand'].dtype

        q_z = torch.distributions.normal.Normal(out['mean'], out['std'])
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(dtype),
            scale=torch.tensor(np.ones([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(dtype))
        kl_loss = self.args.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))


        #
        verts_rhand, _ = forward_mano2(self.rhm_model, out['hand_fpose'], None, out['transl'])
        # verts_rhand = self.rhm_model(**out).vertices

        rh_mesh = Meshes(verts=verts_rhand, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=data['verts_rhand'], faces=self.rh_f).to(self.device).verts_normals_packed().view(-1,778, 3)

        o2h_signed, h2o, _ = point2point_signed(verts_rhand, data['obj_sampled_verts_gt'], rh_mesh)
        o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(data['verts_rhand'], data['obj_sampled_verts_gt'], rh_mesh_gt)

        # addaptive weight for penetration and contact verts
        w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.
        w = self.w_dist.clone()
        w[~w_dist] = .1  # less weight for far away vertices
        w[w_dist_neg] = 1.5  # more weight for penetration

        loss_dist_h = 35 * (1. - self.cfg.kl_coef) * torch.mean(
            torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2))
        loss_dist_o = 30 * (1. - self.cfg.kl_coef) * torch.mean(
            torch.einsum('ij,ij->ij', torch.abs(o2h_signed - o2h_signed_gt), w))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.cfg.kl_coef) * torch.mean(
            torch.einsum('ijk,j->ijk', torch.abs((data['verts_rhand'] - verts_rhand)), self.v_weights))
        ########## edge loss
        loss_edge = 30 * (1. - self.cfg.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe),
                                                                   self.edges_for(data['verts_rhand'], self.vpe))


        ########## pose loss
        pose_rm_gt = torch.cat([data['global_orient_rhand_rotmat'],data['pose_rhand_rotmat']],dim=1) #(B,16,3,3)
        pose_rm_out = out['fullpose']
        loss_pose = geodesic_loss(pose_rm_gt, pose_rm_out)*0.1

        ######### transl loss
        transl_gt = data['trans_rhand'] #(B,3)
        transl_out = out['transl'] #(B,3)
        loss_transl =self.LossL2(transl_out,transl_gt)*0.1

        loss_dict = {'kl': kl_loss,
                     'edge': loss_edge,
                     'm_rec': loss_mesh_rec_w,
                     'dist_h': loss_dist_h,
                     'dist_o': loss_dist_o,
                     'pose':loss_pose,
                     'trans': loss_transl,
                     }

        # loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_total = kl_loss+loss_edge+loss_mesh_rec_w + loss_dist_h+loss_dist_o+loss_pose+loss_transl
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

if __name__ == '__main__':
    a = 1
