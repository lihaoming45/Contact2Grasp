import sys
import os

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d import ops, transforms
from torch.nn import Linear as Lin, BatchNorm1d as Bn
from contact2mesh.models.pointnet.pointnet_utils import PointNetEncoder
from reconstruction.model_rec import feature_upsample, AePcd
from contact2mesh.utils.mano_util import CRot2rotmat, rotmat2aa


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256, norms=None):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        if norms == 'ln':
            self.n1 = nn.LayerNorm(n_neurons)
            self.n2 = nn.LayerNorm(Fout)

        elif norms == 'bn':
            self.n1 = nn.BatchNorm1d(n_neurons)
            self.n2 = nn.BatchNorm1d(Fout)

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.n1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.n2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class CVAE_Param2Mesh(nn.Module):
    def __init__(self, globalD=1024,
                 latentD=32,
                 in_chan=3 + 9 + 15 * 3 * 3,
                 in_pcd=1024,
                 n_neurons=512,
                 args=None,
                 is_train=True,
                 encoder='pnet'
                 ):
        super(CVAE_Param2Mesh, self).__init__()
        self.args = args
        self.bs = args.batch_size
        self.latentD = latentD

        if encoder=='pnet':
            self.obj_pcd_encoder = PointNetEncoder(global_feat=args.glob_feat, feature_transform=True, inchan=3, outchan=args.globalD)
        # # if 'best_model.pth' == os.path.basename(args.obj_enc_pth):
        #     self.obj_pcd_encoder = pnet2_cls(args.obj_enc_pth)
        else:
            # self.obj_pcd_encoder = AeConv1d(in_chan=3,n_neuron=256, is_dec=False)
            self.obj_pcd_encoder = AePcd(args, globalD=args.pcd_globalD, global_feat=args.glob_feat, is_dec=False)
            # if args.obj_enc_pth:
            #     self.obj_pcd_encoder.load_state_dict(torch.load(args.obj_enc_pth))
        #
        # if args.obj_enc_fixed and is_train:


        self.fc_bn1 = nn.BatchNorm1d(in_chan + in_pcd)

        self.fc_rb1 = ResBlock(in_chan + in_pcd, n_neurons, norms='bn')
        self.fc_rb2 = ResBlock(n_neurons + in_chan + in_pcd, n_neurons, norms='bn')

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_var = nn.Linear(n_neurons, self.latentD)

        # self.dec_bn1 = nn.BatchNorm1d(in_pcd)
        self.dec_rb1 = ResBlock(self.latentD + in_pcd, n_neurons, norms='bn')
        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_pcd, n_neurons, norms='bn')

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(self, xyz, trans_rhand, global_orient_rhand_rotmat, pose_rhand_rotmat):
        bs = xyz.size(0)

        obj_f,_ = self.obj_pcd_encoder(xyz)
        X = torch.cat([obj_f, global_orient_rhand_rotmat.view(bs, -1), pose_rhand_rotmat.view(bs, -1), trans_rhand],
                      dim=1)  # (B,1024+9+3)

        X0 = self.fc_bn1(X)
        X = self.fc_rb1(X0, True)
        X = self.fc_rb2(torch.cat([X0, X], dim=1), True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X))), obj_f

    def decode(self, Zin, obj_f):
        # o_f = self.dec_bn1(obj_f)

        X0 = torch.cat([Zin, obj_f], dim=1)  # (B, latenD+512)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        results = parms_decode(pose, trans)
        results['z'] = Zin

        return results

    def forward(self, obj_sampled_verts, trans_rhand, global_orient_rhand_rotmat, pose_rhand_rotmat):
        z, obj_f = self.encode(obj_sampled_verts, trans_rhand, global_orient_rhand_rotmat, pose_rhand_rotmat)
        z_s = z.rsample()

        hand_parms = self.decode(z_s, obj_f)
        results = {'mean': z.mean, 'std': z.scale,'latent':z_s,'obj_glob_feat':obj_f}
        results.update(hand_parms)

        return results

    def sample_poses(self, xyz, seed=None):
        self.eval()
        obj_f,_ = self.obj_pcd_encoder(xyz)
        bs = obj_f.shape[0]
        np.random.seed(seed)
        dtype = obj_f.dtype
        device = obj_f.device
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen, dtype=dtype).to(device)

        return self.decode(Zgen, obj_f)


def parms_decode(pose, trans):
    bs = trans.shape[0]

    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 9])
    fpose = rotmat2aa(pose).view(bs, -1)

    # hand_tforms = CRot2rotmat(tforms)  # (B,6)->(B,3,3)
    # hand_tforms = torch.cat([hand_tforms, trans.unsqueeze(-1)], dim=2)  # (B,3,4)
    # tensor = torch.tensor([0., 0., 0., 1.]).view(1, 1, 4).repeat(bs, 1, 1).to(0)
    # hand_tforms = torch.cat([hand_tforms, t_zeros], dim=1)  # (B,4,4)

    global_orient = fpose[:, :3]
    hand_pose = fpose[:, 3:]  # (B, 48)
    # hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])  # (B,16,3,3)

    hand_parms = {'global_orient': global_orient, 'hand_pose': hand_pose, 'hand_fpose': fpose,
                  'transl': trans, 'fullpose': pose_full,
                  # 'hand_tforms': hand_tforms
                  }

    return hand_parms

