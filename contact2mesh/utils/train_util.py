import torch
import pickle
from torch import optim
from tqdm import tqdm
import numpy as np
import os.path as osp
import contact2mesh.utils.util as util
from contact2mesh.utils.contact_metrics import distDICE
from torch.utils.tensorboard import SummaryWriter
from contact2mesh.models.transformer.pct_model import PCTCVAE
import chamfer_distance as chd
from contact2mesh.models.cvae.cvae import CVAE_Resmlp,CVAE_PNet2_rec,CVAE_PNet2_seg, \
    CVAE_Param2Mesh,CVAE_PNet2_mscale,CVAE_PNet_mscale,CVAE_trans
from contact2mesh.xinzhuo.pointnet2_seg_pre import GraspNet
from contact2mesh.models.KPConv.KPFCNN import KPFCNN

from contact2mesh.models.diffusion.combined_model_c2m import CombinedModel


def model_select(model_name, args):
    if model_name=='Trans':
        model = CVAE_trans(args=args, latentD=args.latentD)

    elif model_name=='PCT':
        model = PCTCVAE(args=args, latentD=args.latentD)

    elif model_name=='mlps':
        model = CVAE_Resmlp(conditionD=args.conditionD)

    elif model_name=='Param2Mesh':
        model = CVAE_Parm2Mesh(latentD=args.latentD, args=args)

    elif model_name == 'PNet2_mscale':
        model = CVAE_PNet2_mscale(latentD=args.latentD, args=args)

    elif model_name == 'PNet_mscale':
        model = CVAE_PNet_mscale(latentD=args.latentD, args=args)

    elif model_name == 'PNet2_seg':
        model =CVAE_PNet2_seg(latentD=args.latentD, args=args)

    elif model_name == 'PNet2_rec':
        model =CVAE_PNet2_rec(latentD=args.latentD, args=args)
    elif model_name=='GraspNet':
        model =GraspNet(feat_trans=True, args=args)

    elif model_name=='KPC':
        model = KPFCNN(args)

    elif model_name=='CombineDiffusion':
        model = CombinedModel(args)

    return model


class Trainer_base:
    def __init__(self, args, model, train_loader, test_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cur_epo = 0
        self.eval_gap = 5
        self.save_step = 50
        self.save_epo=0
        self.best_eval_loss = np.inf

        self.args = args
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.do_eval=False
        self.do_metric=False

        # self.obj_info =pickle.load(open(args.obj_info,'rb'))
        if test_loader:
            self.test_loader = test_loader
            # self.test_gt_dict = self.gt_collection(util.TESI_OBJ_NAMES) #['mug', 'toothpaste', 'flashlight', 'stapler']
            self.do_eval=True
        # self.init_loss_meter()

        self.critetion = torch.nn.BCELoss()
        # self.dice_criterion = util.BinaryDiceLoss()

        if bool(self.args.checkpoints):
            self.model_load()
            print('load the model path: '+self.args.checkpoints+"\n")

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=0.0005)
        # self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)


        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        if bool(args.scheduler):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)


        log_dir = args.save_path if 'save_path' in args else './contact2mesh'
        self.writer = SummaryWriter(log_dir=log_dir +'/runs/' + args.desc)
        self.snap_path = util.makepath(osp.join(log_dir,'checkpoints/') + self.args.desc, False)

    def gt_collection(self,obj_names, key_word='obj_sampled_contact_gt'):
        gt_dict = {}
        for obj_name in obj_names:
            gt_dict[obj_name] = []
            for data in iter(self.test_loader):
                if data['object_name'][0] == obj_name:
                    gt_dict[obj_name].append(data[key_word])
                    a=1


            gt_dict[obj_name] = torch.cat(gt_dict[obj_name],dim=0).squeeze().to(self.device)

        return gt_dict

    def gen_collection(self,obj_names, nsample):
        gen_dict={}
        for obj_name in obj_names:
            a=1

    def model_load(self):
        self.cur_epo = int(osp.basename(self.args.checkpoints).split('_')[1])
        self.model.load_state_dict(torch.load(self.args.checkpoints, map_location=self.device))

    def cal_metrics(self, out, data, nsample=1):
        pred_contact = util.remove_noise(out['pred_contact'],0.05)
        rec_dice = distDICE(pred_contact, data['obj_sampled_contact_gt'].squeeze(-1),return_error=False).mean()

        eval_dict = {
            'rec_dice': rec_dice
        }

        return eval_dict


    def cal_loss(self, out, data, dis_out=None):
        """
        Caluate the loss for the Framework
        :param contact_obj_gt: ground truth of object contact map (already sampled), (B,N,1)
        :param network_out: including the object contact prediction (B,10, N), and latent code (B,latentD)
        :param sampled_verts_idx: The sampled verts idx for the object contact and verts , (B, N)

        """
        contact_gt = data['obj_sampled_contact_gt'].squeeze(-1)
        # caculate the kl, contact loss
        q_z = torch.distributions.normal.Normal(out['mean'], out['std'])
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(contact_gt.dtype),
            scale=torch.tensor(np.ones([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(contact_gt.dtype))
        kl_loss = self.args.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        # mse_loss = self.criterion(out['pred_contact'], contact_gt)*10
        bce_loss = self.critetion(out['pred_contact'], util.soft_to_binary(contact_gt).float())

        total_loss = kl_loss + bce_loss

        pred_contact = out['pred_contact'].clone().detach()
        rec_dice = distDICE(pred_contact, data['obj_sampled_contact_gt'].squeeze(-1),return_error=False).mean()

        loss_dict = {'loss_total': total_loss,
                     # 'mse_loss': mse_loss.item(),
                     'kl_loss': kl_loss,
                     'bce_loss': bce_loss,
                     'rec_dice': rec_dice
                     }

        return total_loss, loss_dict

    def to_device(self,data):
        return util.dict_to_device(data, self.device)

    def train_epoch(self, epoch):
        # self.meter_reset()
        cur_loss_dict = {}
        train_iterator = tqdm(self.train_loader, total=self.train_loader.__len__(), ncols=180)
        idx = 0
        for data in train_iterator:
            data = self.to_device(data)
            self.optimizer.zero_grad()

            out = self.model_forward(data)

            total_loss, loss_dict = self.cal_loss(out, data)
            # if self.model.obj_pcd_encoder.stn.conv1.weight.grad is not None:
            #     if torch.isnan(self.model.obj_pcd_encoder.stn.conv1.weight.grad.sum()):
            #         a=1
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1, norm_type=2)
            # assert torch.isnan(self.model.mu).sum() == 0, print(self.model.mu)

            # if self.model.obj_pcd_encoder.stn.conv1.weight.grad is not None:
            #     if torch.isnan(self.model.obj_pcd_encoder.stn.conv1.weight.grad.sum()):
            #         a=1
            self.optimizer.step()
            # self.optimizer.zero_grad()


            cur_loss_dict = {k: cur_loss_dict.get(k, 0.0) + v.item() for k, v in loss_dict.items()}
            cur_train_loss_dict = {k: v / (idx + 1) for k, v in cur_loss_dict.items()}
            status = self.creat_loss_message(cur_train_loss_dict,epoch)

            train_iterator.set_description(status)
            idx+=1
        train_iterator.close()

        return cur_train_loss_dict

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        cur_loss_dict, eval_metric_dict = {}, {}
        eval_iterator = tqdm(self.test_loader, total=self.test_loader.__len__(), ncols=180)
        idx = 0
        for data in eval_iterator:
            data = self.to_device(data)

            out = self.model_forward(data)
            total_loss, loss_dict = self.cal_loss(out, data)


            cur_loss_dict = {k: cur_loss_dict.get(k, 0.0) + v.item() for k, v in loss_dict.items()}
            cur_eval_loss_dict = {k: v / (idx + 1) for k, v in cur_loss_dict.items()}
            if self.do_metric:
                metrics_dict = self.cal_metrics(out, data)
                eval_metric_dict = {k: eval_metric_dict.get(k, 0.0) + v for k, v in metrics_dict.items()}
                cur_eval_metric_dict = {k: v / (idx + 1) for k, v in eval_metric_dict.items()}
                eval_dict = dict(cur_eval_loss_dict, **cur_eval_metric_dict)

            else:
                eval_dict = cur_eval_loss_dict

            status = self.creat_loss_message(eval_dict, epoch, stage='eval')

            eval_iterator.set_description(status)
            idx += 1
        eval_iterator.close()
        return eval_dict

    def train(self):
        self.model.train()

        for epo in range(self.cur_epo, self.args.epochs + 1):
            self.cur_epo = epo
            train_loss_dict = self.train_epoch(epo)

            # if bool(self.args.scheduler):
            #     self.scheduler.step(train_loss_dict['loss_total'])
            #
            # self.writer.add_scalars('train_loss/total_scalars', {'total_loss': train_loss_dict['loss_total']}, epo)
            # del train_loss_dict['loss_total']
            # self.writer.add_scalars('train_loss/dict_scalars',train_loss_dict, epo)
            #
            # if self.do_eval and epo%self.eval_gap==0:
            #     eval_loss_dict = self.eval_epoch(epo)
            #     self.writer.add_scalars('eval_loss/total_scalars', {'total_loss': eval_loss_dict['loss_total']}, epo)
            #
            #     if eval_loss_dict['loss_total']< self.best_eval_loss:
            #         torch.save(self.model.state_dict(),
            #                    osp.join(self.snap_path,'epo_{}_{}.pt').format(epo, self.args.desc))
            #         self.best_eval_loss = eval_loss_dict['loss_total']
            #
            #     del eval_loss_dict['loss_total']
            #     self.writer.add_scalars('eval_loss/dict_scalars', eval_loss_dict, epo)


    @staticmethod
    def creat_loss_message(loss_dict, epoch_num,stage='train'):
        status = '| '.join(['%s=%.3f' % (k, v) for k,v in loss_dict.items() if k != 'loss_total'])
        return stage+'_epo{%d}: t_loss=%.3f | %s' %(epoch_num, loss_dict['loss_total'], status)


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - x2y_signed: Torch.Tensor
            the sign distance from x to y
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    # ch_dist = chd.ChamferDistance()
    # x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)
    x_near, y_near, xidx_near, yidx_near = chd.ChamferDistance(x,y)


    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop



def geodesic_loss(mat1, mat2):
    batch_size = mat1.shape[0]
    m1 = mat1.reshape(-1, 3, 3)
    m2 = mat2.reshape(-1, 3, 3)
    total_size = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    cos = ( m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(total_size).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(total_size).cuda())*-1)
    theta = torch.acos(cos)
    #theta = torch.min(theta, 2*np.pi - theta)
    error = theta.mean()
    return error

