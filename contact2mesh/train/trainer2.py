import torch
import numpy as np
import os.path as osp
from torch import nn, optim
from contact2mesh.models.cvae.cvae import CVAE_trans
from torch.utils.tensorboard import SummaryWriter
import contact2mesh.utils.util as util
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, args, train_loader, test_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cur_epo = 0
        self.args = args
        self.model = CVAE_trans(t_args=args).to(self.device)

        self.train_loader = train_loader

        self.total_loss_meter = util.AverageMeter('t_Loss', ':.3f')
        #self.mse_loss_meter = util.AverageMeter('m_Loss', ':.3f')
        self.kl_loss_meter = util.AverageMeter('k_Loss', ':.3f')
        self.bce_loss_meter = util.AverageMeter('b_Loss', ':.3f')

        # bin_weights = torch.Tensor(np.loadtxt(util.DEEPCONTACT_BIN_WEIGHTS_FILE)).to(self.device)
        # bin_weights = bin_weights[[0, 9]]


        # self.criterion = torch.nn.MSELoss()
        self.bce_criterion = torch.nn.BCELoss()
        # self.class_criterion = torch.nn.CrossEntropyLoss()
        # self.dice_criterion = util.BinaryDiceLoss()

        if bool(self.args.checkpoints):
            self.model_load()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)


        if bool(args.scheduler):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        self.writer = SummaryWriter(log_dir='./contact2mesh/runs/' + args.desc)

    def model_load(self):
        self.cur_epo = int(osp.basename(self.args.checkpoints).split('_')[1])
        self.model.load_state_dict(torch.load(self.args.checkpoints, map_location=self.device))

    def meter_reset(self):
        self.total_loss_meter.reset()
        #self.mse_loss_meter.reset()
        self.kl_loss_meter.reset()
        self.bce_loss_meter.reset()

    def cal_gen_loss(self, out, contact_gt, class_gt=None):
        """
        Caluate the loss for the Framework
        :param contact_obj_gt: ground truth of object contact map (already sampled), (B,N,1)
        :param network_out: including the object contact prediction (B,10, N), and latent code (B,latentD)
        :param sampled_verts_idx: The sampled verts idx for the object contact and verts , (B, N)

        """
        contact_gt = contact_gt.squeeze(-1)
        # caculate the kl, contact loss
        q_z = torch.distributions.normal.Normal(out['mean'], out['std'])
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(contact_gt.dtype),
            scale=torch.tensor(np.ones([self.args.batch_size, self.args.latentD]), requires_grad=False).to(
                self.device).type(contact_gt.dtype))
        kl_loss = self.args.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        # mse_loss = self.criterion(out['pred_contact'], contact_gt)*16

        bce_loss = self.bce_criterion(out['pred_contact'], util.soft_to_binary(contact_gt).float())
        # cls_loss = self.class_criterion(out['pred_class'], class_gt)*0.3

        total_loss =  kl_loss + bce_loss
        #dice_eval = util.dice_eval((out['hard_contact'][:,:,0]!=0)*1, (contact_obj_gt[:,:,0]!=0)*1)

        loss_dict = {#'mse_loss': mse_loss.item(),
                     'kl_loss': kl_loss.item(),
                     'bce_loss': bce_loss.item(),
                     }

        return total_loss, loss_dict

    def cal_dis_loss(self, real_pred, fake_pred):
        d_real_loss = self.adversarial_criterion(real_pred, self.vaild)
        d_fake_loss = self.adversarial_criterion(fake_pred, self.fake)

        return (d_real_loss + d_fake_loss) / 2

    def train_epoch(self, epoch):
        self.model.train()
        self.meter_reset()

        train_iterator = tqdm(self.train_loader, total=self.train_loader.__len__())
        idx = 0
        for data in train_iterator:
            data = util.dict_to_device(data, self.device)
            # onehot = util.class_to_onehot(data['obj_class'],self.device)

            out = self.model(data['obj_sampled_verts_gt'].transpose(1,2), data['obj_sampled_contact_gt'].transpose(1,2))

            total_loss, loss_dict = self.cal_gen_loss(out, data['obj_sampled_contact_gt'])

            self.total_loss_meter.update(total_loss.item())
            #self.mse_loss_meter.update(loss_dict['mse_loss'])
            self.kl_loss_meter.update(loss_dict['kl_loss'])
            self.bce_loss_meter.update(loss_dict['bce_loss'])

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            status = 'epo{}:t_loss={:.3f} - kl={:.3f} -bce={:.3f}'.format(
                epoch, self.total_loss_meter.avg, self.kl_loss_meter.avg, self.bce_loss_meter.avg)

            train_iterator.set_description(status)
        train_iterator.close()


    def train(self):
        for epo in range(self.cur_epo, self.args.epochs + 1):
            self.cur_epo = epo
            self.train_epoch(epo)

            if bool(self.args.scheduler):
                self.scheduler.step(self.total_loss_meter.avg)

            self.writer.add_scalars('loss/total_scalars', {'total_loss': self.total_loss_meter.avg}, epo)

            self.writer.add_scalars('loss/dict_scalars',
                                    {#'mse_loss': self.mse_loss_meter.avg,
                                     'kl_loss': self.kl_loss_meter.avg,
                                     'bce_loss': self.bce_loss_meter.avg
                                     }, epo)

            if epo % 10 == 0 and epo>100:
                g_path = util.make_dirs('./contact2mesh/checkpoints/'+self.args.desc)
                torch.save(self.model.state_dict(),
                           osp.join(g_path,'epo_{}_{}.pt').format(epo, self.args.desc))

                # d_path = util.make_dirs('./contact2mesh/checkpoints/'+self.args.desc +'/discriminator/')
                # torch.save(self.discriminator.state_dict(),
                #            osp.join(d_path, 'epo_{}_{}.pt').format(epo, self.args.desc))

            # print('\n')


if __name__ == '__main__':
    a = 1
