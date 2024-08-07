import argparse
import datetime
import os.path as osp
import os


def parse_dataset(args):
    """ Converts the --split argument into a dataset file """
    # base_path = 'D:\PycharmProjects\ContactOpt'
    data_path = '/remote-home/share/datasets/haoming/ContactPose/pkl/split'
    args.bps_dir = './contact2mesh/configs/bps.npz'
    args.train_dataset = osp.join(data_path, 'contactpose_fixed_train_replace.pkl')
    args.test_dataset = osp.join(data_path, 'contactpose_fixed_test_replace.pkl')
    args.sample_idx_path = [osp.join(data_path, 'contactpose_fixed_train_primary_idxs.npy'),
                            osp.join(data_path, 'contactpose_fixed_train_secondary_idxs.npy')]

def args_print(args):
    if args.desc == '':
        args.desc = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    all_str = ''
    for key, val in vars(args).items():
        all_str += '--{}={} '.format(key, val)

    print(all_str)  # Convert to dict and print
    args.all_str = all_str


def train_cvae_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--use_model', default='Param2Mesh', type=str)
    parser.add_argument('--kl-coef', default=2e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=800, type=int)

    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--desc', default='Param2mesh_fixed_split_ld64_ContactPose_augrot', type=str)

    #########################################################
    # Transformer parameters
    #########################################################

    args = parser.parse_args()

    if args.desc == '':
        args.desc = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # all_str = ''
    # for key, val in vars(args).items():
    #     all_str += '--{}={} '.format(key, val)
    #
    # print(all_str)  # Convert to dict and print
    # args.all_str = all_str

    parse_dataset(args)

    return args


def train_diffusion_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_model', default='ddim_param2mesh',choices=['ddim_pnet_mscale', 'ddim_param2mesh'], type=str)
    parser.add_argument('--point_dim', default=4, type=int) #
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--globalD', default=1024, type=int) #contact_encoder的 global维度
    parser.add_argument('--glob_feat', default=True, type=int)# glob_feat=False 代表point_feat只有local（B，64,2048，=True代表（B，1024+64,2048）

    parser.add_argument('--only_point_feat', default=False, type=int) # 只要local_feat
    parser.add_argument('--localD', default=64, type=int) #pcd_encoder的 point local维度
    parser.add_argument('--pcd_globalD', default=1024, type=int)#pcd_encoder的 point global维度
    parser.add_argument('--use_local_dec', default=True, type=int) # 用local 解码/或者用global+latent直接解码
    parser.add_argument('--pcd_load_dec', default=True, type=int) # Aepcd 是否导入解码模块
    parser.add_argument('--objective', default='pred_x0', type=str) #
    parser.add_argument('--diff_sched', default='ddim',choices=['ddim', 'ddpm'], type=str) #pred_x0 or pred_noise

    parser.add_argument('--cvae_ckpt',default=None, type=str)
    parser.add_argument('--diff_ckpt', default=None, type=str)

    parser.add_argument('--obj_enc_pth', default=None)

    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--kl-coef', default=5e-4, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--desc', default='CombineDiffusion_Obman_aug_bs64', type=str)
    parser.add_argument( "--exp_dir", default='./contact2mesh/models/diffusion/config/stage2_diff_cond')
    parser.add_argument("--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'")
    parser.add_argument("--batch_size", "-b", default=64, type=int)
    parser.add_argument( "--workers", "-w", default=1, type=int)
    parser.add_argument('--write_latent_code', default=False, type=int) # 写入latent_code到数据集里
    parser.add_argument('--contact_vis', default=False, type=int) # tensorboard可视化conact map
    parser.add_argument('--contact_packpage', default=False, type=int) # 打包生成的contact_map
    parser.add_argument('--batch_generation', default=False, type=int) # Obman数据太多，Batch进行生成

    #########################################################
    # Transformer parameters
    #########################################################
    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


    parse_dataset(args)

    return args


def hand_optim_contactpose_parse_args():
    parser = argparse.ArgumentParser(description='hand optimization')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--steps1', default=31, type=int) #51
    parser.add_argument('--steps2', default=101, type=int) #101

    parser.add_argument('--s1w', default={'consist':1e-3, 'hc':2., 'ftc':3.0, 'pentr':1000.}, type=dict)
    parser.add_argument('--s2w', default={'consist':1e-3, 'hc':2., 'ftc':3.0, 'pentr':1000.}, type=dict)

    parser.add_argument('--desc', default='grasp_optim_ContactPose', type=str)

    parser.add_argument('--latentD', default=64, type=int)
    args = parser.parse_args()
    args_print(args)
    return args

def hand_optim_obman_parse_args():
    parser = argparse.ArgumentParser(description='hand optimization')
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--steps1', default=121, type=int)
    parser.add_argument('--s1w', default={'consist':2e-4, 'hc':2.0, 'ftc':5.0, 'pentr':4000.}, type=dict)
    parser.add_argument('--s2w', default={'consist':1e-3, 'hc':2.0, 'ftc':5.0, 'pentr':4000.}, type=dict)

    parser.add_argument('--desc', default='grasp_optim_obman', type=str)

    parser.add_argument('--latentD', default=64, type=int)
    args = parser.parse_args()
    args_print(args)
    return args

def hand_optim_ho3d_parse_args():
    parser = argparse.ArgumentParser(description='hand optimization')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--steps1', default=51, type=int)
    parser.add_argument('--s1w', default={'consist':1e-3, 'hc':2.5,'pentr':2000.}, type=dict)
    parser.add_argument('--s2w', default={'consist':1e-3, 'hc':2.5,'pentr':1000.}, type=dict)

    parser.add_argument('--desc', default='grasp_optim_ho3d', type=str)

    parser.add_argument('--latentD', default=64, type=int)
    args = parser.parse_args()
    args_print(args)
    return args


