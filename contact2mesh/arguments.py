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


def add_transformer_params(parser):
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--model_name_or_path", default='/remote-home/lihaoming/haoming/Contact2Mesh/contact2mesh/models/transformer/bert-base-uncased/',
                        type=str,
                        required=False, help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                             "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--input_dims", default=[128 + 3, 64, 16], type=str,  # [1027,512,128], [128 + 3, 64, 16]
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_dims", default=[128, 32, 8], type=str,  # [1024,256,64], [128, 32, 8]
                        help="The Image Feature Dimension.")
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")


def train_cvae_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--use_model', default='PNet_mscale', type=str)
    parser.add_argument('--kl-coef', default=2e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=800, type=int)

    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--desc', default='CVAEPNet_mscale_fixed_split_ld64_Obman_augrot', type=str)

    #########################################################
    # Transformer parameters
    #########################################################

    # add_transformer_params(parser)

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


def train_cvae2_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=28, type=int)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--use_model', default='PNet2_mscale', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--kl-coef', default=5e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--desc', default='PNet2_mscale_fixed_split_ld64_ContactPose_augrot_bs28', type=str)

    #########################################################
    # Transformer parameters
    #########################################################
    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


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

def train_diffusion_v2_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_model', default='ddpm_pnet_mscale', type=str)
    parser.add_argument('--num_diffusion_iters', default=200, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--globalD', default=1024, type=int) #contact_encoder的 global维度 default=1024

    parser.add_argument('--timeD', default=3, type=int) #pcd_encoder的 point local维度
    parser.add_argument('--localD', default=64, type=int) #pcd_encoder的 point local维度
    parser.add_argument('--pcd_globalD', default=1024, type=int)#pcd_encoder的 point global维度  default=512
    parser.add_argument('--glob_feat', default=True, type=bool)# glob_feat=False 代表point_feat只有local（B，64,2048，=True代表（B，1024+64,2048）
    parser.add_argument('--only_point_feat', default=False, type=bool) # 只要Aepcd只返回local_feat， 不计算global feat了
    parser.add_argument('--use_local_dec', default=True, type=bool) # 用local 解码/或者用global+latent直接解码
    parser.add_argument('--objective', default='pred_x0',choices=['pred_x0', 'pred_noise'], type=str) #pred_x0 or pred_noise
    parser.add_argument('--diff_sched', default='ddim',choices=['ddim', 'ddpm'], type=str) #pred_x0 or pred_noise
    parser.add_argument('--use_ema', default=False, type=bool) #pred_x0 or pred_noise

    parser.add_argument('--point_dim', default=4, type=int) #

    parser.add_argument('--obj_enc_pth', default=None)

    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--kl-coef', default=5e-4, type=float, help='KL divergence coefficent for Coarsenet training')

    # parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--desc', default='DDPM_ContactPose_augrot_bs32', type=str)
    parser.add_argument( "--exp_dir", default='./contact2mesh/models/diffusion/config/ddpm_contact/ddpm_pnet_mscale')
    parser.add_argument("--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'")
    parser.add_argument("--batch_size", "-b", default=64, type=int)
    parser.add_argument( "--workers", "-w", default=16, type=int)

    # parameter for pointwisenet training.
    parser.add_argument('--latent_dim', type=int, default=256) #j相当于pcd的globalD
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.05)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
    parser.add_argument('--contact_vis', default=False, type=int) # tensorboard可视化conact map

    #########################################################
    # Transformer parameters
    #########################################################
    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


    parse_dataset(args)

    return args



def train_kpc_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--use_model', default='KPC', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--kl-coef', default=5e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--desc', default='KPC_fixed_split_ld64_ContactPose_rot_bs32', type=str)

    #########################################################
    # Transformer parameters
    #########################################################
    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


    parse_dataset(args)

    return args

def train_graspnet_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--use_model', default='GraspNet', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--save_path', default='/home/dataset/haoming/snapshot_runs', type=str)

    parser.add_argument('--desc', default='GraspNet_fix_split_ld64_ContactPose_augrot_add_consist', type=str)

    #########################################################
    # Transformer parameters
    #########################################################

    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)

    parse_dataset(args)

    return args

def train_graspnet_refine_parse_args():
    parser = argparse.ArgumentParser(description='grasp networks fine-tune')
    parser.add_argument('--lr', default=6.25e-6, type=float)
    parser.add_argument('--batch_size', default=115, type=int)
    parser.add_argument('--use_model', default='GraspNet', type=str)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--gpu_num', default=6, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--save_path', default='/home/dataset/haoming/snapshot_runs', type=str)

    parser.add_argument('--desc', default='GraspNetGen_refine_fixed_ContactPose_aug2_gt', type=str)

    #########################################################
    # Transformer parameters
    #########################################################

    # add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


    parse_dataset(args)

    return args

def train_cvae3_parse_args():
    parser = argparse.ArgumentParser(description='cvae networks training')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=512, type=int)
    parser.add_argument('--glob_feat', default=True, type=int) # glob_feat=False 代表point_feat只有local（B，64,2048，=True代表（B，1024+64,2048）
    parser.add_argument('--only_point_feat', default=False, type=int) # 只要local_feat
    parser.add_argument('--localD', default=64, type=int) #pcd_encoder的 point local维度
    parser.add_argument('--pcd_globalD', default=512, type=int)#pcd_encoder的 point global维度


    parser.add_argument('--use_model', default='PNet_mscale', type=str)
    parser.add_argument('--kl-coef', default=1e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')
    parser.add_argument('--epochs', default=600, type=int)

    parser.add_argument('--checkpoints', default=None, type=str)
    parser.add_argument('--desc', default='CVAEPNet_mscale_rand_ld512_ContactPose_aug_localglob', type=str)

    #########################################################
    # Transformer parameters
    #########################################################

    # add_transformer_params(parser)
    args = parser.parse_args()
    args_print(args)

    parse_dataset(args)

    return args



def run_visual_parse_args():
    parser = argparse.ArgumentParser(description='visualization')

    parser.add_argument('--gpu_num', default='6', type=str)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--latentD', default=128, type=int)
    parser.add_argument('--conditionD', default=27, type=int)
    parser.add_argument('--obj_select', default=['apple', 'mouse', 'mug'], type=list)
    parser.add_argument('--sam_k', default=3, type=int)

    parser.add_argument('--save_path', default='/home/haoming/Contact2Mesh/contact2mesh/img_save', type=str)
    parser.add_argument('--checkpoint',
                        default='./contact2mesh/checkpoints/CVAEResMlps_Train_fixed_onehot/epo_900_CVAEResMlps_fixed_mse.pt',
                        type=str)
    parser.add_argument('--desc', default='visual', type=str)

    #########################################################
    # Transformer parameters
    #########################################################
    add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)


    parse_dataset(args)

    return args


def contact_eval_parse_args():
    parser = argparse.ArgumentParser(description='networks evaluation')
    parser.add_argument('--use_model', default='PNet_mscale', type=str)
    parser.add_argument('--key-str', default='ld64', type=str,
                        help='the key word to save imgs or metrics files')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--globalD', default=1024, type=int) #contact_encoder的 global维度 default=1024
    parser.add_argument('--localD', default=64, type=int) #pcd_encoder的 point local维度
    parser.add_argument('--pcd_globalD', default=1024, type=int)#pcd_encoder的 point global维度  default=512

    # parser.add_argument('--obj_id_path', default='/home/dataset/haoming/ObMan_unzip/pkl/class_id_train_1.pkl', type=str)
    parser.add_argument('--obj_id_path', default=None, type=str)

    parser.add_argument('--save_path', default='/home/haoming/Contact2Mesh/contact2mesh/Save', type=str)
    parser.add_argument('--desc', default='CVAE_eval_fixed_Obman', type=str)
    parser.add_argument('--save-img', default=True, type=bool,
                        help='the key word to save imgs or metrics files')

    parser.add_argument('--point_dim', default=4, type=int) #
    parser.add_argument('--glob_feat', default=True, type=int)# glob_feat=False 代表point_feat只有local（B，64,2048，=True代表（B，1024+64,2048）
    parser.add_argument('--only_point_feat', default=False, type=int) # 只要Aepcd只返回local_feat， 不计算global feat了
    parser.add_argument('--use_local_dec', default=True, type=int) # 用local 解码/或者用global+latent直接解码
    parser.add_argument('--pcd_load_dec', default=True, type=int) # Aepcd 是否导入解码模块

    #########################################################
    # Transformer parameters
    #########################################################
    add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)

    parse_dataset(args)

    return args


def grasp_eval_parse_args():
    parser = argparse.ArgumentParser(description='networks evaluation')
    parser.add_argument('--use_model', default='PNet_mscale', type=str)
    parser.add_argument('--key-str', default='ld64', type=str,
                        help='the key word to save imgs or metrics files')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--save_path', default='/home/haoming/Contact2Mesh/contact2mesh/Save', type=str)
    parser.add_argument('--desc', default='Grasp_eval_ContactPose', type=str)
    parser.add_argument('--save-img', default=False, type=bool,
                        help='save grasp img or not')
    parser.add_argument('--save_print', default=True, type=bool,
                        help='save grasp eval metrics or not')

    #########################################################
    # Transformer parameters
    #########################################################
    add_transformer_params(parser)

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

def grasp_eval_parse_args():
    parser = argparse.ArgumentParser(description='networks evaluation')
    parser.add_argument('--use_model', default='PNet_mscale', type=str)
    parser.add_argument('--key-str', default='ld64', type=str,
                        help='the key word to save imgs or metrics files')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--save_path', default='/remote-home/share/datasets/haoming', type=str)
    parser.add_argument('--desc', default='Grasp_eval_ContactPose', type=str)


    #########################################################
    # Transformer parameters
    #########################################################
    add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)

    parse_dataset(args)

    return args


def tsne_parse_args():
    parser = argparse.ArgumentParser(description='tsne_analysis')
    parser.add_argument('--use_model', default='Param2Mesh', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--latentD', default=64, type=int)
    parser.add_argument('--save_path', default='/home/haoming/Contact2Mesh/contact2mesh/Save', type=str)
    parser.add_argument('--desc', default='Grasp_eval_ContactPose', type=str)

    #########################################################
    # Transformer parameters
    #########################################################
    add_transformer_params(parser)

    args = parser.parse_args()
    args_print(args)
    return args

if __name__ == '__main__':
    import os
    from glob import glob




    obj_names = []
    datasets_path = '/home/dataset/haoming/ContactPose/data/contactpose_data'
    p_paths = glob(os.path.join(datasets_path, '*'))
    for p_i in p_paths:
        p_i_obj_list = [os.path.basename(p) for p in glob(p_i + '/*')]
        for obj_name_i in p_i_obj_list:
            if '.zip' in obj_name_i:
                continue
            if obj_name_i not in obj_names:
                obj_names.append(obj_name_i)

    obj_class_dict = {}
    for idx, obj_i in enumerate(obj_names):
        obj_class_dict.update({obj_i: idx})

    print(obj_class_dict)
    a = 1

{'apple': 0, 'banana': 1, 'binoculars': 2, 'bowl': 3, 'camera': 4, 'cell_phone': 5, 'cup': 6, 'eyeglasses': 7,
 'flashlight': 8, 'hammer': 9, 'headphones': 10, 'knife': 11, 'light_bulb': 12, 'mouse': 13, 'mug': 14, 'pan': 15,
 'ps_controller': 16, 'scissors': 17, 'stapler': 18, 'toothbrush': 19, 'toothpaste': 20, 'utah_teapot': 21,
 'water_bottle': 22, 'wine_glass': 23, 'door_knob': 24, 'palm_print': 25, 'train': 26}
