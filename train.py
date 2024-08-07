import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from torch.utils.data import DataLoader
import contact2mesh.arguments as arguments
from contact2mesh.data.cp_loader import ContactDBDataset
from contact2mesh.data.gb_loader import GrabNetDataset
from contact2mesh.data.obman_loader import ObManDataset

from contact2mesh.data.dataloader_Sampler import TwoStreamBatchSampler
from contact2mesh.train.trainer import Train_Param2Mesh,Trainer_PNet_mscale
import numpy as np
import torch

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.multiprocessing.set_start_method('spawn')
    args = arguments.train_cvae2_parse_args()

    if'GrabNet' in args.desc:
        args.dataset_dir = '/home/dataset/haoming/grabnet/new_data/GRAB_V00'
        args.num_workers=2
        ds_test = GrabNetDataset(dataset_dir=args.dataset_dir,ds_name='test', load_params=True, load_contact=False)
        ds_train = GrabNetDataset(dataset_dir=args.dataset_dir,ds_name='train', load_params=True, load_contact=False)

        test_loader = DataLoader(ds_test,batch_size=args.batch_size, shuffle=True,drop_last=True)
        train_loader = DataLoader(ds_train,batch_size=args.batch_size, shuffle=True,drop_last=True, num_workers=args.num_workers)

    if 'Obman' in args.desc:
        args.aug = True if 'aug' in args.desc else False
        args.train_dataset = '/home/dataset/haoming/ObMan_unzip/pkl/obman_train_1.pkl'
        args.test_dataset = '/home/dataset/haoming/ObMan_unzip/pkl/obman_val_1.pkl'
        ds_train = ObManDataset(args.train_dataset, is_aug=args.aug, is_param=True)
        ds_test = ObManDataset(args.test_dataset, is_param=True)

        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    else:
        args.aug = True if 'aug' in args.desc else False
        args.rot = True if 'rot' in args.desc else False
        args.sample_idx_path=None

        args.train_dataset = '/home/dataset/haoming/ContactPose/pkl/split/contactpose_fixed_train_replace.pkl'
        args.test_dataset ='/home/dataset/haoming/ContactPose/pkl/split/contactpose_fixed_test_replace.pkl'
        args.obj_info = '/home/dataset/haoming/ContactPose/pkl/split/obj_info_dict.pkl'
        train_dataset = ContactDBDataset(args.train_dataset, train=True, is_aug=args.aug, is_rot=args.rot,is_param=False)
        test_dataset = ContactDBDataset(args.test_dataset, train=True, is_param=False)

        if args.sample_idx_path:
            primary_idx, secondary_idx =list(np.load(args.sample_idx_path[0])), list(np.load(args.sample_idx_path[1]))
            batch_sampler = TwoStreamBatchSampler(primary_idx, secondary_idx, args.batch_size, args.batch_size//2)
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)


    if args.use_model == 'Param2Mesh':
        args.obj_enc_pth = "/remote-home/share/lqt/grasp_contactmap14/grasp_envs/DAPG/model/checkpoint/epo_180_REC_SPnetDenseEncoder_shapenet55_normrot512.pt"
        args.vpe_path = '/home/haoming/GrabNet/grabnet/configs/verts_per_edge.npy'
        args.c_weights_path = '/home/haoming/GrabNet/grabnet/configs/rhand_weight.npy'
        # args.obj_enc_fixed = True
        trainer = Train_Param2Mesh(args, train_loader, test_loader)

    elif args.use_model == 'PNet_mscale':
        args.glob_feat=True
        args.obj_enc_pth = None
        trainer = Trainer_PNet_mscale(args, train_loader, test_loader)

    all_str = ''
    for key, val in vars(args).items():
        all_str += '--{}={} '.format(key, val)

    print(all_str)  # Convert to dict and print
    args.all_str = all_str


    trainer.train()


