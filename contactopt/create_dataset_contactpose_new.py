 # Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import path
import sys
import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from contactopt.hand_object import HandObject
from contactopt.util import *

sys.path.append('../Contact2Mesh')  # Change this path to point to the ContactPose repo
from contactopt.dataset import get_object_names, ContactPose
from contact2mesh.utils.util import TRAIN_OBJ_NAMES, TEST_OBJ_NAMES
object_cut_list = []

# object_cut_list = ['eyeglasses']


def get_all_contactpose_samples(cfg):
    """
    Gets all participants and objects from ContactPose
    Cuts out grasps with two hands or grasps using left hand
    :return: list of (participant_num, intent, object_name, ContactPose_object)
    """
    samples = []

    print('Reading ContactPose dataset')

    iter = range(1, 51)
    intents = ['handoff', 'use']
    if cfg.p_id:
        iter = cfg.p_id
    if cfg.intents:
        intents = cfg.intents

    for participant_id in tqdm(iter):
        for intent in intents:
            for object_name in get_object_names(participant_id, intent, cfg.data_path):
                cp = ContactPose(participant_id, intent, object_name, data_path=cfg.data_path, load_mano=False)
                if cp._valid_hands != [1]:  # If anything else than just the right hand, remove
                    continue
                if cp.object_name not in args.select_objs:
                    continue

                samples.append((participant_id, intent, object_name, cp))

    print('Valid ContactPose samples:', len(samples))

    return samples



def generate_contactpose_dataset(dataset, output_file, low_p, high_p, num_pert=1, aug_trans=0.02, aug_rot=0.05,
                                 aug_pca=0.3, cfg=None):
    """
    Generates a dataset pkl file and does preprocessing for the PyTorch dataloader
    :param dataset: List of ContactPose objects
    :param output_file: path to output pkl file
    :param low_p: Lower split location of the dataset, [0-1)
    :param high_p: Upper split location of the dataset, [0-1)
    :param num_pert: Number of random perturbations which are computed for every true dataset sample
    :param aug_trans: Std deviation of hand translation noise added to the datasets, meters
    :param aug_rot: Std deviation of hand rotation noise, axis-angle radians
    :param aug_pca: Std deviation of hand pose noise, PCA units
    """
    # low_split = int(len(dataset) * low_p)
    # high_split = int(len(dataset) * high_p)
    # dataset = dataset[low_split:high_split]

    if len(object_cut_list) > 0:
        dataset = [s for s in dataset if s[2] not in object_cut_list]
        print('Some objects are being removed', object_cut_list)

    def process_sample(s, idx):
        ho_gt = HandObject()
        ho_gt.load_from_contactpose(s[3])
        sample_list = []
        # print('Processing', idx)

        for i in range(num_pert):
            # Since we're only saving pointers to the data, it's memory efficient
            sample_data = dict()

            ho_aug = HandObject()
            aug_t = np.random.randn(3) * aug_trans
            aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(15) * aug_pca)).astype(np.float32)
            ho_aug.load_from_ho(ho_gt, aug_p, aug_t)

            sample_data['ho_gt'] = ho_gt
            sample_data['ho_aug'] = ho_aug
            # sample_data['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), SAMPLE_VERTS_NUM)
            sample_data['obj_sampled_idx'] = np.random.choice(len(ho_gt.obj_verts), SAMPLE_VERTS_NUM, replace=False)

            sample_data['hand_feats_gt'], sample_data['obj_feats_gt'] = ho_gt.generate_pointnet_features(
                sample_data['obj_sampled_idx'])
            sample_data['hand_feats_aug'], sample_data['obj_feats_aug'] = ho_aug.generate_pointnet_features(
               sample_data['obj_sampled_idx'])

            sample_list.append(sample_data)

        return sample_list

    def process_objects(s, obj_sampled_idx_dict):
        """
        定义一个新的数据集，只包含一个物体的点云，对应多个contact map, 且每个contact map 对应的采样点云保持一致。
        """
        ho_gt = HandObject()
        ho_gt.load_from_contactpose(s[3])

        sample_list = []
        for i in range(num_pert):
            sample_data = dict()

            ho_aug = HandObject()
            aug_t = np.random.randn(3) * aug_trans
            aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(15) * aug_pca)).astype(np.float32)
            ho_aug.load_from_ho(ho_gt, aug_p, aug_t)

            sample_data['ho_gt'] = ho_gt
            sample_data['ho_aug'] = ho_aug

            sample_data['obj_sampled_idx'] = obj_sampled_idx_dict[s[2]]

            sample_data['hand_feats_gt'], sample_data['obj_feats_gt'] = ho_gt.generate_pointnet_features(
                sample_data['obj_sampled_idx'])
            sample_data['hand_feats_aug'], sample_data['obj_feats_aug'] = ho_aug.generate_pointnet_features(
               sample_data['obj_sampled_idx'])

            sample_list.append(sample_data)

        return sample_list

    obj_names = set([s[2] for s in dataset])
    obj_sampled_idx_dict = fixed_sample_idx(obj_names)
    parallel = True
    if parallel:
        num_cores = multiprocessing.cpu_count()
        print('Running on {} cores'.format(num_cores))
        if args.mode=='rand':
            all_data_2d = Parallel(n_jobs=num_cores)(delayed(process_sample)(s, idx) for idx, s in enumerate(tqdm(dataset)))
        else:
            all_data_2d = Parallel(n_jobs=num_cores)(delayed(process_objects)(s, obj_sampled_idx_dict) for idx, s in enumerate(tqdm(dataset)))

        all_data = [item for sublist in all_data_2d for item in sublist]  # flatten 2d list
    else:
        all_data = []  # Do non-parallel
        for idx, s in enumerate(tqdm(dataset)):
            if args.mode == 'rand':
                all_data.extend(process_sample(s, idx))
            else:
                all_data.extend(process_objects(s, obj_sampled_idx_dict))


    # low_split = int(len(all_data) * low_p)
    # high_split = int(len(all_data) * high_p)
    #
    # train_all_data = all_data[low_split:high_split]
    # test_all_data = all_data[high_split:]

    print('Writing pickle file, often slow and freezes computer')
    pickle.dump(all_data, open(output_file, 'wb'))
    # pickle.dump(train_all_data, open(output_file[0], 'wb'))
    # pickle.dump(test_all_data, open(output_file[1], 'wb'))


if __name__ == '__main__':
    import contactopt.arguments as argument

    args = argument.create_contact_pose_dataset_args()
    # train_file = '../data/contactpose_train.pkl'
    # test_file = '../data/contactpose_test.pkl'

    train_file = '/home/dataset/haoming/ContactPose/pkl/split/contactpose_rand_train.pkl'
    test_file = '/home/dataset/haoming/ContactPose/pkl/split/contactpose_rand_test.pkl'

    # fine_file = 'D:\pycharm_project\Contact2Mesh\data/contactpose_test.pkl'
    data_path = "/home/dataset/haoming/ContactPose/data"

    args.train_file = train_file
    args.test_file = test_file
    args.data_path = data_path
    args.p_id = None # list(range(1,51))
    args.intents = ['use', 'handoff']  # ['use', 'handoff']
    args.mode='fixed'

    aug_trans = 0.05
    aug_rot = 0.1
    aug_pca = 0.5

    ###################### Testset Packpage ########################
    args.select_objs = TEST_OBJ_NAMES
    args.mode = 'rand' if 'fixed' not in args.test_file else 'fixed'

    contactpose_dataset = get_all_contactpose_samples(cfg=args)
    generate_contactpose_dataset(contactpose_dataset, test_file, 0.0, 1.0, num_pert=1, aug_trans=aug_trans,
                                 aug_rot=aug_rot, aug_pca=aug_pca, cfg=args)
    print('Testset Packpage Finish!')

    ###################### Trainset Packpage ########################
    args.select_objs = TRAIN_OBJ_NAMES
    args.mode = 'rand' if 'fixed' not in args.train_file else 'fixed'

    contactpose_dataset = get_all_contactpose_samples(cfg=args)
    generate_contactpose_dataset(contactpose_dataset, train_file, 0.0, 1.0, num_pert=1, aug_trans=aug_trans,
                                 aug_rot=aug_rot, aug_pca=aug_pca, cfg=args)
    print('Trainset Packpage Finish!')




    # if os.path.exists(os.path.dirname(train_file)) and os.path.exists(os.path.dirname(test_file)):
    #     contactpose_dataset = get_all_contactpose_samples(cfg=args)
    #     # Generate Perturbed ContactPose
    #     generate_contactpose_dataset(contactpose_dataset, train_file, 0.0, 0.8, num_pert=1, aug_trans=aug_trans, aug_rot=aug_rot, aug_pca=aug_pca,cfg=args)
    #     # generate_contactpose_dataset(contactpose_dataset, test_file, 0.8, 1.0, num_pert=1, aug_trans=aug_trans, aug_rot=aug_rot, aug_pca=aug_pca)
