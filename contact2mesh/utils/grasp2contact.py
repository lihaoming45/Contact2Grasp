import torch
from pytorch3d.ops.knn import knn_gather, knn_points
import numpy as np
import time
from visual import obj_mesh_instance2

from contactopt.diffcontact import calculate_contact_capsule

def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx


def get_pseudo_cmap(nn_dists):
    '''
    calculate pseudo contactmap: 0~3cm mapped into value 1~0
    :param nn_dists: object nn distance [B, N] or [N,] in meter**2
    :return: pseudo contactmap [B,N] or [N,] range in [0,1]
    '''
    nn_dists = 100.0 * torch.sqrt(nn_dists)  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists*2) -0.5)
    return cmap



if __name__ == '__main__':
    import pickle
    import os
    from open3d import geometry as o3dg
    from open3d import visualization as o3dv
    from open3d import utility as o3du
    from open3d import io
    import trimesh
    import json
    # import pclpy
    # from pclpy import pcl

    hand_name='hand_mesh1'
    obj_name='mug'
    # hand_mesh = trimesh.load('/remote-home/lihaoming/haoming/graspd/grasping/data/contactpose/mug/opt39_hand.obj')
    # obj_mesh =  trimesh.load('/remote-home/lihaoming/haoming/graspd/grasping/data/contactpose/mug/mug.obj')
    hand_mesh = trimesh.load('H:\DoctorResesarch\实验总结\DiffTraj\IJRR\data_material\mug_gt/{}.obj'.format(hand_name))
    hand_mesh.visual.face_colors[:, :3]+=np.asarray([15,30,40],dtype=np.uint32)
    hand_mesh.visual.vertex_colors[:, :3]+=np.asarray([15,20,40],dtype=np.uint32)
    hand_mesh.export('./tmp_save/{}_v2.obj'.format(hand_name))

    obj_mesh =  trimesh.load('H:\DoctorResesarch\实验总结\DiffTraj\IJRR\data_material\mug_gt/{}.obj'.format(obj_name))


    hand_verts = torch.tensor(hand_mesh.vertices).unsqueeze(0).float()
    object_verts = torch.tensor(obj_mesh.vertices).unsqueeze(0).float()


    # Contact map 简化计算方式
    # obj_nn_dist,_ = get_NN(object_verts,hand_verts)
    # obj_contact_map = get_pseudo_cmap(obj_nn_dist)

    # Contact map ContactOpt计算方式
    hand_normals =torch.tensor(hand_mesh.vertex_normals).unsqueeze(0).float()
    obj_normals =torch.tensor(obj_mesh.vertex_normals).unsqueeze(0).float()
    obj_contact_map,_ = calculate_contact_capsule(hand_verts, hand_normals, object_verts, obj_normals)

    # mask1 = (obj_contact_map>0.8)*(obj_contact_map<0.95)
    # mask2 = (obj_contact_map>0.7)*(obj_contact_map<0.8)
    # mask3 = (obj_contact_map>0.6)*(obj_contact_map<0.7)
    # obj_contact_map[mask1]+=0.2
    # obj_contact_map[mask2]+=0.15
    # obj_contact_map[mask3]+=0.1
    # obj_contact_map=torch.clip(obj_contact_map,0.,1.)

    obj_contact_mesh = obj_mesh_instance2(object_verts, obj_mesh.faces, obj_contact_map,colors='jet')
    io.write_triangle_mesh('./tmp_save/{}_{}_contact.obj'.format(obj_name,hand_name),obj_contact_mesh)

    shape_grasps_dir = '/home/dataset/haoming/ObMan_unzip/shapenet_grasps/meshnet_04074963_f677657b2cd55f930d48ff455dd1223_scale_275.0.json'
    with open(shape_grasps_dir, encoding="utf-8") as info:
        contents = json.load(info)

    base_name = os.path.basename(shape_grasps_dir)
    key_strs = base_name.split('_')
    obj_file_path = os.path.join('/home/dataset/gw/MESH/MESH_DATASET/ShapeNetCore.v2', key_strs[1],key_strs[2],'models','model_normalized.obj')
    obj_mesh = io.read_triangle_mesh(obj_file_path)
    # obj_mesh.compute_vertex_normals()
    # obj_mesh.compute_triangle_normals()
    pcd = obj_mesh.sample_points_poisson_disk(4096)
    obj_verts = np.asarray(pcd.points,dtype=np.double)




    # obj_vert = np.asarray(obj_mesh.vertices,dtype=np.float32)
    # obj_vert*=0.275
    # obj_face = np.asarray(obj_mesh.triangles)
    # obj_mesh = trimesh.Trimesh(vertices=obj_vert, faces=obj_face)
    # # obj_mesh.fix_normals()
    # obj_sample_vert, face_id = trimesh.sample.sample_surface(obj_mesh, obj_vert.shape[0]*20)

    # obj_sample_vert_normal = np.zeros((obj_vert.shape[0]*30,3))

    # obj_pcd = o3dg.PointCloud()
    # obj_pcd.points = o3du.Vector3dVector(obj_sample_vert)
    # obj_pcd.paint_uniform_color([0.8, 0.1, 0])
    # # io.write_point_cloud('meshnet_04074963_f677657b2cd55f930d48ff455dd1223_scale_275.ply',obj_pcd)
    # obj_pcd.normals =o3du.Vector3dVector(obj_mesh.face_normals[face_id])
    # o3dv.draw_geometries([obj_pcd])
    # a=1
    #
    # distances = obj_pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 2*avg_dist
    #
    # bpa_mesh = o3dg.TriangleMesh.create_from_point_cloud_ball_pivoting(obj_pcd, o3du.DoubleVector(
    #     [radius, radius * 2]))
    #
    # o3dv.draw_geometries([bpa_mesh])
    # a=1

    # tri_mesh = trimesh.Trimesh(obj_verts, obj_faces,
    #     #                            vertex_normals=np.asarray(obj_mesh.vertex_normals))
    # tri_mesh = trimesh.load(obj_file_path,file_type='obj',force='mesh')



    # tri_mesh.show()
    #
    #
    # ob_dataset_dir = '/home/xinzhuo/code/Contact2Mesh/logging/obman_test_0.01.pkl'
    # ob_dataset = pickle.load(open(ob_dataset_dir, 'rb'))
    #
    #
    #
    # obj_verts = ob_dataset[20]['obj_verts']
    # obj_pcd = o3dg.PointCloud()
    #
    #
    # obj_pcd.points = o3du.Vector3dVector(obj_verts)
    # obj_pcd.paint_uniform_color([0.8, 0.1, 0])
    # obj_pcd.estimate_normals()
    # obj_pcd.normalize_normals()
    # o3dv.draw_geometries([obj_pcd])
    #
    # mesh = o3dg.TriangleMesh()
    #
    # distances = obj_pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3*avg_dist
    #
    # bpa_mesh = o3dg.TriangleMesh.create_from_point_cloud_ball_pivoting(obj_pcd, o3du.DoubleVector(
    #     [radius, radius * 2]))
    #
    #
    #
    # o3dv.draw_geometries([bpa_mesh])
    #
    # tri_mesh = trimesh.Trimesh(np.asarray(bpa_mesh.vertices), np.asarray(bpa_mesh.triangles),
    #                            vertex_normals=np.asarray(bpa_mesh.vertex_normals))
    #
    # dec_mesh = mesh.simplify_quadric_decimation(100000)
    #
    # tri_mesh.show()
    # a=1
