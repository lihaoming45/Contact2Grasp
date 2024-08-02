import numpy as np
import matplotlib.pyplot as plt
from ContactPose.utilities.import_open3d import *
import ContactPose.utilities.misc as mutils
from ContactPose.utilities.dataset import ContactPose

def apply_semantic_colormap_to_mesh(mesh, semantic_idx, sigmoid_a=0.05,
                                    invert=False):
  colors = np.asarray(mesh.vertex_colors)[:, 0]
  colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)

  # apply different colormaps based on finger
  mesh_colors = np.zeros((len(colors), 3))
  cmaps = ['Greys', 'Purples', 'Oranges', 'Greens', 'Blues', 'Reds']
  cmaps = [plt.cm.get_cmap(c) for c in cmaps]
  for semantic_id in np.unique(semantic_idx):
    if (len(cmaps) <= semantic_id):
      print('Not enough colormaps, ignoring semantic id {:d}'.format(
          semantic_id))
      continue
    idx = semantic_idx == semantic_id
    mesh_colors[idx] = cmaps[semantic_id](colors[idx])[:, :3]
  mesh.vertex_colors = o3du.Vector3dVector(mesh_colors)

  return mesh



def apply_semantic_finger_contact(mesh, semantic_idx, sigmoid_a=0.05,
                                    invert=False, use_gt_mask=True):
    """
    :param mesh:
    :param semantic_idx: (N)
    :param sigmoid_a:
    :param invert:
    :param use_gt_mask:
    :return:
    """

    if use_gt_mask:
        colors = np.asarray(mesh.vertex_colors)[:, 0]
        colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)
        contact_mask = (colors >= 0.4) * 1

        semantic_idx += 1
        new_semantic_idx = contact_mask*semantic_idx # 0->no_contact, 1->thumb, 2->index,3->mid, 4->ring, 5->little
    else:
        new_semantic_idx = semantic_idx
    # cyans, Reds, Purples, Oranges, Greens, greys =[0.,1.,1.], [1.,0.,0.], [0.5,0.,0.5], [1.,0.,0.5], [0.,1.,0.], [0.8,0.8,0.8]
    cmaps = [[1.,1.,1.], [1.,0.,0.], [0.5, 0., 0.5], [1.,0.,0.5], [0.,1.,0.], [0.,1.,1.]]

    # apply different colormaps based on finger
    mesh_colors = np.zeros((len(semantic_idx), 3))
    # mesh_colors[contact_mask==1]=1

    for semantic_id in np.unique(new_semantic_idx):
        idx = new_semantic_idx == semantic_id
        mesh_colors[idx] = np.asarray(cmaps[semantic_id])
    mesh.vertex_colors = o3du.Vector3dVector(mesh_colors)

    return mesh



def show_contactmap(p_num, intent, object_name, data_dir,
                    joint_sphere_radius_mm=4.0, bone_cylinder_radius_mm=2.5,
                    bone_color=np.asarray([224.0, 172.0, 105.0])/255,
                    show_axes=False):
    geoms = []

    cp = ContactPose(p_num, intent, object_name,base_dir=data_dir)

    # read contactmap
    mesh = o3dio.read_triangle_mesh(cp.contactmap_filename)
    mesh.compute_vertex_normals()

    # read hands
    line_ids = mutils.get_hand_line_ids()
    joint_locs = cp.hand_joints()

    n_lines_per_hand = len(line_ids)
    n_parts_per_finger = 4
    # find line equations for hand parts
    lines = []
    for hand_joints in joint_locs:
        if hand_joints is None:
            continue

    for line_id in line_ids:
        a = hand_joints[line_id[0]]
        n = hand_joints[line_id[1]] - hand_joints[line_id[0]]
        n /= np.linalg.norm(n)
        lines.append(np.hstack((a, n)))
    lines = np.asarray(lines)

    ops = np.asarray(mesh.vertices) #object vertices
    d_lines = mutils.p_dist_linesegment(ops, lines)
    line_idx = np.argmin(d_lines, axis=1) % n_lines_per_hand
    finger_idx, part_idx = divmod(line_idx, n_parts_per_finger)

    # mesh = apply_semantic_colormap_to_mesh(mesh, finger_idx)
    mesh = apply_semantic_finger_contact(mesh, finger_idx)

    geoms.append(mesh)

    if show_axes:
        geoms.append(o3dg.TriangleMesh.create_coordinate_frame(size=0.2))
    o3dv.draw_geometries(geoms)
    a=1


def create_semantic_contact(mesh, cp, sigmoid_a=0.05, invert=False):

    # read hands
    line_ids = mutils.get_hand_line_ids()
    joint_locs = cp.hand_joints()

    n_lines_per_hand = len(line_ids)
    n_parts_per_finger = 4
    # find line equations for hand parts
    lines = []
    for hand_joints in joint_locs:
        if hand_joints is None:
            continue

    for line_id in line_ids:
        a = hand_joints[line_id[0]]
        n = hand_joints[line_id[1]] - hand_joints[line_id[0]]
        n /= np.linalg.norm(n)
        lines.append(np.hstack((a, n)))
    lines = np.asarray(lines)

    ops = np.asarray(mesh.vertices) #object vertices
    d_lines = mutils.p_dist_linesegment(ops, lines)
    line_idx = np.argmin(d_lines, axis=1) % n_lines_per_hand
    semantic_idx, _ = divmod(line_idx, n_parts_per_finger)

    colors = np.asarray(mesh.vertex_colors)[:, 0]
    colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)
    contact_mask = (colors>=0.4)*1
    semantic_idx+=1
    new_semantic_class = contact_mask*semantic_idx  #0->no_contact, 1->thumb, 2->index,3->mid, 4->ring, 5->little

    return new_semantic_class





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='create semantic fingers contact on object')
    parser.add_argument('--p_num', default=2, type=int, help='Participant number (1-50)')
    parser.add_argument('--intent',default='use', type=str, help='Grasp intent')
    parser.add_argument('--object_name',default='mouse', type=str, help="Name of object")
    parser.add_argument('--data_dir', default='/home/dataset/haoming/ContactPose', type=str, help='ContactPose data load path')

    args = parser.parse_args()

    show_contactmap(args.p_num, args.intent, args.object_name, args.data_dir)

    a=1