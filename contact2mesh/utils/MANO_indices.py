import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

# from torch.utils.data import DataLoader

# from pytorch3d.vis.plotly_vis import plot_scene
# from pytorch3d.structures import Meshes, Pointclouds

# from contactopt.manopth.manopth.manolayer import ManoLayer
# from contact2mesh.data.cp_loader import ContactDBDataset

h, w, sw = pow(3, 0.5) / 2, 0.5, pow(2, 0.5) / 4
zv = torch.tensor([[0., 0., 1.]])  # (1,3)
dec_zv = torch.tensor([[-w, 0, h], [w, 0, h], [0, -w, h], [0, w, h], [sw, sw, h], [-sw, -sw, h], [-sw, sw, h],
                       [sw, -sw, h]])  # (8,3)

# FINGER LIMIT ANGLE FOR RIGHT HAND:
limit_bigfinger_right = torch.FloatTensor([1.2, -0.4, 0.25])  # 36:39
limit_index_right = torch.FloatTensor([-0.0827, -0.4389, 1.5193])  # 0:3
limit_middlefinger_right = torch.FloatTensor([-2.9802e-08, -7.4506e-09, 1.4932e+00])  # 9:12
limit_fourth_right = torch.FloatTensor([0.1505, 0.3769, 1.5090])  # 27:30
limit_small_right = torch.FloatTensor([-0.6235, 0.0275, 1.0519])  # 18:21

limit_secondjoint_bigfinger_right = torch.FloatTensor([0.0, -1.0, 0.0])
limit_secondjoint_index_right = torch.FloatTensor([0.0, 0.0, 1.2])
limit_secondjoint_middlefinger_right = torch.FloatTensor([0.0, 0.4, 1.2])
limit_secondjoint_fourth_right = torch.FloatTensor([0.0, 1.0, 1.0])
limit_secondjoint_small_right = torch.FloatTensor([0.0, 0.0, 1.2])

limit_thirdjoint_bigfinger_right = torch.FloatTensor([0.0, -1.0, 0.0])
limit_thirdjoint_index_right = torch.FloatTensor([0.0, 0.0, 1.2])
limit_thirdjoint_middlefinger_right = torch.FloatTensor([0.0, 0.4, 1.2])
limit_thirdjoint_fourth_right = torch.FloatTensor([0.0, 1.0, 1.0])
limit_thirdjoint_small_right = torch.FloatTensor([0.0, 0.0, 1.2])

# FINGER LIMIT ANGLE FOR LEFT HAND:
limit_bigfinger_left = torch.FloatTensor([1.2, -0.4, 0.25])  # 36:39
limit_index_left = torch.FloatTensor([0.0827, 0.4389, -1.5193])  # 0:3
limit_middlefinger_left = torch.FloatTensor([2.9802e-08, 7.4506e-09, -1.4932e+00])  # 9:12
limit_fourth_left = torch.FloatTensor([-0.1505, -0.3769, -1.5090])  # 27:30
limit_small_left = torch.FloatTensor([-0.6235, 0.1, -1.0519])  # 18:21

limit_secondjoint_bigfinger_left = torch.FloatTensor([0.0, 0.8, -0.8])
limit_secondjoint_index_left = torch.FloatTensor([0.0, 0.0, -1.2])
limit_secondjoint_middlefinger_left = torch.FloatTensor([0.0, -0.4, -1.2])
limit_secondjoint_fourth_left = torch.FloatTensor([0.0, -1.0, -1.0])
limit_secondjoint_small_left = torch.FloatTensor([0.0, 0.0, -1.0])

limit_thirdjoint_bigfinger_left = torch.FloatTensor([0.0, 0.8, -0.8])
limit_thirdjoint_index_left = torch.FloatTensor([0.0, 0.0, -1.2])
limit_thirdjoint_middlefinger_left = torch.FloatTensor([0.0, -0.4, -1.2])
limit_thirdjoint_fourth_left = torch.FloatTensor([0.0, -1.0, -1.0])
limit_thirdjoint_small_left = torch.FloatTensor([0.0, 0.0, -1.0])

# if torch.cuda.is_available():
#     limit_bigfinger_right = limit_bigfinger_right.cuda()
#     limit_index_right = limit_index_right.cuda()
#     limit_middlefinger_right = limit_middlefinger_right.cuda()
#     limit_fourth_right = limit_fourth_right.cuda()
#     limit_small_right = limit_small_right.cuda()
#
#     limit_secondjoint_bigfinger_right = limit_secondjoint_bigfinger_right.cuda()
#     limit_secondjoint_index_right = limit_secondjoint_index_right.cuda()
#     limit_secondjoint_middlefinger_right = limit_secondjoint_middlefinger_right.cuda()
#     limit_secondjoint_fourth_right = limit_secondjoint_fourth_right.cuda()
#     limit_secondjoint_small_right = limit_secondjoint_small_right.cuda()
#
#     limit_thirdjoint_bigfinger_right = limit_thirdjoint_bigfinger_right.cuda()
#     limit_thirdjoint_index_right = limit_thirdjoint_index_right.cuda()
#     limit_thirdjoint_middlefinger_right = limit_thirdjoint_middlefinger_right.cuda()
#     limit_thirdjoint_fourth_right = limit_thirdjoint_fourth_right.cuda()
#     limit_thirdjoint_small_right = limit_thirdjoint_small_right.cuda()
#
#     limit_bigfinger_left = limit_bigfinger_left.cuda()
#     limit_index_left = limit_index_left.cuda()
#     limit_middlefinger_left = limit_middlefinger_left.cuda()
#     limit_fourth_left = limit_fourth_left.cuda()
#     limit_small_left = limit_small_left.cuda()
#
#     limit_secondjoint_bigfinger_left = limit_secondjoint_bigfinger_left.cuda()
#     limit_secondjoint_index_left = limit_secondjoint_index_left.cuda()
#     limit_secondjoint_middlefinger_left = limit_secondjoint_middlefinger_left.cuda()
#     limit_secondjoint_fourth_left = limit_secondjoint_fourth_left.cuda()
#     limit_secondjoint_small_left = limit_secondjoint_small_left.cuda()
#
#     limit_thirdjoint_bigfinger_left = limit_thirdjoint_bigfinger_left.cuda()
#     limit_thirdjoint_index_left = limit_thirdjoint_index_left.cuda()
#     limit_thirdjoint_middlefinger_left = limit_thirdjoint_middlefinger_left.cuda()
#     limit_thirdjoint_fourth_left = limit_thirdjoint_fourth_left.cuda()
#     limit_thirdjoint_small_left = limit_thirdjoint_small_left.cuda()
f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
      750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325,
      326,
      327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353,
      354,
      355]
f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
      439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
      550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
      668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
f0 = [73, 96, 98, 99, 772, 774, 775, 777]

f1_tip = [738, 739, 740, 743, 745, 746, 748, 749, 756, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
f1_se = [757, 755, 741, 754, 753, 704, 700, 699]

f2_tip = [317, 323, 324, 325, 326, 327, 328, 329, 346, 343, 347, 348, 349, 350, 351, 352, 353, 354, 355]
f2_se = [344, 342, 330, 301, 341, 340, 281, 238, 237]
f2_th = [280, 47, 46, 166, 48, 46, 165, 194, 195]

f3_tip = [429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 459, 460, 461, 462, 463, 464, 465, 466, 467]
f3_se = [456, 454, 440, 413, 453, 430, 403, 397, 396]
f3_th = [402, 357, 356, 376, 358, 359, 375, 386, 387]

f4_tip = [545, 546, 547, 548, 549, 550, 553, 555, 566, 570, 571, 572, 573, 574, 575, 576, 577, 578]
f4_se = [503, 523, 568, 567, 565, 551, 524, 564, 563, 514, 507, 506]
f4_th = [484, 487, 502, 513, 469, 468, 486, 470, 471, 485, 496, 497]

f5_tip = [663, 664, 665, 666, 667, 670, 672, 683, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
f5_se = [631, 641, 684, 625, 681, 682, 668, 680, 624]
f5_th = [601, 602, 630, 581, 582, 614, 580, 583, 615]

prior_idx = f1_tip + f1_se + f2_tip + f2_se + f2_th + f3_tip + f3_se + f3_th + f4_tip + f4_se + \
            f4_th + f5_tip + f5_se + f5_th

mano_fingers_front_idxs = {'small': f5_tip, 'middle': f3_tip,
                           'index': f2_tip + f2_se + f2_th, 'fourth': f4_tip,
                           'big': f1_tip}

bigfinger_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715,
                      716, 717, 718, 719, 720, 721, 722, 723, 724,
                      725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
                      738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
                      751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
                      764, 765, 766, 767, 768]

indexfinger_vertices = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 155, 156, 164, 165, 166, 167, 174, 175, 189,
                        194, 195, 212, 213, 221, 222, 223, 224, 225, 226, 237, 238,245, 272, 273, 280, 281, 282, 283, 294,
                        295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
                        314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
                        333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
                        352, 353, 354, 355]

middlefinger_vertices = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367,
                         372, 373, 374, 375, 376, 377, 381,
                         382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
                         395, 396, 397, 398, 400, 401, 402, 403, 404, 405, 406, 407,
                         408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
                         421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
                         434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
                         447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                         460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_vertices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482, 483, 484, 485, 486, 487, 491,
                         492,
                         495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
                         508, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
                         521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
                         534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
                         547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                         560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
                         573, 574, 575, 576, 577, 578]

smallfinger_vertices = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
                        598, 599, 600, 601, 602, 603,
                        609, 610, 613, 614, 615, 616, 617, 618, 619, 620,
                        621, 622, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633,
                        634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
                        647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                        660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
                        673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
                        686, 687, 688, 689, 690, 691, 692, 693, 694, 695]

indexfinger_secondjoint_vertices = [221, 224, 237, 238, 272, 273, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300, 301,
                                    302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
                                    319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
                                    336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352,
                                    353, 354, 355]

fi_s = indexfinger_secondjoint_vertices

indexfinger_thirdjoint_vertices = [304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 317, 318, 319, 320, 322, 323,
                                   324, 325, 326, 327, 328, 329, 332, 333, 334, 335, 336, 337, 338, 339, 343, 346, 347,
                                   348, 349, 350, 351, 352, 353, 354, 355]

middlefinger_secondjoint_vertices = [390, 393,
                                     396, 397, 398, 400, 401, 403, 404, 405, 406, 407,
                                     408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
                                     421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
                                     434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
                                     447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                                     460, 461, 462, 463, 464, 465, 466, 467]
middlefinger_thirdjoint_vertices = [416, 417, 418, 419,
                                    421, 423, 424, 425, 426, 428, 429, 432, 433,
                                    434, 435, 436, 437, 438, 439, 442, 443, 444, 445, 446,
                                    447, 448, 449, 450, 451, 455, 458, 459,
                                    460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_secondjoint_vertices = [500, 503, 506, 507,
                                     508, 511, 512, 514, 515, 516, 517, 518, 519, 520,
                                     521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
                                     534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
                                     547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                                     560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
                                     573, 574, 575, 576, 577, 578]
fourthfinger_thirdjoint_vertices = [527, 528, 529, 530, 532, 533,
                                    534, 535, 536, 537, 538, 539, 540, 543, 544, 545, 546,
                                    547, 548, 549, 550, 553, 554, 555, 556, 557, 558, 559,
                                    560, 561, 562, 566, 569, 570, 571, 572,
                                    573, 574, 575, 576, 577, 578]

smallfinger_secondjoint_vertices = [618,
                                    621, 624, 625, 626, 628, 629, 631, 632, 633,
                                    634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
                                    647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                                    660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
                                    673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
                                    686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
smallfinger_thirdjoint_vertices = [644, 645, 646,
                                   651, 652, 653, 654, 656, 657,
                                   660, 661, 662, 663, 664, 665, 666, 667, 670, 671, 672,
                                   673, 674, 675, 676, 677, 678, 679, 683,
                                   686, 687, 688, 689, 690, 691, 692, 693, 694, 695]

bigfinger_secondjoint_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713,
                                  714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
                                  725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
                                  738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
                                  751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
                                  764, 765, 766, 767, 768]
bigfinger_thirdjoint_vertices = [745, 744, 766, 729, 735, 751, 765, 730, 752, 764, 738, 728, 768,
                                 727, 767, 743, 747, 720, 748, 717, 750, 734, 761, 737, 724, 762,
                                 763, 726, 740, 719, 746, 718, 725, 722, 723, 733, 749, 716, 731,
                                 721, 736, 759, 739, 760, 756]

mano_fingers_idxs = {'small': smallfinger_vertices, 'middle': middlefinger_vertices,
                     'index': indexfinger_vertices, 'fourth': fourthfinger_vertices,
                     'big': bigfinger_vertices}

# Initialize MANO layer
# MANO = ManoLayer(
#     mano_root='/home/enric/libraries/manopth/mano/models/', side='right', use_pca=True, ncomps=45, flat_hand_mean=True)
# #if torch.cuda.device_count() > 1:
#     #print("Let's use", torch.cuda.device_count(), "GPUs!")
#     #MANO = torch.nn.DataParallel(MANO)
# MANO = MANO.cuda()
def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    import trimesh
    # import contact2mesh.arguments as arguments
    # import contact2mesh.utils.util as util
    from open3d import utility as o3du
    from open3d import geometry as o3dg
    from open3d import visualization as o3dv
    # from contact2mesh.utils.util import to_tensor
    from visual import pcd_instance
    from contactopt.manopth.manopth.manolayer import ManoLayer
    from contact2mesh.utils.mano_util import forward_mano2

    # from pytorch3d.ops import box3d_overlap
    # box1 = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],]).float().unsqueeze(0)
    # box2 = box1+0.1
    #
    # intersect = box3d_overlap(box1,box2)
    # a=1

    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # from mesh_to_sdf import sample_sdf_near_surface

    # args = arguments.run_visual_parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # rhm_model = ManoLayer(mano_root='/home/haoming/GrabNet/contactopt/manopth/mano/models', use_pca=False,
    #                       ncomps=45, side='right', flat_hand_mean=True).cuda()
    # 'D:\PycharmProjects\Contact2Mesh\contactopt\manopth\mano\models'
    #'/remote-home/lihaoming/haoming/Contact2Mesh/contactopt/manopth/mano/models'
    rhm_model = ManoLayer(mano_root='D:\PycharmProjects\Contact2Mesh\contactopt\manopth\mano\models',
                          use_pca=False,
                          ncomps=45, side='right', flat_hand_mean=True)
    faces = rhm_model.th_faces.int().cpu().numpy()

    verts, joints = forward_mano2(rhm_model, torch.zeros(1, 48), torch.zeros(1), torch.zeros(1))

    verts = verts.detach().cpu().squeeze().numpy()

    hmesh = trimesh.Trimesh(verts, faces)
    # points, sdf = sample_sdf_near_surface(hmesh, number_of_points=1000)
    tips = [745, 317, 444, 556, 673]

    f1_tip = [738, 739, 740, 743, 745, 746, 748, 749, 756, 759, 760, 761, 762, 763,
              764, 765, 766, 767, 768]  # big
    f1_se = [757, 755, 741, 754, 753, 704, 700, 699]

    f2_tip = [317, 323, 324, 325, 326, 327, 328, 329, 346, 343, 347, 348, 349, 350, 351, 352, 353, 354, 355]  # index
    f2_se = [344, 342, 330, 301, 341, 340, 281, 238, 237]
    f2_th = [280, 47, 46, 166, 48, 46, 165, 194, 195]

    f3_tip = [429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 459, 460, 461,
              462, 463, 464, 465, 466, 467]  # middle
    f3_se = [456, 454, 440, 413, 453, 430, 403, 397, 396]
    f3_th = [402, 357, 356, 376, 358, 359, 375, 386, 387]

    f4_tip = [545, 546, 547, 548, 549, 550, 553, 555, 566, 570, 571, 572,
              573, 574, 575, 576, 577, 578]  # forth
    f4_se = [503, 523, 568, 567, 565, 551, 524, 564, 563, 514, 507, 506]
    f4_th = [484, 487, 502, 513, 469, 468, 486, 470, 471, 485, 496, 497]

    f5_tip = [663, 664, 665, 666, 667, 670, 672, 683, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]  # small
    f5_se = [631, 641, 684, 625, 681, 682, 668, 680, 624]
    f5_th = [601, 602, 630, 581, 582, 614, 580, 583, 615]

    big_tip_box_idxs = [756, 716, 721]  # [756, 760, 739]
    index_tip_box_idxs = [343, 336, 325]
    mid_tip_box_idxs = [455, 424, 432]

    ft_tip_box_idxs = [566, 559, 546]
    sm_tip_box_ixs = [683, 676, 645]

    ax_idx = verts[sm_tip_box_ixs, :]
    ax1 = ax_idx[0] - ax_idx[1]
    ax2 = ax_idx[0] - ax_idx[2]
    axz = np.cross(ax1, ax2)
    axz = axz / np.linalg.norm(axz, keepdims=True)

    axx = ax_idx[2] - ax_idx[1]
    axx = axx / np.linalg.norm(axx, keepdims=True)

    axy = np.cross(axx, axz)
    a = 1
    center = np.mean(verts[smallfinger_thirdjoint_vertices], axis=0)

    w = 0.0055

    p0 = center + w * (-axx + axy - axz)
    p1 = center + w * (+axx + axy - axz)
    p2 = center + w * (+axx - axy - axz)
    p3 = center + w * (-axx - axy - axz)
    p4 = center + w * (-axx + axy + axz)
    p5 = center + w * (+axx + axy + axz)
    p6 = center + w * (+axx - axy + axz)
    p7 = center + w * (-axx - axy + axz)

    box = np.stack([p0, p1, p2, p3, p4, p5, p6, p7])

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3dg.LineSet()
    line_set.points = o3du.Vector3dVector(box)
    line_set.lines = o3du.Vector2iVector(lines)
    line_set.colors = o3du.Vector3dVector(colors)

    contacts = torch.zeros(778, 1)

    finger_ids = to_tensor(np.concatenate(
        [bigfinger_vertices, indexfinger_vertices, middlefinger_vertices, fourthfinger_vertices,
         smallfinger_vertices])).long()

    palm_ids = torch.tensor(list(range(778)))
    palm_ids = torch.cat([palm_ids, finger_ids])
    uniset, count = palm_ids.unique(return_counts=True)
    palm_ids = uniset.masked_select(mask=(count == 1))

    idx = palm_ids

    idx = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype=np.int32)

    contacts[idx] = 0.8

    part_pcd = pcd_instance(verts, contact=contacts)
    o3dv.draw_geometries([part_pcd])

    pcd_idxs = np.zeros(778, dtype=np.long)
    for i in range(pcd_idxs.__len__()):
        # pcd_idxs[i] = i
        pcd_idxs[i] = i if i in idx else 0
    pcd_idxs = list(pcd_idxs)

    app = o3dv.gui.Application.instance
    app.initialize()
    vis = o3dv.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", part_pcd)

    for idx in pcd_idxs:
        vis.add_3d_label(part_pcd.points[idx], "{}".format(idx))
    vis.reset_camera_to_default()

    # vis.add_window(vis)
    app.add_window(vis)
    app.run()
    a = 1

    o3dv.draw_geometries([part_pcd, line_set])
    a = 1
