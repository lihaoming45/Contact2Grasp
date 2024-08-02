import numpy as np

buffer_dist = 1.03



def ComputerMaxDistance(vertice, center):
    p_num = vertice.shape[0]
    maxDistance = -1.0
    center_exp = center[None, :].repeat(p_num, axis=0)  # (p_num,3)
    dists = np.linalg.norm(vertice - center_exp, axis=1)  # (p_num,)

    maxdist = np.max(dists)
    maxDistance = np.max([maxDistance, maxdist])
    return maxDistance


def ComputeCenter(hand_v, obj_v):
    minP, maxP = ComputeMinMax(hand_v)
    minPObj, maxPObj = ComputeMinMax(obj_v)

    minCat = np.stack((minP, minPObj), axis=0)
    maxCat = np.stack((maxP, maxPObj), axis=0)

    xMin, yMin, zMin = np.min(minCat, axis=0)
    xMax, yMax, zMax = np.max(maxCat, axis=0)

    center = np.array([(xMin + xMax) / 2., (yMin + yMax) / 2., (zMin + zMax) / 2.])
    return center


def ComputeMinMax(vertices):
    xMin_v, yMin_v, zMin_v = np.min(vertices, axis=0)
    xMax_v, yMax_v, zMax_v = np.max(vertices, axis=0)

    minPoint = np.array([xMin_v, yMin_v, zMin_v])
    maxPoint = np.array([xMax_v, yMax_v, zMax_v])

    # xMin_v = np.min(vertices[:,0])
    # yMin_v = np.min(vertices[:,1])
    return minPoint, maxPoint



def NormalizationWithParams(vertice, center, max_d):
    # p_num = vertice.shape[0]
    # center_exp = center[None, :].repeat(p_num, axis=0)
    vertice = vertice - center[None, :]

    # vertice_norm = vertice / max_d
    vertice_norm = vertice

    return vertice_norm

def process_verts(hand_v, obj_v):
    center = ComputeCenter(hand_v, obj_v)
    max_dist_hand = ComputerMaxDistance(hand_v, center)
    max_dist_obj = ComputerMaxDistance(obj_v, center)

    max_dist_before_normalize = np.max([max_dist_hand, max_dist_obj])
    max_dist_before_normalize *= buffer_dist

    hand_v_norm = NormalizationWithParams(hand_v, center, max_dist_before_normalize)
    obj_v_norm = NormalizationWithParams(obj_v, center, max_dist_before_normalize)
    return hand_v_norm, obj_v_norm

