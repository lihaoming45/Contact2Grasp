import torch
import numpy as np
from contact2mesh.utils.util import soft_to_binary
from scipy.stats import entropy
from sklearn.cluster import MeanShift, DBSCAN, estimate_bandwidth
from scipy.spatial.distance import dice

def div_sample_extract(sample_set):
    N = sample_set.size(0)
    M_gg_dice, _ = _pairwise_MSE_DICE(sample_set, sample_set, return_error=False)

    M_gg_dice_select = (M_gg_dice<0.9)*1
    dice_idx_select = [torch.nonzero(M_gg_dice_select[m_i]) for m_i in range(N)]

    idx_concat = torch.cat(dice_idx_select)

    unique_idx = idx_concat.unique()
    new_sample_set = sample_set[unique_idx]
    return new_sample_set



def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


# 3D 姿态估计评价指标
def error(preds, gts, config):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    """
    N = preds.shape[0]

    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    pampjpe = np.zeros([N, config["n_joints"]])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]

    _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
    frame_pred = (b * frame_pred.dot(T)) + c
    pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe


def mse(x, y, axis=None):
    return np.square(np.subtract(x, y)).mean(axis)


def dice_distance(x,y,smooth=1e-4):
    dice = (2*x*y+smooth)/(x.sum()+y.sum()+smooth)
    return 1-dice


def dbscan_cluster(X, min_s=5):


    dbscan = DBSCAN(eps=0.6, min_samples=min_s, metric=dice, p=1).fit(np.array(X))
    labels_=dbscan.labels_

    labels_count=[{str(i):np.sum(labels_==i)} for i in np.unique(labels_)]

    return labels_, labels_count



def meanshift_cluster(X, quantile=0.08):
    """
    :param X:(N, 2048)
    :param bwidth:
    :return:
    """
    N, num_f = X.shape

    bwidth = estimate_bandwidth(X, quantile=quantile)
    ms = MeanShift(bwidth, n_jobs=4)
    ms.fit(X)
    label_ = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_cluster = np.unique(label_)

    return label_, cluster_centers, n_cluster


def nn_search(X, band_q=0.2, num_k=4, return_tensor=True):
    """
    :param X: generated set (N, 2048)
    :param _num_k: the num of nearest neighbour
    :return: nearest_samples: (N, num_k, 2048)
    """
    label, cluster_centers, n_cluster = meanshift_cluster(np.array(X), quantile=band_q)

    nearest_dct = {}
    for c_i in range(len(n_cluster)):
        c_num = np.sum(label == c_i)
        c_samples = X[label == c_i, :]

        center_i = cluster_centers[c_i]
        exp_center_i = np.expand_dims(center_i, 0).repeat(len(c_samples), 0)
        mse_erors = mse(exp_center_i, c_samples, axis=1)

        if c_num <= num_k:
            nearest_k_idx = np.argpartition(mse_erors, c_num - 1)[:c_num]
        else:
            nearest_k_idx = np.argpartition(mse_erors, num_k)[:num_k]

        nearset_k = c_samples[nearest_k_idx]
        if return_tensor:
            nearset_k = torch.from_numpy(nearset_k).cuda().float()

        nearest_dct.update({'c_' + str(c_i): nearset_k})

    if return_tensor:
        cluster_centers = torch.from_numpy(cluster_centers).cuda().float()

    return nearest_dct, label, cluster_centers, n_cluster


# Contact map 生成评价指标

def distDICE(X, Y,use_hard=True, return_error=True):
    """
    :param X: sample set X (N, 2048), each sample(n_i) in X is same
    :param Y: sample set Y (N, 2048)
    :return: dice_M: The metric of dice distance
    """
    if use_hard:
        X, Y = soft_to_binary(X), soft_to_binary(Y)
    smooth = 1e-4
    inter = 2 * (X * Y).sum(axis=1)  # (N,)
    uni = X.sum(axis=1) + Y.sum(axis=1)  # (N,)

    if return_error:
        dice_m = 1. - (inter + smooth) / (uni + smooth)
    else:
        dice_m = (inter + smooth) / (uni + smooth)
    return dice_m.view(1, -1)


def distMSE(X, Y):
    """
    :param X: sample set X (N, 2048), each sample(n_i) in X is same
    :param Y: sample set Y (N, 2048)
    :return: mse_M: The metric of dice distance
    """
    mse_M = torch.square(X - Y).mean(dim=1)
    return mse_M.view(1, -1)


def _pairwise_MSE_DICE(sample_set, gt_set, return_error=True):
    N_gen = sample_set.shape[0]
    N_gt = gt_set.shape[0]

    dice_lst = []
    mse_lst = []
    iterator = range(N_gen)
    for s_i in iterator:
        sample_i = sample_set[s_i]

        sample_i_exp = sample_i.view(1, -1).expand(N_gt, -1)
        sample_i_exp = sample_i_exp.contiguous()

        dices = distDICE(sample_i_exp, gt_set, return_error)
        mses = distMSE(sample_i_exp, gt_set)

        dice_lst.append(dices)
        mse_lst.append(mses)

    all_dice = torch.cat(dice_lst, dim=0)  # (N_gen, N_gt)
    all_mse = torch.cat(mse_lst, dim=0)  # (N_gen, N_gt)

    return all_dice, all_mse


def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def KNNAccuracy(sam_set, gt_set, k=1):
    results={}
    M_sg_dice, M_sg_mse = _pairwise_MSE_DICE(sam_set, gt_set)
    M_ss_dice, M_ss_mse = _pairwise_MSE_DICE(sam_set, sam_set)
    M_gg_dice, M_gg_mse = _pairwise_MSE_DICE(gt_set, gt_set)

    one_nn_dice = knn(M_gg_dice, M_sg_dice.transpose(1,0), M_ss_dice, k, sqrt=False)
    one_nn_mse = knn(M_gg_mse, M_sg_mse.transpose(1,0), M_ss_mse, k, sqrt=False)


    results.update({
        "1-NN-DICE-%s" % k: v.cpu().numpy() for k, v in one_nn_dice.items() if 'acc' in k
    })

    results.update({
        "1-NN-MSE-%s" % k: v.cpu().numpy() for k, v in one_nn_mse.items() if 'acc' in k
    })

    return results


def MaxMatch_DICE_MSE(sam_set, gt_set,return_error=False):
    M_sg_dice, M_sg_mse = _pairwise_MSE_DICE(sam_set, gt_set, return_error)

    MMD_eval = M_sg_dice.max(1)[0].mean()
    MM_MSE_eval = M_sg_mse.min(1)[0].mean()

    return MMD_eval.numpy(), MM_MSE_eval.numpy()


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


if __name__ == '__main__':
    gen_set = torch.rand(40, 2048)
    gt_set = torch.rand(40, 2048)*2

    # div_sample_extract(gen_set)

    # ------cluster test------
    result = dbscan_cluster(gen_set)

    # ------calculate eval metrics------
    # mm_dice, mm_mse = MaxMatch_DICE_MSE(gen_set,gt_set)
    # results = KNNAccuracy(gen_set, gt_set)
    a=1