import torch


def get_s_t_topk(P, k, s_only=False,nn_idx=None):
    """
    Get nearest neighbors per point (similarity value and index) for source and target shapes

    Args:
        P (BxNsxNb Tensor): Similarity matrix
        k: number of neighbors per point
    """
    if(nn_idx is not None):
        assert s_only, "Only for self-construction currently"
        s_nn_idx = nn_idx
        s_nn_val = P.gather(dim=2,index=nn_idx)
        t_nn_val = t_nn_idx = None
    else:
        s_nn_val, s_nn_idx = P.topk(k=min(k,P.shape[2]), dim=2)

        if not s_only:
            t_nn_val, t_nn_idx = P.topk(k=k, dim=1)

            t_nn_val = t_nn_val.transpose(2, 1)
            t_nn_idx = t_nn_idx.transpose(2, 1)
        else:
            t_nn_val = None
            t_nn_idx = None

    return s_nn_val, s_nn_idx, t_nn_val, t_nn_idx


def get_s_t_neighbors(k, P, sim_normalization, s_only=False, ignore_first=False,nn_idx=None):
    from utils import switch_functions
    s_nn_sim, s_nn_idx, t_nn_sim, t_nn_idx = get_s_t_topk(P, k, s_only=s_only,nn_idx=nn_idx)
    if ignore_first:
        s_nn_sim, s_nn_idx = s_nn_sim[:, :, 1:], s_nn_idx[:, :, 1:]

    s_nn_weight = switch_functions.normalize_P(s_nn_sim, sim_normalization, dim=2)

    if not s_only:
        if ignore_first:
            t_nn_sim, t_nn_idx = t_nn_sim[:, :, 1:], t_nn_idx[:, :, 1:]

        t_nn_weight = switch_functions.normalize_P(t_nn_sim, sim_normalization, dim=2)
    else:
        t_nn_weight = None

    return s_nn_weight, s_nn_sim, s_nn_idx, t_nn_weight, t_nn_sim, t_nn_idx


def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1,0))     
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist