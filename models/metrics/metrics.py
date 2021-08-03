from pytorch_lightning.metrics import Accuracy
import torch
import torch.nn.functional as F

class AccuracyAssumeEye(Accuracy):
    def __init__(self):
        super().__init__()

    def update(self, P: torch.Tensor, dim=1):
        preds = P.argmax(dim)

        dim_labels = dim - 1 if dim == len(P.shape) - 1 else dim + 1
        labels = torch.arange(P.shape[dim_labels]).repeat(preds.shape[0], 1).to(preds.device)
        super().update(preds, labels)


class AccuracyAssumeEyeSoft(Accuracy):
    def __init__(self, top_k):
        super().__init__(top_k=top_k)

    def update(self, P: torch.Tensor, dim=1, normalize=True):
        if normalize:
            P_normalized = F.softmax(P, dim=dim)
        else:
            P_normalized = P

        if dim != 1:  # the accuracy metric assumes that the classes dim (where the probabilities sum to 1) is dim = 1
            soft_preds = P_normalized.transpose(dim, 1)
        else:
            soft_preds = P_normalized

        dim_labels = dim - 1 if dim == len(P.shape) - 1 else dim + 1
        labels = torch.arange(P.shape[dim_labels]).repeat(soft_preds.shape[0], 1).to(soft_preds.device)
        super().update(soft_preds, labels)


def accuracy_hit_neighbors_soft(preds, neighbor_idxs, k, accuracy_dtype=torch.float32):
    if len(preds.shape) < len(neighbor_idxs.shape):
        preds = torch.unsqueeze(preds, dim=-1)

    assert k <= neighbor_idxs.shape[2], "The number of neighbors %d for AccuracyHitNeghiborsSoft should be smaller or equal to %d" % (k, neighbor_idxs.shape[2])
    neighbor_idxs = neighbor_idxs[:, :, :k]

    soft_hit = (preds == neighbor_idxs)
    soft_hit_accuracy = soft_hit.any(dim=2).to(accuracy_dtype)

    return soft_hit_accuracy.mean()


def accuracy_euclidean_soft(error_per_point, d_max, tolerance):
    error_per_point_normed = torch.sqrt(error_per_point) / d_max
    soft_hit_accuracy = (error_per_point_normed <= tolerance).to(error_per_point.dtype)

    return soft_hit_accuracy.mean()


def accuracy_cycle_assume_eye(P: torch.Tensor):
    source_preds = P.argmax(2)
    target_preds = P.argmax(1)

    source_labels = torch.arange(P.shape[2]).repeat(P.shape[0], 1).to(P.device)
    target_labels = torch.arange(P.shape[1]).repeat(P.shape[0], 1).to(P.device)

    accuracy_cycle = torch.logical_and(source_preds == source_labels, target_preds == target_labels).to(P.dtype)
    return accuracy_cycle.mean()


def uniqueness(preds, dtype=torch.float32):
    unique_preds = [torch.unique(preds[i]) for i in range(preds.shape[0])]
    unique_count = torch.tensor([len(unique_pred) for unique_pred in unique_preds], dtype=dtype) / preds.shape[1]

    return unique_count.mean()
