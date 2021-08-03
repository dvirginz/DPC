import torch

def chamfer_distance_with_batch(p1, p2, debug=False):
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)
    
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    dist = torch.min(dist, dim=2)[0]
    dist = torch.sum(dist)

    return dist