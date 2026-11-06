import torch
from torch import nn
import numpy as np

# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

# def softmax_weights(dist, mask, eps: float = 1e-12):
#     logits = dist.masked_fill(mask == 0, float('-inf'))
#     max_v = torch.amax(logits, dim=1, keepdim=True)
#     logits = logits - max_v
#     w = torch.exp(logits)
#     Z = w.sum(dim=1, keepdim=True) + eps
#     W = w / Z
#     return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'.

    Optional camera-aware variant: positives are unchanged, while
    same-camera negatives receive a configurable (lower) weight.  This mirrors
    the evaluator, which removes every gallery image from the query camera.
    """

    def __init__(self, use_camera=False, cam_neg_weight=1.0):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.use_camera = use_camera
        self.cam_neg_weight = cam_neg_weight

    def forward(self, inputs, targets, normalize_feature=False,
                camids=None, norm_beta=0.0):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        if norm_beta > 0:
            # Keep the original feature scale while mixing in its direction.
            unit = normalize(inputs, axis=-1)
            scale = inputs.norm(dim=1, keepdim=True).mean().detach()
            inputs = (1 - norm_beta) * inputs + norm_beta * (unit * scale)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        pos_mask = is_pos
        neg_mask = is_neg
        if self.use_camera and camids is not None:
            same_cam = camids.reshape(-1, 1).eq(camids.reshape(1, -1)).float()
            neg_mask = is_neg * (1.0 - same_cam) + is_neg * same_cam * self.cam_neg_weight
            # A row whose negatives are all same-camera would otherwise have
            # an all-zero negative mask; fall back to the original mask.
            empty = neg_mask.sum(dim=1) <= 0
            if empty.any():
                neg_mask[empty] = is_neg[empty]

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * pos_mask
        dist_an = dist_mat * neg_mask

        weights_ap = softmax_weights(dist_ap, pos_mask)
        weights_an = softmax_weights(-dist_an, neg_mask)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    emb1 = emb1.float()
    emb2 = emb2.float()
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx
