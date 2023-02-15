from random import sample
import torch
import torch.nn.functional as F


def class_calibration(pred_mat):
    """
    args
        pred_mat: 
            [n, c] n is num of test samples, c is num of classes
    return:
        pred_class:
            [n] class ids for each sample with all classes have same number
    """
    N, C = pred_mat.size()
    # assert N % C == 0
    num_each = N // C
    num_each_last = N - N // C * (C - 1)
    pred_class = torch.zeros(N) - 1
    mask = torch.ones(N, C).to(pred_mat.device)
    for clsid in range(C):
        if clsid == C - 1:
            _, sample_idx = torch.topk(pred_mat[:, clsid], num_each_last)
        else:
            _, sample_idx = torch.topk(pred_mat[:, clsid], num_each)
        print(sample_idx)
        pred_class[sample_idx] = clsid
        mask[sample_idx, :] = 0
        pred_mat = pred_mat * mask
    # pred_class[pred_class == -1] = 0
    return pred_class

def class_calibration_v2(pred_mat):
    """
    args
        pred_mat: 
            [n, c] n is num of test samples, c is num of classes
    return:
        pred_class:
            [n] class ids for each sample with all classes have same number
    """
    # pred_mat = F.softmax(pred_mat, dim=1)
    N, C = pred_mat.size()
    # assert N % C == 0
    num_each = N // C
    # num_each_last = N - N // C * (C - 1)
    pred_class = torch.zeros(N) - 1
    mask = torch.ones(N, C).to(pred_mat.device)

    # first four majority classes, arrange each N//C
    for clsid in range(4):
        _, sample_idx = torch.topk(pred_mat[:, clsid], num_each)
        print(sample_idx)
        pred_class[sample_idx] = clsid
        mask[sample_idx, :] = 0
        pred_mat = pred_mat * mask
    # long-tailed classes, direct argmax (without majority)
    minor_samples = torch.where(pred_mat.sum(dim=1) > 0)[0]
    for idx in minor_samples:
        pred_class[idx] = torch.argmax(pred_mat[idx, 4:]) + 4

    return pred_class


if __name__ == "__main__":
    pseudo_prob = torch.rand(150, 10)
    pred_class = class_calibration(pseudo_prob)
    print(pred_class)
    print(torch.unique(pred_class, return_counts=True))