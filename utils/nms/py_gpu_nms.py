import torch
import torchvision


def py_gpu_nms(dets, nms_thresh):
    """
    Arguments
        dets (Tensors[N, 5])
    :return
    """
    dets = torch.tensor(dets).cuda()
    boxes = dets[:, :4]
    scores = dets[:, -1]
    idxs = torch.zeros_like(scores)
    keep = torchvision.ops.batched_nms(boxes, scores, idxs, nms_thresh)
    return keep.detach().cpu().numpy()
