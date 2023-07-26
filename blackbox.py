import torch
import numpy as np

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms

# from utils.box_utils import decode, decode_landm
import torchvision

from .demo import RetinaFaceDetector as RetinaFaceDetector_

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [batch size,num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]

    return boxes

def py_gpu_nms(dets, nms_thresh):
    """
    Arguments
        dets (Tensors[N, 5])
    :return
    """
    # dets = torch.tensor(dets).cuda()
    boxes = dets[:, :4]
    scores = dets[:, -1]
    idxs = torch.zeros_like(scores)
    keep = torchvision.ops.batched_nms(boxes, scores, idxs, nms_thresh)
    return keep#.detach().cpu().numpy()


class RetinaFaceDetector(RetinaFaceDetector_):
    def __init__(self, trained_model, network, use_cpu=False, confidence_threshold=0.02, top_k=5000,
                 nms_threshold=0.4, keep_top_k=750, vis_thres=0.6, im_height=720, im_width=1280,
                 fpn_pruned=False, nms_gpu=True):
        super(RetinaFaceDetector, self).__init__(trained_model, network, use_cpu, confidence_threshold, top_k,
                 nms_threshold, keep_top_k, vis_thres, im_height, im_width,
                 fpn_pruned, nms_gpu)

    def batch_detect(self, img):
        _, _, im_height, im_width = img.shape

        if im_height != self.im_height or im_width != self.im_width:
            self.im_height = im_height
            self.im_width = im_width

            self.scale = torch.Tensor([self.im_width, self.im_height, self.im_width, self.im_height])
            self.scale = self.scale.to(self.device)

            self.scale1 = torch.Tensor([
                self.im_width, self.im_height,
                self.im_width, self.im_height,
                self.im_width, self.im_height,
                self.im_width, self.im_height,
                self.im_width, self.im_height
            ])
            self.scale1 = self.scale1.to(self.device)

            self.priorbox = PriorBox(self.cfg, image_size=(self.im_height, self.im_width))
            self.priors = self.priorbox.forward()
            self.priors = self.priors.to(self.device)
            self.prior_data = self.priors.data

        img = img.to(self.device).float()#[[2, 1, 0], :, :]
        img = self.norm(img)
        # img = img.unsqueeze(0)

        loc, conf, landms = self.net(img)

        boxes = decode(loc.data, self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize

        # boxes = boxes.cpu().numpy()
        # scores = conf.data.cpu().numpy()[:, :, 1]
        scores = conf.data[:, :, 1]
        # landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        # landms = landms * self.scale1 / self.resize
        # landms = landms.cpu().numpy()

        # ignore low scores
        # inds = np.where(scores > self.confidence_threshold)#[0]
        # boxes = boxes[inds]
        # landms = landms[inds]
        # scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort(dim = -1, descending = True)[:, :self.top_k]

        boxes = torch.gather(boxes, dim=1, index=order.unsqueeze(2).expand(-1, -1, 4))
        # boxes = boxes[order]
        # landms = landms[order]
        # scores = scores[order]
        scores = torch.gather(scores, dim=1, index=order)

        # boxes = boxes.cpu().numpy()
        # scores = scores.cpu().numpy()


        # do NMS
        # dets = np.hstack((boxes, scores[:, :, np.newaxis])).astype(np.float32, copy=False)
        batch_dets = torch.cat((boxes, scores.unsqueeze(2)), dim = 2).detach().clone()#.cpu().numpy()

        batch_dets_ = []

        for dets in batch_dets:
            if self.nms_gpu:
                # pass
                keep = py_gpu_nms(dets.clone(), self.nms_threshold).clone()
            # else:
            #     keep = py_cpu_nms(dets, self.nms_threshold)
            # # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            # landms = landms[keep]

            # # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :4]
            # landms = landms[:self.keep_top_k, :]

            batch_dets_.append(dets.cpu().numpy().astype(int))

        # dets = np.concatenate((dets, landms), axis=1)
        
        return batch_dets_


def retinaface_resnet50(ckpt):
    detector =  RetinaFaceDetector(trained_model=ckpt, network="resnet50", im_height=250, im_width=250, keep_top_k=1)
    return lambda x: detector.batch_detect(x * 255)
