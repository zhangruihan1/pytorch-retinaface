# coding:utf-8
"""
    Created by cheng star at 2022/2/18 22:06
    @email : xxcheng0708@163.com
"""
import os
import cv2
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from imutils.video import FPS
import torchvision
from torchvision.io import read_image
from utils.timer import print_execute_info


class RetinaFaceDetector(object):
    def __init__(self, trained_model, network, use_cpu=False, confidence_threshold=0.02, top_k=5000,
                 nms_threshold=0.4, keep_top_k=750, vis_thres=0.6, im_height=720, im_width=1280):
        super(RetinaFaceDetector, self).__init__()
        self.trained_model = trained_model
        self.network = network
        self.use_cpu = use_cpu
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.im_height = im_height
        self.im_width = im_width
        self.device = torch.device("cpu" if self.use_cpu else "cuda")
        self.norm = torchvision.transforms.Normalize(mean=[104.0, 117.0, 123.0], std=[1.0, 1.0, 1.0])

        torch.set_grad_enabled(False)
        self.cfg = None
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50

        self.net = RetinaFace(cfg=self.cfg, phase="test")
        self.load_model(self.trained_model, self.use_cpu)
        self.net.eval()
        print(self.net)
        cudnn.benchmark = True

        self.net = self.net.to(self.device)

        self.resize = 1
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

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(self.net, pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)

    @print_execute_info
    def detect(self, img):
        _, im_height, im_width = img.shape

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

        img = img.to(self.device)[[2, 1, 0], :, :].float()
        img = self.norm(img)
        img = img.unsqueeze(0)

        loc, conf, landms = self.net(img)

        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms * self.scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets


if __name__ == '__main__':
    import shutil

    detector = RetinaFaceDetector(trained_model="./weights/Resnet50_Final.pth", network="resnet50",
                                  im_height=720, im_width=1280)

    fps = FPS()
    fps.start()

    data_path = "./images"
    output_path = "./outputs"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for image_name in os.listdir(data_path):
        image_path = os.path.join(data_path, image_name)

        img = read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        results = detector.detect(img)

        fps.update()

        # save results
        if False:
            img_raw = cv2.imread(image_path)

            for b in results:
                if b[4] < detector.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            cv2.imwrite(os.path.join(output_path, image_name), img_raw)

    fps.stop()
    print("duration time: {} s, fps: {}".format(fps.elapsed(), fps.fps()))
