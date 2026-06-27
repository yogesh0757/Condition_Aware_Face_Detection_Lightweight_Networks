from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_CAFACLite, cfg_BV4, cfg_MV1, cfg_SV2
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.cafaclite import CAFACLite
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from ptflops import get_model_complexity_info as cp


parser = argparse.ArgumentParser(description='LWFD')
parser.add_argument('-m', '--trained_model', default='./weights/CAFACLite_BV4.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='BBLiteV4', help='Backbone network BBLiteV4, mobilenet0.25 or shufflenet_v2_x0_5')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--confidence_threshold_weight', default=0.05, type=float, help='confidence_threshold_weight')
parser.add_argument('--confidence_threshold_blur', default=0.05, type=float, help='confidence_threshold_blur')
parser.add_argument('--confidence_threshold_occlusion', default=0.09, type=float, help='confidence_threshold_occlusion')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
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


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg_net = None
    cfg = cfg_CAFACLite
    if args.network == 'BBLiteV4':
        cfg_net = cfg_BV4
    elif args.network == 'mobilenet0.25':
        cfg_net = cfg_MV1
    elif args.network == "shufflenet_v2_x0_5":
        cfg_net = cfg_SV2
    # net and model
    net = CAFACLite(cfg_net=cfg_net, cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    condition_weight_apply = cfg['condition_we_apply']
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img1 = np.float32(img_raw)
        size = [500, 800, 1100, 1400, 1700]
        # testing scale
        _t['misc'].tic()
        for j in range(len(size)):
            target_size = 800
            max_size = 1200
            img = img1.copy()
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            
            resize = float(size[j])/target_size*im_scale
            if args.origin_size:
                resize = 1
            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            #_t['forward_pass'].tic()
            if condition_weight_apply == True:
                loc, conf, conf_we, conf_blur, conf_occ, landms = net(img)  # forward pass
            else:
                loc, conf, landms = net(img)  # forward pass
            #_t['forward_pass'].toc()
            #_t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores1 = conf.squeeze(0).data.cpu().numpy()[:, 1]
            if condition_weight_apply == True:
                scores2 = conf_we.squeeze(0).data.cpu().numpy()[:, 1]
                scores3 = conf_blur.squeeze(0).data.cpu().numpy()[:, 1]
                scores4 = conf_occ.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()
            
            

            if condition_weight_apply == True:
                inds1 = np.where(scores1 > args.confidence_threshold)[0]
                boxes1 = boxes[inds1]
                landms1 = landms[inds1]
                scores1 = scores1[inds1]

                inds2 = np.where(scores2 > args.confidence_threshold_weight)[0]
                boxes2 = boxes[inds2]
                landms2 = landms[inds2]
                scores2 = scores2[inds2]

                inds3 = np.where(scores3 > args.confidence_threshold_blur)[0]
                boxes3 = boxes[inds3]
                landms3 = landms[inds3]
                scores3 = scores3[inds3]

                inds4 = np.where(scores4 > args.confidence_threshold_occlusion)[0]
                boxes4 = boxes[inds4]
                landms4 = landms[inds4]
                scores4 = scores4[inds4]

                scores =  np.concatenate((scores1, scores2, scores3, scores4), axis=0)
                boxes =  np.concatenate((boxes1, boxes2, boxes3, boxes4), axis=0)
                landms =  np.concatenate((landms1, landms2, landms3, landms4), axis=0)

            else:
                inds = np.where(scores1 > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores1[inds]


            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            #_t['misc'].toc()
            if j == 0:
                dets2 = dets.copy()
            else:
                dets2 = np.concatenate([dets, dets2], axis=0)
        keep = py_cpu_nms(dets2[:, :5], args.nms_threshold)
        dets = dets2[keep, :]
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} misc: {:.4f}s'.format(i + 1, num_images, _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

