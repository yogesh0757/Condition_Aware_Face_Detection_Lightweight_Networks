import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import FPN as FPN
from models.net import SSH as SSH
from models.net import BBLiteV4
from models.net import MobileNetV1



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2) 

class ClassHead_We(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead_We,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2) 

class ClassHead_Blur(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead_Blur,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2) 

class ClassHead_Occ(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead_Occ,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2) 

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class CAFACLite(nn.Module):
    def __init__(self, cfg_net = None, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(CAFACLite,self).__init__()
        self.phase = phase
        backbone = None
        if cfg_net['name'] == 'BBLiteV4':
            LiteNetwork = BBLiteV4()
            checkpoint = torch.load('/home/pguha/Face_work/Codes_of_Papers_Gitghub/ICPR2026/weights/BBLiteV4.pth.tar', map_location=torch.device('cpu'))
            LiteNetwork.load_state_dict(checkpoint['state_dict'])
            in_channels_list = [
                64,
                128,
                256,
            ]
        elif cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("/home/pguha/Face_work/IEEE_TShort_condition/TPAMI_SHORT/Condition_1_4_1_9/MobileNetV1x0_25_condition_All_Head/weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
            in_channels_list = [
                64,
                128,
                256,
            ]
        elif cfg_net['name'] == 'shufflenet_v2_x0_5':
            import torchvision.models as models
            backbone = models.shufflenet_v2_x0_5(pretrained=cfg['pretrain'])
            in_channels_list = [
                48,
                96,
                1024,
            ]

        self.body = _utils.IntermediateLayerGetter(LiteNetwork, cfg['return_layers'])
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.ClassHead_WE = self._make_class_head_we(fpn_num=3, inchannels=cfg['out_channel'])
        self.ClassHead_BLUR = self._make_blur_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.ClassHead_OCC = self._make_occlusion_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    def _make_class_head_we(self,fpn_num=3,inchannels=64,anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead_We(inchannels,anchor_num))
        return classhead
    
    def _make_blur_head(self,fpn_num=3,inchannels=64,anchor_num=3):
        blurhead = nn.ModuleList()
        for i in range(fpn_num):
            blurhead.append(ClassHead_Blur(inchannels,anchor_num))
        return blurhead
    
    def _make_occlusion_head(self,fpn_num=3,inchannels=64,anchor_num=3):
        occlusionhead = nn.ModuleList()
        for i in range(fpn_num):
            occlusionhead.append(ClassHead_Occ(inchannels,anchor_num))
        return occlusionhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)
        
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        
        classifications_we = torch.cat([self.ClassHead_WE[i](feature) for i, feature in enumerate(features)],dim=1)
        classifications_blur = torch.cat([self.ClassHead_BLUR[i](feature) for i, feature in enumerate(features)],dim=1)
        classifications_occ = torch.cat([self.ClassHead_OCC[i](feature) for i, feature in enumerate(features)],dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, classifications_we, classifications_blur, classifications_occ, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), F.softmax(classifications_we, dim=-1), F.softmax(classifications_blur, dim=-1), F.softmax(classifications_occ, dim=-1), ldm_regressions)
        return output
