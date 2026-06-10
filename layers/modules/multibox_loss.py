import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, match_blur, match_occ
from data import cfg_CAFACLite
GPU = cfg_CAFACLite['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, condition_weight, condition_apply):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.condition_weight = condition_weight
        self.condition_apply = condition_apply

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        if self.condition_apply == True:
            loc_data, conf_data, conf_data_we, conf_data_blur, conf_data_occ, landm_data = predictions
        else:
            loc_data, conf_data, landm_data = predictions
        priors = priors
        #mapbox = mapBox
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        #num_boxs = (mapbox.size(0)blur_data)
        
        #conf_data =  conf_data - scale_margin

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        neg_conf_t = torch.LongTensor(num, num_priors)
        blur_t = torch.Tensor(num, num_priors)
        occl_t = torch.Tensor(num, num_priors)
        if self.condition_apply == True:
            blur_conf_t = torch.LongTensor(num, num_priors)
            blur_neg_conf_t = torch.LongTensor(num, num_priors)
            occ_conf_t = torch.LongTensor(num, num_priors)
            occ_neg_conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            blur = targets[idx][:, 14].data
            occl = targets[idx][:, 18].data
            defaults = priors.data
            if self.condition_apply == True:
                ind_blur = torch.where(blur > 0)[0]
                truths_blur = truths[ind_blur]
                labels_blur = labels[ind_blur]
                ind_occ = torch.where(occl > 0)[0]
                truths_occ = truths[ind_occ]
                labels_occ = labels[ind_occ]
                match_blur(self.threshold, truths_blur, defaults, self.variance, labels_blur, blur_conf_t, blur_neg_conf_t, idx)
                match_occ(self.threshold, truths_occ, defaults, self.variance, labels_occ, occ_conf_t, occ_neg_conf_t, idx)
                match(self.threshold, truths, blur, occl, defaults, self.variance, labels, landms, loc_t, conf_t, neg_conf_t, landm_t, blur_t, occl_t, idx)
            else:
                match(self.threshold, truths, blur, occl, defaults, self.variance, labels, landms, loc_t, conf_t, neg_conf_t, landm_t, blur_t, occl_t, idx)
        
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            neg_conf_t = neg_conf_t.cuda()
            if self.condition_apply == True:
                blur_conf_t = blur_conf_t.cuda()
                occ_conf_t = occ_conf_t.cuda()
                blur_neg_conf_t = blur_neg_conf_t.cuda()
                occ_neg_conf_t = occ_neg_conf_t.cuda()
                blur_t = blur_t.cuda()
                occl_t = occl_t.cuda() 
        
        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos_land = conf_t > zeros
        num_pos_landm = pos_land.long().sum(1, keepdim=True)
        NL = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos_land.unsqueeze(pos_land.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos1 = conf_t != zeros
        conf_t[pos1] = 1

        neg1 = neg_conf_t != zeros
        neg_conf_t[neg1] = 1
        

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx1].view(-1, 4)
        loc_t = loc_t[pos_idx1].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')


        ####################################################################
        # Compute max conf across batch for hard negative mining
        batch_conf0 = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf0) - batch_conf0.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[neg1.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos1.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos1.size(1)-1)
        neg0_1 = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx0 = pos1.unsqueeze(2).expand_as(conf_data)
        neg_idx0 = neg0_1.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx0+neg_idx0).gt(0)].view(-1,self.num_classes)
        targets = conf_t[(pos1+neg0_1).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets, reduction='sum')
        ####################################################################
        
        if self.condition_apply == True:
            blur_targets = blur_t[pos1]
            occl_targets = occl_t[pos1]
            
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data_we.view(-1, self.num_classes)
            loss_c_we = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c_we[neg1.view(-1, 1)] = 0 # filter out pos boxes for now
            loss_c_we = loss_c_we.view(num, -1)
            _, loss_idx_we = loss_c_we.sort(1, descending=True)
            _, idx_rank_we = loss_idx_we.sort(1)
            num_pos_we = pos1.long().sum(1, keepdim=True)
            num_neg_we = torch.clamp(self.negpos_ratio*num_pos_we, max=pos1.size(1)-1)
            neg1_1 = idx_rank_we < num_neg_we.expand_as(idx_rank_we)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos1.unsqueeze(2).expand_as(conf_data_we)
            neg_idx = neg1_1.unsqueeze(2).expand_as(conf_data_we)
            conf_p_pos = conf_data_we[pos_idx.gt(0)].view(-1,self.num_classes)
            conf_p_neg = conf_data_we[neg_idx.gt(0)].view(-1,self.num_classes)
            targets_pos = conf_t[(pos1).gt(0)]
            targets_neg = conf_t[(neg1_1).gt(0)]
            
            blur_weights = torch.ones_like(blur_targets, dtype=torch.float)
            blur_weights[blur_targets == 1] = 1.3
            blur_weights[blur_targets == 2] = 1.7
            
            occl_weights = torch.ones_like(occl_targets, dtype=torch.float)
            occl_weights[occl_targets == 1] = 1.3
            occl_weights[occl_targets == 2] = 1.7
            
            loss_c_pos = F.cross_entropy(conf_p_pos, targets_pos, reduction='none')  # shape: [N_pos]
            loss_c_pos = (loss_c_pos * blur_weights*occl_weights).sum()

            # Normal face classification loss for negative anchors
            loss_c_neg = F.cross_entropy(conf_p_neg, targets_neg, reduction='sum')

            # Final face classification loss
            loss_c_we = loss_c_pos + loss_c_neg

            ################################# BLUR_HEAD ################################
            pos2 = blur_conf_t != zeros
            blur_conf_t[pos2] = 1

            neg2 = blur_neg_conf_t != zeros
            blur_neg_conf_t[neg2] = 1


            # Compute max conf across batch for hard negative mining
            batch_conf2 = conf_data_blur.view(-1, self.num_classes)
            loss_c_blur = log_sum_exp(batch_conf2) - batch_conf2.gather(1, blur_conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c_blur[neg2.view(-1, 1)] = 0 # filter out pos boxes for now
            loss_c_blur = loss_c_blur.view(num, -1)
            _, loss_idx_blur = loss_c_blur.sort(1, descending=True)
            _, idx_rank_blur = loss_idx_blur.sort(1)
            num_pos_blur = pos2.long().sum(1, keepdim=True)
            num_neg_blur = torch.clamp(self.negpos_ratio*num_pos_blur, max=pos2.size(1)-1)
            neg2_1 = idx_rank_blur < num_neg_blur.expand_as(idx_rank_blur)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx2 = pos2.unsqueeze(2).expand_as(conf_data_blur)
            neg_idx2 = neg2_1.unsqueeze(2).expand_as(conf_data_blur)
            conf_p_blur = conf_data_blur[(pos_idx2+neg_idx2).gt(0)].view(-1,self.num_classes)
            targets_blur = blur_conf_t[(pos2+neg2_1).gt(0)]
            loss_c_blur = F.cross_entropy(conf_p_blur, targets_blur, reduction='sum')
            ################################# BLUR_HEAD ################################

            ################################# OCC_HEAD ################################
            pos3 = occ_conf_t != zeros
            occ_conf_t[pos3] = 1

            neg3 = occ_neg_conf_t != zeros
            occ_neg_conf_t[neg3] = 1


            # Compute max conf across batch for hard negative mining
            batch_conf3 = conf_data_occ.view(-1, self.num_classes)
            loss_c_occ = log_sum_exp(batch_conf3) - batch_conf3.gather(1, occ_conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c_occ[neg3.view(-1, 1)] = 0 # filter out pos boxes for now
            loss_c_occ = loss_c_occ.view(num, -1)
            _, loss_idx_occ = loss_c_occ.sort(1, descending=True)
            _, idx_rank_occ = loss_idx_occ.sort(1)
            num_pos_occ = pos3.long().sum(1, keepdim=True)
            num_neg_occ = torch.clamp(self.negpos_ratio*num_pos_occ, max=pos3.size(1)-1)
            neg3_1 = idx_rank_occ < num_neg_occ.expand_as(idx_rank_occ)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx3 = pos3.unsqueeze(2).expand_as(conf_data_occ)
            neg_idx3 = neg3_1.unsqueeze(2).expand_as(conf_data_occ)
            conf_p_occ = conf_data_occ[(pos_idx3+neg_idx3).gt(0)].view(-1,self.num_classes)
            targets_occ = occ_conf_t[(pos3+neg3_1).gt(0)]
            loss_c_occ = F.cross_entropy(conf_p_occ, targets_occ, reduction='sum')
            ################################# OCC_HEAD ################################
        
        
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N1 = max(num_pos.data.sum().float(), 1)
        if self.condition_apply == True:
            N2 = max(num_pos_blur.data.sum().float(), 1)
            N3 = max(num_pos_occ.data.sum().float(), 1)
        loss_l /= N1
        loss_c /= N1
        loss_landm /= NL
        if self.condition_apply == True:
            loss_c_we /= N1
            loss_c_blur /= N2
            loss_c_occ /= N3
        
        if self.condition_apply == True:
            return loss_l, loss_c, loss_c_we, loss_c_blur, loss_c_occ, loss_landm
        else:
            return loss_l, loss_c, loss_landm
