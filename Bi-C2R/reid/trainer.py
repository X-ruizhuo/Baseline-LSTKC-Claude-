from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.loss.softmax_loss import KnowledgeDistillation
from reid.utils.make_loss import make_loss
import copy

from reid.metric_learning.distance import cosine_similarity
class Trainer(object):
    def __init__(self,cfg,args, model, model_trans, model_trans2, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.model_trans = model_trans
        self.model_trans2 = model_trans2
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)

        self.criterion_transform_x = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.criterion_transform = nn.MSELoss()
        self.criterion_anti_forget = nn.KLDivLoss(reduction='batchmean')
      
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

        self.weight_trans = args.weight_trans
        self.weight_anti = args.weight_anti
        self.weight_discri = args.weight_discri
        self.weight_transx = args.weight_transx

    def loss_cr(self, targets_, s_features_old_, trans_old_features_norm_):

        local_pids_temp_ = targets_
        local_pids_temp_ = local_pids_temp_.expand(len(targets_), len(targets_))
        pid_mask_ = (local_pids_temp_ == local_pids_temp_.T)

        old_sim_ = s_features_old_ @ s_features_old_.T
        new_sim_ = trans_old_features_norm_ @ trans_old_features_norm_.T

        old_sim_prob_ = F.softmax(old_sim_, 1)
        new_sim_prob_ = F.softmax(new_sim_, 1)

        old_sim_prob_unpair_ = torch.where(pid_mask_, 0, old_sim_prob_)
        new_sim_prob_unpair_ = torch.where(pid_mask_, 0, new_sim_prob_)

        old_sim_prob_unpair_ = old_sim_prob_unpair_ / old_sim_prob_unpair_.sum(-1)
        new_sim_prob_unpair_ = new_sim_prob_unpair_ / new_sim_prob_unpair_.sum(-1)

        old_sim_prob_unpair_ = torch.where(pid_mask_, new_sim_prob_, old_sim_prob_unpair_)
        new_sim_prob_unpair_ = torch.where(pid_mask_, new_sim_prob_, new_sim_prob_unpair_)

        new_sim_prob_unpair_log_ = torch.log(new_sim_prob_unpair_)
        return self.weight_anti * self.criterion_anti_forget(new_sim_prob_unpair_log_, old_sim_prob_unpair_)

    def stkr_rectify(self, R_old, targets):
        """
        Short-Term Knowledge Rectification (STKR) from LSTKC++.
        Corrects erroneous relation scores in R_old using GT identity labels.
        R_old: [B, B] affinity matrix (softmax-normalized, each row sums to 1)
        targets: [B] identity labels
        Returns rectified matrix R_tilde, L1-normalized per row.
        """
        B = R_old.shape[0]
        same_id = (targets.unsqueeze(1) == targets.unsqueeze(0))  # [B, B]
        diff_id = ~same_id

        # sn[i] = min affinity among same-ID pairs for row i (negative threshold)
        # sp[i] = max affinity among diff-ID pairs for row i (positive threshold)
        R_same = R_old.masked_fill(diff_id, float('inf'))
        R_diff = R_old.masked_fill(same_id, float('-inf'))
        sn = R_same.min(dim=1, keepdim=True)[0]  # [B, 1]
        sp = R_diff.max(dim=1, keepdim=True)[0]  # [B, 1]

        # Clamp: intra-ID scores >= sp, inter-ID scores <= sn
        R_tilde = torch.where(same_id, torch.maximum(R_old, sp), torch.minimum(R_old, sn))

        # L1 normalize each row
        row_sum = R_tilde.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return R_tilde / row_sum

    def complementary_stkr(self, R_short, R_long, targets):
        """
        Complementary Short-Term Knowledge Rectification (C-STKR) from LSTKC++.
        Uses two old models' relation matrices to produce a better rectified target.
        """
        B = R_short.shape[0]
        same_id = (targets.unsqueeze(1) == targets.unsqueeze(0))  # [B, B]
        diff_id = ~same_id

        # Determine correctness for each element in each model
        # A score is "correct" if:
        #   same-ID pair: score >= max diff-ID score (sp) for that row
        #   diff-ID pair: score <= min same-ID score (sn) for that row
        def is_correct(R):
            R_same = R.masked_fill(diff_id, float('inf'))
            R_diff = R.masked_fill(same_id, float('-inf'))
            sn = R_same.min(dim=1, keepdim=True)[0]
            sp = R_diff.max(dim=1, keepdim=True)[0]
            correct_intra = same_id & (R >= sp)
            correct_inter = diff_id & (R <= sn)
            return correct_intra | correct_inter  # [B, B] bool

        correct_s = is_correct(R_short)
        correct_l = is_correct(R_long)

        # Rectify each model independently first
        R_tilde_s = self.stkr_rectify(R_short, targets)
        R_tilde_l = self.stkr_rectify(R_long, targets)

        # Complementary fusion:
        # both correct or both wrong -> average
        # only one correct -> use the correct one
        both_correct = correct_s & correct_l
        only_s_correct = correct_s & ~correct_l
        only_l_correct = ~correct_s & correct_l

        R_tilde = (R_tilde_s + R_tilde_l) / 2.0
        R_tilde = torch.where(only_s_correct, R_tilde_s, R_tilde)
        R_tilde = torch.where(only_l_correct, R_tilde_l, R_tilde)

        # L1 normalize each row
        row_sum = R_tilde.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return R_tilde / row_sum

    def get_affinity(self, model, imgs, tau=0.1):
        """Extract normalized affinity matrix from a frozen model."""
        with torch.no_grad():
            feats, _, _, _ = model(imgs, get_all_feat=True)
            if isinstance(feats, tuple):
                feats = feats[0]
        feats = F.normalize(feats, p=2, dim=1)
        sim = feats @ feats.T
        return F.softmax(sim / tau, dim=1)

    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None, old_model_long=None,
              ):

        self.model.train()
        self.model_trans.train()
        self.model_trans2.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()

        losses_ca = AverageMeter()
        losses_cr = AverageMeter()
        losses_ad = AverageMeter()
        losses_dc = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains, = self._parse_data(train_inputs)
            targets += add_num
            s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)

            '''calculate the base loss'''
            loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
            loss = loss_ce + loss_tp
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            if old_model is not None:
                with torch.no_grad():
                    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old=s_features_old[0]

                # LSTKC++ knowledge rectification distillation
                Affinity_matrix_new = self.get_normal_affinity(s_features)
                if old_model_long is not None:
                    # C-STKR: complementary rectification using both short-term and long-term old models
                    R_short = self.get_affinity(old_model, s_inputs)
                    R_long  = self.get_affinity(old_model_long, s_inputs)
                    R_tilde = self.complementary_stkr(R_short, R_long, targets)
                else:
                    # STKR: single old model rectification (step t=2)
                    R_old = self.get_affinity(old_model, s_inputs)
                    R_tilde = self.stkr_rectify(R_old, targets)
                divergence = self.KLDivLoss(torch.log(Affinity_matrix_new + 1e-8), R_tilde)
                loss = loss + divergence * self.AF_weight

                
                trans_old_features = self.model_trans(s_features_old)
                trans_old_features_norm = F.normalize(trans_old_features, p=2, dim=1)

                trans_new_features = self.model_trans2(s_features)
                trans_new_features_norm = F.normalize(trans_new_features, p=2, dim=1)
                
                trans_loss = self.weight_trans * self.criterion_transform(trans_old_features_norm, s_features)\
                           + self.weight_trans * self.criterion_transform(trans_new_features_norm, s_features_old)
                losses_ca.update(trans_loss.item())
                
                anti_loss = self.loss_cr(targets, s_features_old, trans_old_features_norm) + self.loss_cr(targets, s_features, trans_new_features_norm)
                
                losses_cr.update(anti_loss.item())
                
                
                
                s_features_old_origin = old_model.module.pooling_layer(feat_final_layer_old)[..., 0, 0]
                mean_old_features = s_features_old_origin.mean(dim=-1, keepdim=True).detach()
                std_old_features = s_features_old_origin.std(dim=-1, keepdim=True, unbiased=False).detach()
                trans_old_features_norm_unnorm = trans_old_features_norm * std_old_features + mean_old_features
                bn_trans_old_features_norm_unnorm = old_model.module.bottleneck(trans_old_features_norm_unnorm.unsqueeze(-1).unsqueeze(-1))
                trans_old_logit = old_model.module.classifier(bn_trans_old_features_norm_unnorm[..., 0, 0])
                discri_loss_forward = self.weight_discri * KnowledgeDistillation(trans_old_logit, cls_outputs_old[:,:])


                s_features_new_origin = self.model.module.pooling_layer(feat_final_layer)[..., 0, 0]
                mean_new_features = s_features_new_origin.mean(dim=-1, keepdim=True).detach()
                std_new_features = s_features_new_origin.std(dim=-1, keepdim=True, unbiased=False).detach()
                trans_new_features_norm_unnorm = trans_new_features_norm * std_new_features + mean_new_features
                bn_trans_new_features_norm_unnorm = self.model.module.bottleneck(trans_new_features_norm_unnorm.unsqueeze(-1).unsqueeze(-1))
                trans_new_logit = self.model.module.classifier(bn_trans_new_features_norm_unnorm[..., 0, 0])
                discri_loss_backward = self.weight_discri * KnowledgeDistillation(trans_new_logit, cls_outputs[:,:])

                discri_loss = discri_loss_forward + discri_loss_backward
                losses_ad.update(discri_loss.item())
                
                trans_x_loss_forward = self.weight_transx * (1-self.criterion_transform_x(F.normalize(s_features-s_features_old, p=2, dim=1), F.normalize(trans_old_features_norm-s_features_old, p=2, dim=1)).mean())
                trans_x_loss_backward = self.weight_transx * (1-self.criterion_transform_x(F.normalize(s_features_old-s_features, p=2, dim=1), F.normalize(s_features_old-trans_new_features_norm, p=2, dim=1)).mean())
                trans_x_loss = trans_x_loss_forward + trans_x_loss_backward
                losses_dc.update(trans_x_loss.item())
                
                loss = loss + trans_loss + anti_loss + discri_loss + trans_x_loss
                
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()           

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_ca_{}".format(training_phase), scalar_value=losses_ca.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_cr_{}".format(training_phase), scalar_value=losses_cr.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_ad_{}".format(training_phase), scalar_value=losses_ad.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_dc_{}".format(training_phase), scalar_value=losses_dc.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      'Loss_ca {:.3f} ({:.3f})\t'
                      'Loss_cr {:.3f} ({:.3f})\t'
                      'Loss_ad {:.3f} ({:.3f})\t'
                      'Loss_dc {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_ca.val, losses_ca.avg,
                              losses_cr.val, losses_cr.avg,
                              losses_ad.val, losses_ad.avg,
                              losses_dc.val, losses_dc.avg,
                  ))       

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old,targets):
        Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
        Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        '''# prediction is correct on old model'''
        Old_Keep = attri_old['TN'] + attri_old['TP']
        Target_1 = Affinity_matrix_old * Old_Keep
        '''# prediction is false on old model but correct on mew model'''
        New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        Target_2 = Affinity_matrix_new * New_keep
        '''# both missed correct person'''
        Hard_pos = attri_new['FN'] * attri_old['FN']
        Thres_P = torch.maximum(attri_new['Thres_P'], attri_old['Thres_P'])
        Target_3 = Hard_pos * Thres_P

        '''# both false wrong person'''
        Hard_neg = attri_new['FP'] * attri_old['FP']
        Thres_N = torch.minimum(attri_new['Thres_N'], attri_old['Thres_N'])
        Target_4 = Hard_neg * Thres_N

        Target__ = Target_1 + Target_2 + Target_3 + Target_4
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization


        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)

        return divergence

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))

        FN=Gts-TP

        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris

