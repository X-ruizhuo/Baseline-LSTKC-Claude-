"""
Long-Short Term Knowledge Consolidation Loss Functions
长短期知识解耦损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from reid.metric_learning.distance import cosine_similarity


class LSTKCLoss(nn.Module):
    """
    长短期知识解耦损失函数
    """
    def __init__(self,
                 long_align_weight=1.0,
                 long_relation_weight=2.0,
                 short_adapt_weight=0.5,
                 short_relation_weight=0.5):
        super(LSTKCLoss, self).__init__()

        self.long_align_weight = long_align_weight
        self.long_relation_weight = long_relation_weight
        self.short_adapt_weight = short_adapt_weight
        self.short_relation_weight = short_relation_weight

        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def contrastive_relation_loss(self, targets, features_old, features_trans):
        """
        关系保持损失 - 保持样本间的相似度关系
        """
        local_pids = targets.expand(len(targets), len(targets))
        pid_mask = (local_pids == local_pids.T)

        old_sim = features_old @ features_old.T
        trans_sim = features_trans @ features_trans.T

        old_sim_prob = F.softmax(old_sim, dim=1)
        trans_sim_prob = F.softmax(trans_sim, dim=1)

        old_sim_prob_unpair = torch.where(pid_mask, 0, old_sim_prob)
        trans_sim_prob_unpair = torch.where(pid_mask, 0, trans_sim_prob)

        old_sim_prob_unpair = old_sim_prob_unpair / (old_sim_prob_unpair.sum(-1, keepdim=True) + 1e-8)
        trans_sim_prob_unpair = trans_sim_prob_unpair / (trans_sim_prob_unpair.sum(-1, keepdim=True) + 1e-8)

        old_sim_prob_unpair = torch.where(pid_mask, trans_sim_prob, old_sim_prob_unpair)
        trans_sim_prob_unpair = torch.where(pid_mask, trans_sim_prob, trans_sim_prob_unpair)

        trans_sim_prob_log = torch.log(trans_sim_prob_unpair + 1e-8)

        return self.kl_loss(trans_sim_prob_log, old_sim_prob_unpair)

    def compute_adaptive_weights(self, trans_long_features, trans_short_features, current_features):
        """
        计算自适应融合权重
        """
        sim_long = F.cosine_similarity(trans_long_features, current_features, dim=1).mean()
        sim_short = F.cosine_similarity(trans_short_features, current_features, dim=1).mean()

        sim_long = torch.clamp(sim_long, min=0.0)
        sim_short = torch.clamp(sim_short, min=0.0)

        total_sim = sim_long + sim_short + 1e-8
        weight_long = sim_long / total_sim
        weight_short = sim_short / total_sim

        return weight_long, weight_short

    def forward(self, targets, current_features,
                feat_long, trans_long_features,
                feat_short, trans_short_features):
        """
        前向传播计算LSTKC损失

        Args:
            targets: 样本标签
            current_features: 当前模型特征
            feat_long: 长期模型特征
            trans_long_features: 转换后的长期特征
            feat_short: 短期模型特征
            trans_short_features: 转换后的短期特征

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 计算自适应权重
        weight_long, weight_short = self.compute_adaptive_weights(
            trans_long_features, trans_short_features, current_features
        )

        # 长期知识损失 (强约束)
        loss_long_align = self.mse_loss(trans_long_features, current_features)
        loss_long_relation = self.contrastive_relation_loss(
            targets, feat_long, trans_long_features
        )
        loss_long = (self.long_align_weight * loss_long_align +
                    self.long_relation_weight * loss_long_relation)

        # 短期知识损失 (弱约束)
        loss_short_adapt = self.mse_loss(trans_short_features, current_features)
        loss_short_relation = self.contrastive_relation_loss(
            targets, feat_short, trans_short_features
        )
        loss_short = (self.short_adapt_weight * loss_short_adapt +
                     self.short_relation_weight * loss_short_relation)

        # 自适应融合
        total_loss = weight_long * loss_long + weight_short * loss_short

        loss_dict = {
            'loss_long_align': loss_long_align.item(),
            'loss_long_relation': loss_long_relation.item(),
            'loss_short_adapt': loss_short_adapt.item(),
            'loss_short_relation': loss_short_relation.item(),
            'weight_long': weight_long.item(),
            'weight_short': weight_short.item(),
            'loss_lstkc_total': total_loss.item()
        }

        return total_loss, loss_dict


class DomainConfusionLoss(nn.Module):
    """
    域混淆损失 - 用于长期知识模型训练
    鼓励特征的域不变性
    """
    def __init__(self, weight=0.1):
        super(DomainConfusionLoss, self).__init__()
        self.weight = weight

    def forward(self, features):
        """
        计算域混淆损失
        通过最大化特征间相似度来减少域特异性
        """
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = features_norm @ features_norm.T

        # 排除对角线 (自身相似度)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        # 负号表示最大化相似度
        loss = -similarity_matrix.mean() * self.weight

        return loss
