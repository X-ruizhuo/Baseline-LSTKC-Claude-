import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDecomposition(nn.Module):
    """长短期知识分解模块 - LSTKC++核心组件"""
    def __init__(self, in_planes=2048):
        super().__init__()
        # 长期知识提取器 (跨域稳定特征)
        self.long_term_extractor = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2),
            nn.BatchNorm1d(in_planes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes)
        )
        # 短期知识提取器 (任务特定特征)
        self.short_term_extractor = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2),
            nn.BatchNorm1d(in_planes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes)
        )
        # 知识门控机制
        self.gate = nn.Sequential(
            nn.Linear(in_planes, in_planes),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        long_term = self.long_term_extractor(feat)
        short_term = self.short_term_extractor(feat)
        gate_weight = self.gate(feat)

        decomposed_feat = gate_weight * long_term + (1 - gate_weight) * short_term
        return decomposed_feat, long_term, short_term, gate_weight


class KnowledgeFilter(nn.Module):
    """知识过滤模块 - 评估旧知识质量"""
    def __init__(self, feature_dim=2048, threshold=0.6):
        super().__init__()
        self.threshold = threshold
        self.quality_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, old_feat, new_feat):
        """评估旧知识质量,返回置信度分数"""
        combined = torch.cat([old_feat, new_feat], dim=1)
        quality_score = self.quality_estimator(combined)
        return quality_score

    def get_filtered_features(self, old_feat, new_feat):
        """返回过滤后的特征"""
        quality_score = self.forward(old_feat, new_feat)
        mask = (quality_score > self.threshold).float()
        filtered_feat = old_feat * quality_score * mask + old_feat * (1 - mask) * 0.5
        return filtered_feat, quality_score


class AdaptiveKnowledgeIntegration(nn.Module):
    """自适应知识整合模块 - 增强的alpha计算"""
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.stability_estimator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.importance_estimator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_stability(self, long_term_knowledge):
        """计算长期知识稳定性"""
        stability = self.stability_estimator(long_term_knowledge)
        return stability.mean()

    def compute_importance(self, short_term_knowledge):
        """计算短期知识重要性"""
        importance = self.importance_estimator(short_term_knowledge)
        return importance.mean()

    def compute_enhanced_alpha(self, affinity_diff, long_term_old, short_term_old):
        """增强的alpha计算"""
        long_term_stability = self.compute_stability(long_term_old)
        short_term_importance = self.compute_importance(short_term_old)

        alpha = (1 - affinity_diff) * long_term_stability + \
                affinity_diff * (1 - short_term_importance)

        alpha = torch.clamp(alpha, 0.3, 0.9)
        return alpha, long_term_stability, short_term_importance


class LSTKCEnhancedTransNet(nn.Module):
    """集成LSTKC++的增强转换网络"""
    def __init__(self, base_transnet, in_planes=2048):
        super().__init__()
        self.base_transnet = base_transnet
        self.knowledge_decomp = KnowledgeDecomposition(in_planes)

    def forward(self, x, return_decomposition=False):
        trans_feat = self.base_transnet(x)

        if return_decomposition:
            decomposed, long_term, short_term, gate = self.knowledge_decomp(trans_feat)
            return decomposed, long_term, short_term, gate
        else:
            return trans_feat
