"""
Long-Short Term Knowledge Decoupling Module
长短期知识解耦模块 - 核心创新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from reid.loss.lstkc_loss import DomainConfusionLoss


def decouple_knowledge(model_old, init_loader, args, epochs=5, lr=0.001):
    """
    知识解耦函数 - 将旧模型解耦为长期和短期知识模型

    Args:
        model_old: 旧任务训练完成的模型
        init_loader: 初始化数据加载器
        args: 命令行参数
        epochs: 解耦训练轮数 (默认5)
        lr: 学习率 (默认0.001)

    Returns:
        model_long: 长期知识模型 (通用特征)
        model_short: 短期知识模型 (特异特征)
    """
    print("=" * 80)
    print("开始知识解耦训练...")
    print("=" * 80)

    # 1. 复制两个独立的模型副本
    model_long = copy.deepcopy(model_old)
    model_short = copy.deepcopy(model_old)

    model_long.cuda()
    model_short.cuda()

    # 2. 冻结 backbone,只训练后续层
    for param in model_long.module.base.parameters():
        param.requires_grad = False
    for param in model_short.module.base.parameters():
        param.requires_grad = False

    print("已冻结 backbone,仅训练后续层")

    # 3. 设置优化器
    params_long = []
    params_short = []

    for key, value in model_long.named_parameters():
        if value.requires_grad:
            params_long.append({"params": [value], "lr": lr, "weight_decay": 1e-4})

    for key, value in model_short.named_parameters():
        if value.requires_grad:
            params_short.append({"params": [value], "lr": lr, "weight_decay": 1e-4})

    optimizer_long = torch.optim.SGD(params_long, momentum=0.9)
    optimizer_short = torch.optim.SGD(params_short, momentum=0.9)

    # 4. 损失函数
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    domain_confusion_loss = DomainConfusionLoss(weight=0.1)

    # 5. 训练长期知识模型
    print("\n训练长期知识模型 (域不变特征)...")
    model_long.train()
    model_old.eval()

    for epoch in range(epochs):
        total_loss_long = 0.0
        total_recon_loss = 0.0
        total_domain_loss = 0.0
        num_batches = 0

        for batch_idx, inputs in enumerate(init_loader):
            imgs, pids, _, _, _ = inputs
            imgs = imgs.cuda()
            pids = pids.cuda()

            # 提取旧模型特征 (作为目标)
            with torch.no_grad():
                feat_old, _, _, _ = model_old(imgs, get_all_feat=True)
                if isinstance(feat_old, tuple):
                    feat_old = feat_old[0]

            # 提取长期模型特征
            feat_long, _, _, _ = model_long(imgs, get_all_feat=True)
            if isinstance(feat_long, tuple):
                feat_long = feat_long[0]

            # 特征重构损失
            loss_recon = mse_loss(feat_long, feat_old.detach())

            # 域混淆损失 (鼓励通用性)
            loss_domain = domain_confusion_loss(feat_long)

            # 总损失
            loss = loss_recon + loss_domain

            optimizer_long.zero_grad()
            loss.backward()
            optimizer_long.step()

            total_loss_long += loss.item()
            total_recon_loss += loss_recon.item()
            total_domain_loss += loss_domain.item()
            num_batches += 1

            if batch_idx >= 100:  # 限制每个epoch的迭代次数
                break

        avg_loss = total_loss_long / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_domain = total_domain_loss / num_batches

        print(f"Epoch [{epoch+1}/{epochs}] Long-term: "
              f"Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Domain={avg_domain:.4f}")

    # 6. 训练短期知识模型
    print("\n训练短期知识模型 (场景特异特征)...")
    model_short.train()

    for epoch in range(epochs):
        total_loss_short = 0.0
        total_recon_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for batch_idx, inputs in enumerate(init_loader):
            imgs, pids, _, _, _ = inputs
            imgs = imgs.cuda()
            pids = pids.cuda()

            # 提取旧模型特征
            with torch.no_grad():
                feat_old, _, _, _ = model_old(imgs, get_all_feat=True)
                if isinstance(feat_old, tuple):
                    feat_old = feat_old[0]

            # 提取短期模型特征和logits
            feat_short, bn_feat_short, logits_short, _ = model_short(imgs, get_all_feat=True)
            if isinstance(feat_short, tuple):
                feat_short = feat_short[0]

            # 精确重构损失
            loss_recon = mse_loss(feat_short, feat_old.detach())

            # 分类损失 (保持判别力)
            loss_cls = ce_loss(logits_short, pids) * 0.5

            # 正则化损失 (防止过度偏移)
            loss_reg = torch.norm(feat_short - feat_old.detach(), p=2, dim=1).mean() * 0.01

            # 总损失
            loss = loss_recon + loss_cls + loss_reg

            optimizer_short.zero_grad()
            loss.backward()
            optimizer_short.step()

            total_loss_short += loss.item()
            total_recon_loss += loss_recon.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            num_batches += 1

            if batch_idx >= 100:
                break

        avg_loss = total_loss_short / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_cls = total_cls_loss / num_batches
        avg_reg = total_reg_loss / num_batches

        print(f"Epoch [{epoch+1}/{epochs}] Short-term: "
              f"Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Cls={avg_cls:.4f}, Reg={avg_reg:.4f}")

    print("\n知识解耦完成!")
    print("=" * 80)

    # 7. 设置为评估模式
    model_long.eval()
    model_short.eval()

    return model_long, model_short
