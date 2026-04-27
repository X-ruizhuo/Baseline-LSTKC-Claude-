"""
测试LSTKC++模块集成的正确性
"""
import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_lstkc_modules():
    """测试LSTKC++模块是否正确导入和运行"""
    print("=" * 60)
    print("测试 LSTKC++ 模块集成")
    print("=" * 60)

    try:
        from reid.models.lstkc_modules import (
            KnowledgeDecomposition,
            KnowledgeFilter,
            AdaptiveKnowledgeIntegration
        )
        print("✓ LSTKC++模块导入成功")
    except Exception as e:
        print(f"✗ LSTKC++模块导入失败: {e}")
        return False

    # 测试知识分解模块
    try:
        batch_size = 4
        feature_dim = 2048

        decomp = KnowledgeDecomposition(in_planes=feature_dim).cuda()
        test_feat = torch.randn(batch_size, feature_dim).cuda()

        decomposed, long_term, short_term, gate = decomp(test_feat)

        assert decomposed.shape == (batch_size, feature_dim)
        assert long_term.shape == (batch_size, feature_dim)
        assert short_term.shape == (batch_size, feature_dim)
        assert gate.shape == (batch_size, feature_dim)

        print(f"✓ KnowledgeDecomposition 测试通过")
        print(f"  - 输入形状: {test_feat.shape}")
        print(f"  - 输出形状: {decomposed.shape}")
        print(f"  - 门控均值: {gate.mean().item():.4f}")
    except Exception as e:
        print(f"✗ KnowledgeDecomposition 测试失败: {e}")
        return False

    # 测试知识过滤模块
    try:
        filter_module = KnowledgeFilter(feature_dim=feature_dim).cuda()
        old_feat = torch.randn(batch_size, feature_dim).cuda()
        new_feat = torch.randn(batch_size, feature_dim).cuda()

        quality_score = filter_module(old_feat, new_feat)
        filtered_feat, score = filter_module.get_filtered_features(old_feat, new_feat)

        assert quality_score.shape == (batch_size, 1)
        assert filtered_feat.shape == (batch_size, feature_dim)
        assert 0 <= quality_score.min() <= 1
        assert 0 <= quality_score.max() <= 1

        print(f"✓ KnowledgeFilter 测试通过")
        print(f"  - 质量分数范围: [{quality_score.min().item():.4f}, {quality_score.max().item():.4f}]")
        print(f"  - 质量分数均值: {quality_score.mean().item():.4f}")
    except Exception as e:
        print(f"✗ KnowledgeFilter 测试失败: {e}")
        return False

    # 测试自适应知识整合模块
    try:
        integration = AdaptiveKnowledgeIntegration(feature_dim=feature_dim).cuda()

        affinity_diff = torch.tensor(0.3).cuda()
        long_term_old = torch.randn(batch_size, feature_dim).cuda()
        short_term_old = torch.randn(batch_size, feature_dim).cuda()

        alpha, stability, importance = integration.compute_enhanced_alpha(
            affinity_diff, long_term_old, short_term_old
        )

        assert 0.3 <= alpha <= 0.9
        assert 0 <= stability <= 1
        assert 0 <= importance <= 1

        print(f"✓ AdaptiveKnowledgeIntegration 测试通过")
        print(f"  - Alpha: {alpha.item():.4f}")
        print(f"  - 长期稳定性: {stability.item():.4f}")
        print(f"  - 短期重要性: {importance.item():.4f}")
    except Exception as e:
        print(f"✗ AdaptiveKnowledgeIntegration 测试失败: {e}")
        return False

    return True


def test_transnet_integration():
    """测试TransNet与LSTKC++的集成"""
    print("\n" + "=" * 60)
    print("测试 TransNet_adaptive 集成")
    print("=" * 60)

    try:
        from reid.models.resnet import TransNet_adaptive

        # 测试不启用LSTKC
        model_baseline = TransNet_adaptive(enable_lstkc=False).cuda()
        test_feat = torch.randn(4, 2048).cuda()

        output_baseline = model_baseline(test_feat)
        assert output_baseline.shape == (4, 2048)
        print("✓ TransNet_adaptive (baseline) 测试通过")

        # 测试启用LSTKC
        model_lstkc = TransNet_adaptive(enable_lstkc=True).cuda()

        output_simple = model_lstkc(test_feat, return_decomposition=False)
        assert output_simple.shape == (4, 2048)
        print("✓ TransNet_adaptive (LSTKC, 简单模式) 测试通过")

        decomposed, long_term, short_term, gate = model_lstkc(test_feat, return_decomposition=True)
        assert decomposed.shape == (4, 2048)
        assert long_term.shape == (4, 2048)
        assert short_term.shape == (4, 2048)
        assert gate.shape == (4, 2048)
        print("✓ TransNet_adaptive (LSTKC, 分解模式) 测试通过")
        print(f"  - 门控均值: {gate.mean().item():.4f}")

    except Exception as e:
        print(f"✗ TransNet_adaptive 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_memory_usage():
    """测试显存使用情况"""
    print("\n" + "=" * 60)
    print("显存使用测试")
    print("=" * 60)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        from reid.models.resnet import make_model, TransNet_adaptive

        # 创建模型
        model = make_model(None, num_class=751, camera_num=0, view_num=0).cuda()
        model_trans = TransNet_adaptive(enable_lstkc=True).cuda()
        model_trans2 = TransNet_adaptive(enable_lstkc=True).cuda()

        # 模拟前向传播
        batch_size = 48
        test_input = torch.randn(batch_size, 3, 256, 128).cuda()

        with torch.no_grad():
            features, bn_feat, cls_outputs, feat_final = model(test_input)
            trans_feat = model_trans(features)

        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"✓ 显存测试完成")
        print(f"  - 当前分配: {memory_allocated:.2f} GB")
        print(f"  - 当前保留: {memory_reserved:.2f} GB")
        print(f"  - 峰值使用: {max_memory:.2f} GB")

        if max_memory < 32:
            print(f"  ✓ 显存使用在32GB限制内")
        else:
            print(f"  ✗ 警告: 显存使用超过32GB")

        return True

    except Exception as e:
        print(f"✗ 显存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Bi-C2R + LSTKC++ 集成测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("✗ CUDA不可用，无法运行测试")
        return

    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    results = []

    # 运行测试
    results.append(("LSTKC++模块", test_lstkc_modules()))
    results.append(("TransNet集成", test_transnet_integration()))
    results.append(("显存使用", test_memory_usage()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ 所有测试通过！框架集成成功。")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
