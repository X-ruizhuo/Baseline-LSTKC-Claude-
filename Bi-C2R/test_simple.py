"""
简化的LSTKC++模块测试 - 不依赖完整训练环境
"""
import sys
import os
sys.path.insert(0, '/home/Baseline-LSTKC-Claude-/Bi-C2R')

def test_imports():
    """测试核心模块导入"""
    print("=" * 60)
    print("测试 LSTKC++ 模块导入")
    print("=" * 60)

    try:
        from reid.models.lstkc_modules import (
            KnowledgeDecomposition,
            KnowledgeFilter,
            AdaptiveKnowledgeIntegration
        )
        print("✓ LSTKC++模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transnet_integration():
    """测试TransNet集成"""
    print("\n" + "=" * 60)
    print("测试 TransNet_adaptive 集成")
    print("=" * 60)

    try:
        from reid.models.resnet import TransNet_adaptive

        # 测试baseline模式
        model_baseline = TransNet_adaptive(enable_lstkc=False)
        print("✓ TransNet_adaptive (baseline模式) 创建成功")

        # 测试LSTKC模式
        model_lstkc = TransNet_adaptive(enable_lstkc=True)
        print("✓ TransNet_adaptive (LSTKC模式) 创建成功")

        return True
    except Exception as e:
        print(f"✗ TransNet集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试配置文件")
    print("=" * 60)

    try:
        from config import cfg
        from yacs.config import CfgNode

        # 测试加载base配置
        cfg.merge_from_file('config/base.yml')
        print("✓ config/base.yml 加载成功")

        # 检查LSTKC配置段
        if hasattr(cfg, 'LSTKC'):
            print(f"✓ LSTKC配置段存在")
            print(f"  - ENABLE: {cfg.LSTKC.ENABLE}")
            print(f"  - LONG_TERM_WEIGHT: {cfg.LSTKC.LONG_TERM_WEIGHT}")
            print(f"  - SHORT_TERM_WEIGHT: {cfg.LSTKC.SHORT_TERM_WEIGHT}")
        else:
            print("✗ LSTKC配置段不存在")
            return False

        # 检查MEMORY配置段
        if hasattr(cfg, 'MEMORY'):
            print(f"✓ MEMORY配置段存在")
            print(f"  - MIXED_PRECISION: {cfg.MEMORY.MIXED_PRECISION}")
            print(f"  - GRADIENT_ACCUMULATION: {cfg.MEMORY.GRADIENT_ACCUMULATION}")
        else:
            print("✗ MEMORY配置段不存在")
            return False

        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("Bi-C2R + LSTKC++ 简化集成测试")
    print("=" * 60 + "\n")

    os.chdir('/home/Baseline-LSTKC-Claude-/Bi-C2R')

    results = []
    results.append(("模块导入", test_imports()))
    results.append(("TransNet集成", test_transnet_integration()))
    results.append(("配置文件", test_config()))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ 核心模块集成成功！")
        print("\n后续步骤:")
        print("1. 安装完整依赖: pip install -r requirement.txt")
        print("2. 在GPU环境测试: python test_lstkc_integration.py")
        print("3. 运行训练: bash run_lstkc.sh")
    else:
        print("\n✗ 部分测试失败")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
