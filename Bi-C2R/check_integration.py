"""
语法和导入检查脚本 - 验证代码集成的正确性
"""
import sys
import os
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_syntax(filepath):
    """检查Python文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_imports(filepath):
    """检查文件导入"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return True, imports
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Bi-C2R + LSTKC++ 代码验证")
    print("=" * 60)

    files_to_check = [
        'reid/models/lstkc_modules.py',
        'reid/models/resnet.py',
        'reid/trainer.py',
        'continual_train.py',
    ]

    all_passed = True

    for filepath in files_to_check:
        print(f"\n检查: {filepath}")

        if not os.path.exists(filepath):
            print(f"  ✗ 文件不存在")
            all_passed = False
            continue

        # 语法检查
        syntax_ok, error = check_syntax(filepath)
        if syntax_ok:
            print(f"  ✓ 语法正确")
        else:
            print(f"  ✗ 语法错误: {error}")
            all_passed = False
            continue

        # 导入检查
        imports_ok, imports = check_imports(filepath)
        if imports_ok:
            print(f"  ✓ 导入检查通过 ({len(imports)} 个导入)")
        else:
            print(f"  ✗ 导入检查失败: {imports}")
            all_passed = False

    # 检查配置文件
    print(f"\n检查配置文件:")
    config_files = [
        'config/base.yml',
        'config/lstkc_enhanced.yml',
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✓ {config_file} 存在")
        else:
            print(f"  ✗ {config_file} 不存在")
            all_passed = False

    # 检查脚本文件
    print(f"\n检查运行脚本:")
    scripts = [
        'run1.sh',
        'run2.sh',
        'run_lstkc.sh',
    ]

    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script} 存在")
        else:
            print(f"  ✗ {script} 不存在")
            all_passed = False

    # 检查文档
    print(f"\n检查文档:")
    docs = [
        'docs/Bi-C2R_LSTKC++_融合方案.md',
        'README_LSTKC.md',
    ]

    for doc in docs:
        if os.path.exists(doc):
            print(f"  ✓ {doc} 存在")
        else:
            print(f"  ✗ {doc} 不存在")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！代码集成完成。")
        print("\n后续步骤:")
        print("1. 在有GPU的环境中运行: python test_lstkc_integration.py")
        print("2. 训练baseline: bash run1.sh")
        print("3. 训练LSTKC++版本: bash run_lstkc.sh")
    else:
        print("✗ 部分检查失败，请修复错误。")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
