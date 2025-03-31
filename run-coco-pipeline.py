import os
import sys
import subprocess
import time
import argparse

def run_script(script_path, description, args=None):
    """运行给定的Python脚本并检查执行状态
    
    Args:
        script_path: 脚本文件路径
        description: 脚本描述
        args: 传递给脚本的命令行参数列表
    """
    print("="*80)
    print(f"开始执行: {description}")
    print(f"脚本文件: {script_path}")
    if args:
        print(f"参数: {' '.join(args)}")
    print("="*80)
    
    # 检查文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 脚本文件 '{script_path}' 不存在")
        return False
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 构建命令
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
            
        # 运行脚本
        result = subprocess.run(cmd, check=True)
        
        # 计算运行时间
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print(f"成功完成: {description}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print("="*80 + "\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 脚本执行失败，返回代码 {e.returncode}")
        return False
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='COCO图像标题处理流水线')
    parser.add_argument('--num-images', type=int, default=20, 
                        help='要下载的图片数量 (默认: 20)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新下载图片，即使数据库已有数据')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='随机种子，用于图片选择 (默认: 使用系统时间)')
    parser.add_argument('--clean', action='store_true',
                        help='在处理前清空现有数据库')
    args = parser.parse_args()
    
    # 设置路径
    results_dir = "coco_processing_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建运行日志文件
    log_file_path = os.path.join(results_dir, f"processing_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 保存原始的标准输出和标准错误
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    print(f"处理日志将保存到: {log_file_path}")
    print(f"设置下载图片数量: {args.num_images}")
    if args.force:
        print("已启用强制下载模式")
    if args.random_seed is not None:
        print(f"使用随机种子: {args.random_seed}")
    
    try:
        # 将输出重定向到日志文件
        log_file = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file
        
        print("COCO图像标题处理流水线")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"下载图片数量: {args.num_images}")
        if args.force:
            print("强制下载模式: 启用")
        if args.random_seed is not None:
            print(f"随机种子: {args.random_seed}")
        print("="*80)
        
        # 如果需要清空数据库
        if args.clean:
            print("\n清空现有数据库...")
            clean_success = run_script("clean-database.py", "清空数据库和图片", ["--auto-confirm"])
            if not clean_success:
                print("警告: 清空数据库操作失败，但将继续执行流水线")
        
        # 定义要运行的脚本及其描述
        scripts = []
        
        # COCO数据库创建和图片下载
        coco_db_args = ["--num-images", str(args.num_images)]
        if args.force:
            coco_db_args.append("--force")
        if args.random_seed is not None:
            coco_db_args.extend(["--random-seed", str(args.random_seed)])
        
        scripts.append(("coco-database-creation.py", "第1步: 创建COCO数据库并下载图片", coco_db_args))
        scripts.append(("vocabulary-builder.py", "第2步: 构建词汇表", None))
        scripts.append(("vocab-to-database.py", "第3步: 将词汇表导入数据库", None))
        scripts.append(("remove-stopwords.py", "第4步: 从词汇表中移除停用词", None))
        
        # 顺序运行所有脚本
        for i, (script, description, script_args) in enumerate(scripts):
            success = run_script(script, description, script_args)
            
            # 如果脚本执行失败，则中止整个流程
            if not success:
                print(f"\n错误: '{description}' 执行失败，中止后续处理")
                break
                
            # 如果不是最后一个脚本，等待几秒后继续
            if i < len(scripts) - 1:
                print("等待5秒后继续下一步...")
                time.sleep(5)
        
        print("\n" + "="*80)
        print(f"处理完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    finally:
        # 恢复标准输出和标准错误
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        if 'log_file' in locals():
            log_file.close()
            print(f"处理完成! 详细日志已保存至: {log_file_path}")

if __name__ == "__main__":
    print("开始COCO图像标题处理流水线...")
    main()
    print("流水线执行结束!")