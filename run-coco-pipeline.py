import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """运行给定的Python脚本并检查执行状态"""
    print("="*80)
    print(f"开始执行: {description}")
    print(f"脚本文件: {script_path}")
    print("="*80)
    
    # 检查文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 脚本文件 '{script_path}' 不存在")
        return False
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_path], check=True)
        
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
    # 定义要运行的脚本及其描述
    scripts = [
        ("coco-database-creation.py", "第1步: 创建COCO数据库并下载图片"),
        ("vocabulary-builder.py", "第2步: 构建词汇表"),
        ("vocab-to-database.py", "第3步: 将词汇表导入数据库"),
        ("remove-stopwords.py", "第4步: 从词汇表中移除停用词")
    ]
    
    # 创建目录记录结果
    results_dir = "coco_processing_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建运行日志文件
    log_file_path = os.path.join(results_dir, f"processing_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 保存原始的标准输出和标准错误
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    print(f"处理日志将保存到: {log_file_path}")
    
    try:
        # 将输出重定向到日志文件
        log_file = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file
        
        print("COCO图像标题处理流水线")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 顺序运行所有脚本
        for i, (script, description) in enumerate(scripts):
            success = run_script(script, description)
            
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
