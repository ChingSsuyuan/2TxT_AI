# COCO图像标题处理流水线使用指南

## 概述

COCO图像标题处理流水线是一个自动化工具，用于下载、处理和分析COCO数据集的图像及其相应的标题。该流水线包括以下主要功能：

1. 创建数据库并下载COCO图像
2. 构建词汇表
3. 将词汇表导入数据库
4. 移除停用词
5. 使用ResNet模型进行图像特征编码

## 安装依赖

在使用流水线之前，请确保已安装所需的依赖项：

```bash
pip install torch torchvision tqdm pillow numpy
```

## 基本用法

### 运行完整流水线

最简单的用法是直接运行流水线脚本，这将使用默认参数执行所有处理步骤：

```bash
python run-coco-pipeline.py
```

### 常用参数

流水线提供了多个命令行参数来自定义处理行为：

```bash
# 指定要下载的图片数量
python run-coco-pipeline.py --num-images 50

# 强制重新下载图片，即使数据库中已有数据
python run-coco-pipeline.py --force

# 在处理前清空现有数据库
python run-coco-pipeline.py --clean

# 启用调试输出
python run-coco-pipeline.py --debug

# 使用特定的随机种子(用于图片选择的可重复性)
python run-coco-pipeline.py --random-seed 42
```

### 图像编码选项

```bash
# 选择使用的ResNet模型类型
python run-coco-pipeline.py --resnet-model resnet18
# 可选: resnet18, resnet34, resnet50, resnet101, resnet152(默认为resnet50)

# 跳过图像编码步骤
python run-coco-pipeline.py --skip-encoding
```

### 组合参数示例

```bash
# 下载100张图片，使用resnet101进行编码，并启用调试模式
python run-coco-pipeline.py --num-images 100 --resnet-model resnet101 --debug

# 清空数据库，重新下载30张图片，并使用轻量级的resnet18
python run-coco-pipeline.py --clean --num-images 30 --resnet-model resnet18 --force
```

## 高级用法

### 单独运行图像编码器

如果只需要执行图像编码步骤，可以直接运行图像编码器脚本：

```bash
python image-encoder-db.py --model resnet50
```

图像编码器的其他参数：

```bash
# 指定数据库路径
python image-encoder-db.py --db-path custom/path/to/database.db

# 指定图像目录
python image-encoder-db.py --images-dir custom/path/to/images

# 设置批处理大小
python image-encoder-db.py --batch-size 32

# 测试特定图像ID的特征检索
python image-encoder-db.py --test-image-id 9
```

## 输出和日志

流水线会在`coco_processing_results`目录下生成日志文件，格式为：

```
processing_log_YYYYMMDD_HHMMSS.txt
```

日志文件包含每个处理步骤的详细信息，包括执行时间、成功/失败状态、以及任何错误消息。

## 数据存储

处理后的数据存储在以下位置：

- 数据库文件: `coco_image_title_data/image_title_database.db`
- 下载的图像: `coco_image_title_data/images/`

## 故障排除

### 常见问题

1. **流水线在图像下载步骤失败**
   - 检查网络连接
   - 确保COCO API可访问
   - 尝试减少`--num-images`参数值

2. **图像编码过程耗时过长**
   - 对于大量图像，考虑使用较小的模型(如`--resnet-model resnet18`)
   - 减小批处理大小以减少内存使用(`--batch-size`)
   - 如果只需处理文本数据，可以使用`--skip-encoding`选项

3. **GPU内存不足错误**
   - 减小批处理大小
   - 使用较小的ResNet模型
   - 确保没有其他进程占用GPU内存

### 调试提示

启用调试模式可以获取更详细的处理信息：

```bash
python run-coco-pipeline.py --debug
```

这将输出更多的中间状态信息，帮助识别可能的问题所在。

## 扩展和定制

流水线设计为模块化结构，可以通过修改单个脚本来定制特定步骤。例如：

- 修改`image-encoder-db.py`以使用不同的深度学习模型
- 自定义`vocabulary-builder.py`以实现特定的词汇处理逻辑
- 添加新的处理步骤，如文本分析或图像分割
