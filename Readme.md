# 基于CNN+Transformer的图像自动标题生成系统

这个项目使用预训练CNN提取图像特征，并通过Transformer解码器生成相关的图像标题。

## 项目概述

本系统通过以下步骤自动为图像生成高质量、相关性强的标题：
1. 使用预训练CNN (ResNet50) 提取图像特征
2. 将特征存储在数据库中
3. 使用Transformer解码器将图像特征转换为自然语言标题

## 系统架构

```
图像 → CNN特征提取 → 特征数据库 → Transformer解码器 → 自然语言标题
```

## 流程详解

### 1. 图像编码部分 (CNN)

- **位置**: 模型的第一部分，处理视觉输入
- **实现**: 使用预训练ResNet50提取特征
- **输入**: 图像张量 `[批量大小, 3, H, W]`
- **输出**: 全局图像特征 `[批量大小, 2048]`
- **存储**: 特征以BLOB格式存储在SQLite数据库中

### 2. 文本处理

- **词汇表构建**: 基于数据集中的现有标题
- **标记化**: 将文本分词并映射到索引
- **特殊标记**: 包含`<START>`, `<END>`, `<PAD>`, `<UNK>`等标记
- **序列准备**: 为Transformer解码器准备输入序列

### 3. Transformer解码器

- **架构**: 标准Transformer解码器层
- **输入**: 
  - 图像特征作为上下文
  - 目标序列的移位版本作为输入
- **注意力机制**: 
  - 自注意力层处理文本上下文
  - 交叉注意力层连接图像特征和文本
- **训练目标**: 使用教师强制训练，预测下一个标记

### 4. 训练过程

- **数据加载**: 从数据库加载图像特征和对应标题
- **批处理准备**: 将特征和标题组织成批次
- **优化**: 使用Adam优化器和交叉熵损失
- **学习率调度**: 使用预热和衰减策略
- **评估**: 使用BLEU、ROUGE等指标评估生成质量

### 5. 标题生成

- **特征输入**: 将新图像的特征向量输入到模型
- **自回归生成**: 逐标记生成文本序列
- **Beam Search**: 使用束搜索提高生成质量
- **后处理**: 移除特殊标记，格式化最终标题

## 数据库结构

项目使用SQLite数据库存储图像特征和相关信息：

### 表结构

1. **images**: 存储图像元数据
   - id (主键)
   - coco_id
   - file_name
   - width/height
   - created_at
   - local_path

2. **captions**: 存储图像描述
   - id (主键)
   - image_id (外键，关联到images表)
   - caption

3. **vocabulary**: 词汇表
   - id (主键)
   - word
   - frequency
   - rank
   - percentage
   - special_token

4. **word_indices**: 词到索引的映射
   - word_id (外键，关联到vocabulary表)
   - index_id
   - special_token

5. **image_features_{model_name}**: 存储图像特征
   - id (主键)
   - image_id (外键，关联到images表)
   - features (BLOB类型，存储特征向量)
   - created_at

## 数据集信息

- **图片数量**: 60张图片
- **标题数量**: 300条描述 (每张图片约5个标题)
- **词汇量**: 275个单词

## 技术栈

- **特征提取**: PyTorch, ResNet50
- **数据存储**: SQLite
- **文本处理**: NLTK/spaCy
- **生成模型**: Transformer解码器(PyTorch实现)

## 安装与配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置数据路径
export IMAGE_DIR="coco_image_title_data/images"
export DB_PATH="coco_image_title_data/image_title_database.db"
```
# 指定下载10张图片
```
python run-coco-pipeline.py --num-images 10
```
## 使用方法

### 特征提取

```bash
python image_encoder_db.py --model resnet50 --db-path $DB_PATH --image-dir $IMAGE_DIR
```

### 模型训练

```bash
python train_transformer.py --features-table image_features_resnet50 --db-path $DB_PATH --batch-size 16 --epochs 50
```

### 标题生成

```bash
python generate_titles.py --model-path models/transformer_title_generator.pt --image-path sample.jpg
```

## 基线模型与扩展

本项目基于标准的CNN+Transformer架构，这是图像标题生成任务的公认基线方法。未来的扩展可能包括：

- 使用更先进的视觉编码器(如ViT或CLIP)
- 采用预训练语言模型(如T5或BART)进行微调
- 实现注意力可视化以解释生成过程

## 性能指标

- **BLEU分数**: 评估生成标题与参考标题的n-gram重叠
- **ROUGE分数**: 评估召回率导向的文本生成质量
- **CIDEr分数**: 专为图像描述设计的评估指标
- **SPICE分数**: 基于语义命题的图像描述评估

## 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目基于MIT许可证 - 详情请参阅LICENSE文件

## 致谢

- COCO数据集提供的图像和描述数据
- PyTorch团队的优秀深度学习框架
- 开源社区的图像标题生成实现