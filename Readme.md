# 基于Diffusion模型的图像自动标题生成系统

这个项目使用预训练CNN提取图像特征，并通过Diffusion模型生成相关的图像标题。

## 项目概述

本系统通过以下步骤自动为图像生成高质量、相关性强的标题：
1. 使用预训练CNN提取图像特征
2. 将特征存储在数据库中
3. 使用Diffusion模型将图像特征转换为自然语言标题

## 系统架构

```
图像 → CNN特征提取 → 特征数据库 → Diffusion模型 → 自然语言标题
```

## 流程详解

### 1. 特征准备阶段

- **特征获取**：从数据库中读取CNN提取的图像特征（BLOB格式）
- **特征转换**：将二进制BLOB数据转换为张量格式
- **归一化处理**：标准化特征向量以确保稳定的训练和生成过程
- **数据集构建**：将图像特征与对应的标题文本配对，构建训练数据集

### 2. 文本处理阶段

- **标题预处理**：对现有标题进行清洗、分词、规范化
- **文本标记化**：将标题文本转换为标记序列
- **词汇表建立**：创建模型将使用的词汇表，包括特殊标记
- **文本表示**：将标记序列转换为嵌入向量或其他适合diffusion模型的表示形式

### 3. Diffusion模型设计

- **模型架构选择**：设计适合文本生成的diffusion模型架构
- **条件机制设计**：确定如何将图像特征作为条件注入diffusion过程
- **去噪网络定义**：通常使用Transformer或改进的U-Net架构
- **噪声调度设置**：配置适合文本生成的噪声添加与去除进度表

### 4. 训练流程

- **条件嵌入**：将图像特征嵌入到模型中作为条件信息
- **文本噪声处理**：对目标标题表示添加不同级别的噪声
- **迭代去噪训练**：模型学习从噪声中恢复原始标题表示
- **梯度累积**：处理可能的长序列或大批量数据
- **验证与监控**：定期评估模型生成标题的质量和相关性

### 5. 标题生成过程

- **特征输入**：将新图像的特征向量输入到训练好的模型
- **噪声初始化**：从随机噪声开始生成过程
- **条件引导**：使用图像特征引导去噪过程
- **逐步去噪**：执行预定义的去噪步骤数量
- **采样策略**：应用技术如DDIM采样以加速生成过程
- **序列转换**：将去噪后的表示转换回文本标记
- **后处理**：将标记序列解码为自然语言标题

### 6. 评估与优化

- **相关性评估**：评估生成标题与图像内容的相关度
- **质量评估**：检查标题的语法正确性、流畅度和信息量
- **人工评审**：获取人类评估者对标题质量的反馈
- **模型微调**：基于评估结果调整diffusion参数或训练策略
- **提示工程**：优化用于引导生成过程的特征表示方式

### 7. 部署与应用

- **模型导出**：将训练好的模型导出为适合部署的格式
- **流程整合**：将特征提取和标题生成流程整合为端到端系统
- **性能优化**：应用技术如蒸馏或量化以提高推理速度
- **批处理支持**：设计支持批量处理多个图像的机制
- **监控与更新**：建立机制监控系统性能并定期更新模型

## 数据库结构

项目使用SQLite数据库存储图像特征和相关信息：

### 表结构

1. **images**：存储图像元数据
   - id (主键)
   - coco_id
   - file_name
   - width/height
   - created_at
   - local_path

2. **captions**：存储图像描述
   - id (主键)
   - image_id (外键，关联到images表)
   - caption

3. **image_features_{model_name}**：存储图像特征
   - id (主键)
   - image_id (外键，关联到images表)
   - features (BLOB类型，存储特征向量)
   - created_at

## 技术栈

- **特征提取**：PyTorch, ResNet/ViT
- **数据存储**：SQLite
- **文本处理**：NLTK/spaCy
- **Diffusion模型**：基于PyTorch的自定义实现

## 安装与配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置数据路径
export IMAGE_DIR="coco_image_title_data/images"
export DB_PATH="coco_image_title_data/image_title_database.db"
```

## 使用方法

### 特征提取

```bash
python extract_features.py --model resnet50 --db-path $DB_PATH --image-dir $IMAGE_DIR
```

### 模型训练

```bash
python train_diffusion.py --features-table image_features_resnet50 --db-path $DB_PATH
```

### 标题生成

```bash
python generate_titles.py --model-path models/diffusion_title_generator.pt --image-path sample.jpg
```

## 性能指标

- **BLEU分数**: 衡量生成标题与参考标题的相似度
- **ROUGE分数**: 评估生成标题的召回率
- **CIDEr分数**: 特别适用于图像描述任务的评估指标
- **人工评分**: 由人类评估者评定的标题质量和相关性

## 示例输出

| 图像 | 生成标题 |
|------|---------|
| 厨房场景 | "两名厨师在专业金属厨房中准备食物" |
| 户外活动 | "一群人在湖边野餐的欢乐场景" |
| 办公环境 | "商务人士在会议室讨论项目计划" |

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
- Diffusion模型的开创性研究工作