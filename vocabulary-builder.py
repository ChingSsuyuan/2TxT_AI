import sqlite3
import os
import re
import string
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import nltk

# 确保所有需要的nltk数据都已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载nltk punkt分词器...")
    nltk.download('punkt')

# 额外检查和下载punkt_tab资源
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    print("下载nltk punkt_tab资源...")
    nltk.download('punkt_tab')

print("="*70)
print("构建图像标题词汇表")
print("="*70)

# 设置路径
base_dir = 'coco_image_title_data'
db_path = os.path.join(base_dir, 'image_title_database.db')
vocab_dir = os.path.join(base_dir, 'vocabulary')
os.makedirs(vocab_dir, exist_ok=True)

# 连接到数据库
print(f"连接到数据库: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取所有标题
print("获取所有标题数据...")
cursor.execute("SELECT caption FROM captions")
captions = [row[0] for row in cursor.fetchall()]
print(f"获取到 {len(captions)} 条标题")

# 标题预处理和分词函数
def preprocess_caption(caption):
    """预处理标题：转换为小写，移除标点符号"""
    # 转换为小写
    caption = caption.lower()
    # 移除标点符号
    caption = re.sub(f'[{string.punctuation}]', ' ', caption)
    # 移除多余空格
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption

# 简化的分词函数，避免使用可能缺少的NLTK资源
def tokenize_captions(captions):
    """对标题进行分词 - 使用简单的空格分割而不是NLTK"""
    all_tokens = []
    processed_captions = []
    
    for caption in captions:
        processed = preprocess_caption(caption)
        processed_captions.append(processed)
        
        # 使用简单的空格分割作为备选方案
        try:
            # 首先尝试NLTK的word_tokenize
            tokens = nltk.word_tokenize(processed)
        except LookupError:
            # 如果失败，使用简单的空格分割
            print("警告: 使用简单空格分词作为替代")
            tokens = processed.split()
            
        all_tokens.extend(tokens)
    
    return all_tokens, processed_captions

print("对标题进行预处理和分词...")
all_tokens, processed_captions = tokenize_captions(captions)
print(f"分词后得到 {len(all_tokens)} 个词")

# 统计词频
print("统计词频...")
word_counts = Counter(all_tokens)
print(f"词汇表大小（去重后）: {len(word_counts)}")

# 将词汇表按频率排序
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 保存词汇表
vocab_file = os.path.join(vocab_dir, 'vocabulary.json')
print(f"保存词汇表到: {vocab_file}")

vocabulary = {
    'word_counts': dict(sorted_word_counts),
    'total_words': len(all_tokens),
    'unique_words': len(word_counts)
}

with open(vocab_file, 'w', encoding='utf-8') as f:
    json.dump(vocabulary, f, indent=2, ensure_ascii=False)

# 生成词频统计结果
print("\n词频统计结果:")
print("-" * 50)
print("Top 20 最频繁词:")
for word, count in sorted_word_counts[:20]:
    print(f"{word}: {count} 次 ({count/len(all_tokens)*100:.2f}%)")

# 创建词汇索引表
print("\n创建索引表...")
# 添加特殊标记
special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}

# 使用频率过滤词汇（出现次数>=2的词）
min_word_count = 2
filtered_words = [word for word, count in word_counts.items() if count >= min_word_count]
print(f"过滤后词汇量 (频率 >= {min_word_count}): {len(filtered_words)}")

# 将过滤后的词添加到索引表
for word in filtered_words:
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)

idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 保存索引表
index_file = os.path.join(vocab_dir, 'word_indices.json')
print(f"保存词汇索引表到: {index_file}")

indices = {
    'word_to_idx': word_to_idx,
    'idx_to_word': {str(k): v for k, v in idx_to_word.items()},  # JSON的key必须是字符串
    'vocab_size': len(word_to_idx)
}

with open(index_file, 'w', encoding='utf-8') as f:
    json.dump(indices, f, indent=2, ensure_ascii=False)

# 保存处理后的标题
processed_captions_file = os.path.join(vocab_dir, 'processed_captions.json')
print(f"保存处理后的标题到: {processed_captions_file}")

# 获取图片ID和标题的对应关系
cursor.execute("""
SELECT i.coco_id, c.caption 
FROM images i
JOIN captions c ON i.id = c.image_id
""")
image_captions = {}
for coco_id, caption in cursor.fetchall():
    coco_id_str = str(coco_id)  # 确保键是字符串，避免JSON序列化问题
    if coco_id_str not in image_captions:
        image_captions[coco_id_str] = []
    image_captions[coco_id_str].append(preprocess_caption(caption))

with open(processed_captions_file, 'w', encoding='utf-8') as f:
    json.dump(image_captions, f, indent=2, ensure_ascii=False)

# 可视化词频分布
print("\n生成词频分布可视化...")
# 获取前50个频繁词和它们的频率（如果有那么多的话）
top_count = min(50, len(sorted_word_counts))
top_words = [word for word, _ in sorted_word_counts[:top_count]]
top_counts = [count for _, count in sorted_word_counts[:top_count]]

plt.figure(figsize=(15, 8))
plt.bar(range(len(top_words)), top_counts, color='skyblue')
plt.xticks(range(len(top_words)), top_words, rotation=90)
plt.title('Top 词频分布')
plt.xlabel('词')
plt.ylabel('频率')
plt.tight_layout()
plt.savefig(os.path.join(vocab_dir, 'word_frequency.png'))

# 计算词频分布统计
freq_dist = np.array([count for _, count in sorted_word_counts])
# 确保有足够的数据点进行百分位计算
if len(freq_dist) >= 4:  # 至少需要几个数据点
    percentiles = np.percentile(freq_dist, [25, 50, 75, 90, 95, 99])
    
    print("\n词频分布统计:")
    print(f"最低词频: {freq_dist.min()}")
    print(f"最高词频: {freq_dist.max()}")
    print(f"平均词频: {freq_dist.mean():.2f}")
    print(f"中位数词频: {np.median(freq_dist):.2f}")
    print(f"25% 分位数: {percentiles[0]}")
    print(f"50% 分位数: {percentiles[1]}")
    print(f"75% 分位数: {percentiles[2]}")
    print(f"90% 分位数: {percentiles[3]}")
    print(f"95% 分位数: {percentiles[4]}")
    print(f"99% 分位数: {percentiles[5]}")
else:
    print("\n词频分布统计 (简化版):")
    print(f"最低词频: {freq_dist.min()}")
    print(f"最高词频: {freq_dist.max()}")
    print(f"平均词频: {freq_dist.mean():.2f}")
    if len(freq_dist) > 0:
        print(f"中位数词频: {np.median(freq_dist):.2f}")

# 绘制长尾分布
plt.figure(figsize=(12, 6))
plt.plot(range(len(freq_dist)), np.sort(freq_dist)[::-1], color='blue')
plt.title('词频长尾分布')
plt.xlabel('词的排名')
plt.ylabel('频率')
if len(freq_dist) > 1 and freq_dist.max() > 1:  # 确保有足够的数据使用对数尺度
    plt.yscale('log')  # 使用对数尺度更好地展示长尾
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(vocab_dir, 'frequency_tail.png'))

# 创建标题长度统计 - 使用简单的空格分割而不是nltk，确保兼容性
def simple_word_count(text):
    """简单的词数统计函数"""
    return len(text.split())

caption_lengths = [simple_word_count(caption) for caption in processed_captions]
if caption_lengths:  # 确保有数据
    max_len = max(caption_lengths)
    min_len = min(caption_lengths)
    avg_len = sum(caption_lengths) / len(caption_lengths)
    
    plt.figure(figsize=(10, 6))
    bins = range(min_len, max_len + 2)
    if len(bins) > 1:  # 确保有足够的bins
        plt.hist(caption_lengths, bins=bins, color='green', alpha=0.7)
    else:
        plt.hist(caption_lengths, bins=10, color='green', alpha=0.7)  # 默认10个bins
    plt.title('标题词数分布')
    plt.xlabel('词数')
    plt.ylabel('标题数量')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vocab_dir, 'caption_length_dist.png'))
    
    print("\n标题长度统计:")
    print(f"最短标题: {min_len} 词")
    print(f"最长标题: {max_len} 词")
    print(f"平均长度: {avg_len:.2f} 词")

# 关闭数据库连接
conn.close()

print("词汇表构建完成!")
print(f"词汇表目录: {vocab_dir}")
