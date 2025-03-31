import sqlite3
import os
import json
import sys

print("="*70)
print("将词汇表按频率存入数据库")
print("="*70)

# 设置路径
base_dir = 'coco_image_title_data'
db_path = os.path.join(base_dir, 'image_title_database.db')
vocab_dir = os.path.join(base_dir, 'vocabulary')
vocab_file = os.path.join(vocab_dir, 'vocabulary.json')

# 检查词汇表文件是否存在
if not os.path.exists(vocab_file):
    print(f"错误: 词汇表文件不存在: {vocab_file}")
    print("请先运行vocabulary-builder.py脚本生成词汇表")
    sys.exit(1)

# 读取词汇表
print(f"读取词汇表: {vocab_file}")
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocabulary = json.load(f)

# 提取词频数据
word_counts = vocabulary.get('word_counts', {})
total_words = vocabulary.get('total_words', 0)

# 将词汇按频率排序
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
print(f"从词汇表中加载了 {len(sorted_words)} 个唯一词汇")

# 连接到数据库
print(f"连接到数据库: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 定义特殊标记
special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']

# 检查vocabulary表是否存在，如果存在则先保存特殊标记信息
special_token_ids = {}
try:
    # 查询vocabulary表看是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vocabulary'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        print("检查vocabulary表中的特殊标记...")
        # 尝试查询特殊标记
        for token in special_tokens:
            cursor.execute("SELECT id FROM vocabulary WHERE word = ?", (token,))
            result = cursor.fetchone()
            if result:
                special_token_ids[token] = result[0]
                print(f"找到特殊标记: {token}, ID: {result[0]}")
    
except sqlite3.Error as e:
    print(f"查询表时出错: {e}")
    # 表不存在或有其他问题，继续创建新表

# 添加special_token列到vocabulary表（如果表已存在但没有此列）
try:
    # 检查vocabulary表是否存在special_token列
    cursor.execute("PRAGMA table_info(vocabulary)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'vocabulary' in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        if 'special_token' not in columns:
            print("向vocabulary表添加special_token列...")
            cursor.execute("ALTER TABLE vocabulary ADD COLUMN special_token BOOLEAN DEFAULT 0")
            conn.commit()
    
except sqlite3.Error as e:
    print(f"检查/修改表结构时出错: {e}")

# 重新创建词汇表
print("创建或更新词汇表结构...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS vocabulary (
    id INTEGER PRIMARY KEY,
    word TEXT UNIQUE,
    frequency INTEGER,
    rank INTEGER,
    percentage REAL,
    special_token BOOLEAN DEFAULT 0
)
''')

# 创建或更新索引表
cursor.execute('''
CREATE TABLE IF NOT EXISTS word_indices (
    word_id INTEGER,
    index_id INTEGER,
    special_token BOOLEAN DEFAULT 0,
    FOREIGN KEY (word_id) REFERENCES vocabulary (id),
    PRIMARY KEY (word_id, index_id)
)
''')

# 清空现有数据（保留ID）
if not special_token_ids:  # 如果没有找到特殊标记ID，才清空数据
    print("清空现有词汇表数据...")
    cursor.execute("DELETE FROM vocabulary")
    cursor.execute("DELETE FROM word_indices")
    conn.commit()

    # 将特殊标记添加到词汇表
    print("添加特殊标记到词汇表...")
    for i, token in enumerate(special_tokens):
        cursor.execute('''
        INSERT INTO vocabulary (word, frequency, rank, percentage, special_token)
        VALUES (?, ?, ?, ?, ?)
        ''', (token, 0, -i-1, 0.0, 1))  # 使用负数排名区分特殊标记
        
        # 获取插入的词ID
        word_id = cursor.lastrowid
        special_token_ids[token] = word_id
        
        # 将特殊标记添加到索引表
        cursor.execute('''
        INSERT INTO word_indices (word_id, index_id, special_token)
        VALUES (?, ?, ?)
        ''', (word_id, i, 1))
else:
    # 如果已经有特殊标记，更新它们的special_token字段
    print("更新现有特殊标记...")
    for token, token_id in special_token_ids.items():
        cursor.execute('''
        UPDATE vocabulary SET special_token = 1 WHERE id = ?
        ''', (token_id,))
    
    # 删除普通词汇（非特殊标记）
    print("删除现有普通词汇...")
    cursor.execute('''
    DELETE FROM vocabulary WHERE special_token = 0 OR special_token IS NULL
    ''')
    cursor.execute('''
    DELETE FROM word_indices WHERE word_id NOT IN (
        SELECT id FROM vocabulary WHERE special_token = 1
    )
    ''')

conn.commit()

# 将词汇表数据批量插入数据库（跳过特殊标记）
print("将词汇数据插入数据库...")
batch_size = 1000
# 过滤掉特殊标记
filtered_words = [(word, freq) for word, freq in sorted_words if word not in special_tokens]
total_batches = (len(filtered_words) + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(filtered_words))
    batch = filtered_words[start_idx:end_idx]
    
    vocab_data = []
    for rank, (word, freq) in enumerate(batch, start=start_idx+1):  # 排名从1开始
        percentage = (freq / total_words) * 100 if total_words > 0 else 0
        vocab_data.append((word, freq, rank, percentage, 0))  # 0表示不是特殊标记
    
    cursor.executemany('''
    INSERT INTO vocabulary (word, frequency, rank, percentage, special_token)
    VALUES (?, ?, ?, ?, ?)
    ''', vocab_data)
    
    conn.commit()
    print(f"批处理进度: {end_idx}/{len(filtered_words)} 个词汇已处理 ({(end_idx/len(filtered_words)*100):.1f}%)")

# 将索引信息添加到word_indices表
print("添加索引信息到word_indices表...")
# 从数据库中获取词汇ID映射
cursor.execute("SELECT id, word FROM vocabulary WHERE special_token = 0")
word_to_db_id = {row[1]: row[0] for row in cursor.fetchall()}

# 读取索引文件
index_file = os.path.join(vocab_dir, 'word_indices.json')
if os.path.exists(index_file):
    with open(index_file, 'r', encoding='utf-8') as f:
        indices_data = json.load(f)
        word_to_idx = indices_data.get('word_to_idx', {})
        
    # 过滤出非特殊标记（特殊标记已在前面添加）
    normal_indices = []
    for word, idx in word_to_idx.items():
        if word not in special_tokens and word in word_to_db_id:
            normal_indices.append((word_to_db_id[word], idx))
    
    # 批量插入索引数据
    if normal_indices:
        batch_size = 1000
        for i in range(0, len(normal_indices), batch_size):
            batch = normal_indices[i:i+batch_size]
            cursor.executemany('''
            INSERT INTO word_indices (word_id, index_id, special_token)
            VALUES (?, ?, 0)
            ''', batch)
            conn.commit()

# 创建视图以方便查询
print("创建便捷视图...")
cursor.execute('DROP VIEW IF EXISTS vocab_view')
cursor.execute('''
CREATE VIEW vocab_view AS
SELECT v.id, v.word, v.frequency, v.rank, v.percentage, 
       wi.index_id, CASE WHEN v.special_token = 1 THEN 'Yes' ELSE 'No' END as is_special
FROM vocabulary v
LEFT JOIN word_indices wi ON v.id = wi.word_id
ORDER BY CASE WHEN v.rank < 0 THEN 0 ELSE 1 END, v.rank
''')

# 验证数据库内容
print("\n验证数据库内容:")
print("-" * 50)

# 查询词汇数量
cursor.execute("SELECT COUNT(*) FROM vocabulary")
vocab_count = cursor.fetchone()[0]
print(f"数据库中的词汇总数量: {vocab_count}")

# 查询特殊标记数量
cursor.execute("SELECT COUNT(*) FROM vocabulary WHERE special_token = 1")
special_count = cursor.fetchone()[0]
print(f"特殊标记数量: {special_count}")

# 查询普通词汇数量
cursor.execute("SELECT COUNT(*) FROM vocabulary WHERE special_token = 0")
normal_count = cursor.fetchone()[0]
print(f"普通词汇数量: {normal_count}")

# 显示词汇表中的前10个词
cursor.execute('''
SELECT word, frequency, rank, percentage, special_token
FROM vocabulary
ORDER BY CASE WHEN rank < 0 THEN 0 ELSE 1 END, rank
LIMIT 10
''')

print("\n词汇表前10个词:")
print("-" * 70)
print("词 | 频率 | 排名 | 百分比(%) | 是否特殊标记")
print("-" * 70)

for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {'是' if row[4] else '否'}")

# 查询索引表
cursor.execute('''
SELECT v.word, wi.index_id, wi.special_token
FROM vocabulary v
JOIN word_indices wi ON v.id = wi.word_id
ORDER BY wi.index_id
LIMIT 10
''')

print("\n索引表前10项:")
print("-" * 50)
print("词 | 索引ID | 是否特殊标记")
print("-" * 50)

rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"{row[0]} | {row[1]} | {'是' if row[2] else '否'}")
else:
    print("索引表为空")

# 添加索引以提高性能
print("\n为词汇表添加索引以提高查询性能...")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_word ON vocabulary(word)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_rank ON vocabulary(rank)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_freq ON vocabulary(frequency)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_special ON vocabulary(special_token)")

# 关闭数据库连接
conn.commit()
conn.close()


print("完成! 词汇表已存入数据库。")
print(f"数据库路径: {db_path}")
print("=" * 50)
print("\n提示: 你现在可以使用SQL查询来分析词汇数据,例如:")
print("- 查询最常用的前N个词: SELECT word, frequency FROM vocabulary WHERE special_token = 0 ORDER BY rank LIMIT N")
print("- 查找特定词的排名: SELECT * FROM vocabulary WHERE word = '某个词'")
print("- 获取频率分布: SELECT frequency, COUNT(*) FROM vocabulary GROUP BY frequency ORDER BY frequency DESC")
print("=" * 50)