import sqlite3
import os
import sys

print("="*70)
print("从词汇表中删除停用词")
print("="*70)

# 设置路径
base_dir = 'coco_image_title_data'
db_path = os.path.join(base_dir, 'image_title_database.db')

# 定义要删除的停用词列表
# 包括: 冠词，常见介词，表示数量的词等
stopwords_to_remove = [
    # 冠词
    "a", "an", "the",
    
    # 常见介词
    "at", "in", "on", "of", "for", "to", "with", "by", "from", "about","up","upon",
    "s",
    # 表示数量的词
    "one", "two", "three", "four", "five", "six","seven",
    
    # 其他常见停用词
    "and", "is", "are", "it", "its", "this", "that", "these", "those",
    "there", "here", "some", "many", "as", "be", "been", "has", "have", 
    "was", "were", "will", "would", "could", "should","who","whose",
]

# 连接到数据库
print(f"连接到数据库: {db_path}")
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
except sqlite3.Error as e:
    print(f"连接数据库时出错: {e}")
    sys.exit(1)

# 检查词汇表是否存在
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vocabulary'")
if cursor.fetchone() is None:
    print("错误: 数据库中不存在词汇表。请先运行词汇表创建脚本。")
    conn.close()
    sys.exit(1)

# 获取当前词汇表大小
cursor.execute("SELECT COUNT(*) FROM vocabulary WHERE special_token = 0")
initial_count = cursor.fetchone()[0]
print(f"初始普通词汇数量: {initial_count}")

# 获取要删除的停用词在数据库中的实际存在情况
print("\n要删除的停用词:")
words_to_delete = []
for word in stopwords_to_remove:
    cursor.execute("SELECT id, frequency FROM vocabulary WHERE word = ? AND special_token = 0", (word,))
    result = cursor.fetchone()
    if result:
        word_id, frequency = result
        words_to_delete.append((word_id, word, frequency))
        status = f"存在(频率:{frequency})"
    else:
        status = "不存在"
    print(f"- {word}: {status}")

if not words_to_delete:
    print("\n词汇表中未找到需要删除的停用词。")
    conn.close()
    sys.exit(0)

print(f"\n找到 {len(words_to_delete)} 个要删除的停用词")

# 创建事务以确保操作的原子性
try:
    # 开始事务
    conn.execute("BEGIN TRANSACTION")
    
    # 从词汇表中删除停用词
    print("\n从词汇表中删除停用词...")
    for word_id, word, _ in words_to_delete:
        # 首先删除word_indices表中的引用
        cursor.execute("DELETE FROM word_indices WHERE word_id = ?", (word_id,))
        # 然后删除vocabulary表中的记录
        cursor.execute("DELETE FROM vocabulary WHERE id = ?", (word_id,))
        print(f"已删除: {word}")
    
    # 提交事务
    conn.commit()
    print("所有删除操作已成功提交")
    
except sqlite3.Error as e:
    # 出错时回滚事务
    conn.rollback()
    print(f"删除操作出错，已回滚: {e}")
    conn.close()
    sys.exit(1)

# 重新计算排名
print("\n重新计算词汇排名...")
try:
    # 获取所有非特殊标记词汇，按频率排序
    cursor.execute("""
    SELECT id, frequency 
    FROM vocabulary 
    WHERE special_token = 0 
    ORDER BY frequency DESC, word
    """)
    
    words = cursor.fetchall()
    
    # 更新排名
    for rank, (word_id, _) in enumerate(words, 1):  # 排名从1开始
        cursor.execute("UPDATE vocabulary SET rank = ? WHERE id = ?", (rank, word_id))
    
    conn.commit()
    print(f"已更新 {len(words)} 个词汇的排名")
    
except sqlite3.Error as e:
    conn.rollback()
    print(f"更新排名时出错: {e}")

# 验证结果
cursor.execute("SELECT COUNT(*) FROM vocabulary WHERE special_token = 0")
final_count = cursor.fetchone()[0]

print("\n结果摘要:")
print("-" * 50)
print(f"初始词汇数量: {initial_count}")
print(f"删除的停用词数量: {len(words_to_delete)}")
print(f"最终词汇数量: {final_count}")
print(f"验证: {initial_count} - {len(words_to_delete)} = {initial_count - len(words_to_delete)}")
if final_count == initial_count - len(words_to_delete):
    print("验证成功: 词汇数量匹配")
else:
    print("验证失败: 词汇数量不匹配，请检查数据库")

# 显示当前词汇表中的前10个词
cursor.execute('''
SELECT word, frequency, rank, percentage
FROM vocabulary
WHERE special_token = 0
ORDER BY rank
LIMIT 10
''')

print("\n处理后词汇表前10个词:")
print("-" * 60)
print("词 | 频率 | 排名 | 百分比(%)")
print("-" * 60)

for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]:.4f}")

# 关闭数据库连接
conn.close()

print("完成! 已从词汇表中删除指定停用词")
print(f"数据库路径: {db_path}")

