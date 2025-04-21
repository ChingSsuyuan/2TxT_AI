import sqlite3


# 数据库路径
db_path = "coco_image_title_data/image_title_database.db"

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. 创建新表 Training_Set（如果不存在）
cursor.execute('''
CREATE TABLE IF NOT EXISTS Training_Set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    caption TEXT NOT NULL
)
''')

# 2. 清空已有数据（可选：防止重复插入）
cursor.execute('DELETE FROM Training_Set')

# 3. 插入 file_name 和 caption 对应数据
cursor.execute('''
INSERT INTO Training_Set (file_name, caption)
SELECT images.file_name, captions.caption
FROM images
JOIN captions ON images.id = captions.image_id
''')

# 提交更改并关闭连接
conn.commit()
conn.close()

print("✅ 已成功创建并填充 Training_Set 表。")
# 数据库路径
db_path = "coco_image_title_data/image_title_database.db"

# 连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建新表 Validation_Set 和 Test_Set（如果不存在）
cursor.execute('''
CREATE TABLE IF NOT EXISTS Validation_Set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    caption TEXT NOT NULL
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS Test_Set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    caption TEXT NOT NULL
)
''')

# 获取原始 Training_Set 总数
cursor.execute('SELECT COUNT(*) FROM Training_Set')
total_count = cursor.fetchone()[0]

# 计算划分数量
val_count = int(total_count * 0.10)
test_count = int(total_count * 0.05)
val_test_total = val_count + test_count

# 随机抽取 15% 的样本（用于 validation + test）
cursor.execute(f'''
SELECT id, file_name, caption FROM Training_Set
ORDER BY RANDOM()
LIMIT {val_test_total}
''')
val_test_samples = cursor.fetchall()

# 分配前 val_count 行给 Validation_Set，后 test_count 行给 Test_Set
val_samples = val_test_samples[:val_count]
test_samples = val_test_samples[val_count:]

# 插入 Validation_Set
cursor.executemany('''
INSERT INTO Validation_Set (file_name, caption)
VALUES (?, ?)
''', [(row[1], row[2]) for row in val_samples])

# 插入 Test_Set
cursor.executemany('''
INSERT INTO Test_Set (file_name, caption)
VALUES (?, ?)
''', [(row[1], row[2]) for row in test_samples])

# 删除从 Training_Set 中抽取的 val+test 样本
ids_to_delete = [str(row[0]) for row in val_test_samples]
id_list_str = ",".join(ids_to_delete)

cursor.execute(f'''
DELETE FROM Training_Set
WHERE id IN ({id_list_str})
''')

# 提交并关闭
conn.commit()
conn.close()

print(f"✅ 数据划分完成：训练集 {total_count - val_test_total}，验证集 {val_count}，测试集 {test_count}")
