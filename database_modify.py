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
# 再次连接数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 4. 创建 Validation_Set 表（如果不存在）
cursor.execute('''
CREATE TABLE IF NOT EXISTS Validation_Set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    caption TEXT NOT NULL
)
''')

# 5. 获取 Training_Set 总数量
cursor.execute('SELECT COUNT(*) FROM Training_Set')
total_count = cursor.fetchone()[0]
val_count = int(total_count * 0.15)

# 6. 随机选择 15% 的数据作为验证集
cursor.execute(f'''
INSERT INTO Validation_Set (file_name, caption)
SELECT file_name, caption FROM Training_Set
ORDER BY RANDOM()
LIMIT {val_count}
''')

# 7. 从 Training_Set 中删除这些验证集数据（确保无重复）
cursor.execute('''
DELETE FROM Training_Set
WHERE id IN (
    SELECT id FROM Validation_Set
)
''')

# 提交更改并关闭连接
conn.commit()
conn.close()

print(f"✅ 验证集划分完成：总数 {total_count}，验证集 {val_count}，训练集 {total_count - val_count}")
