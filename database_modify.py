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
