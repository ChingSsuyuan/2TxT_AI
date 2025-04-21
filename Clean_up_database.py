import sqlite3
import os

# 数据库路径
db_path = 'coco_image_title_data/image_title_database.db'

# 检查文件是否存在
if not os.path.exists(db_path):
    print(f"数据库文件不存在：{db_path}")
else:
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有非系统表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    tables = cursor.fetchall()

    # 删除每一个表
    for table_name in tables:
        print(f"正在删除表：{table_name[0]}")
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name[0]}"')

    # 提交更改并关闭连接
    conn.commit()
    conn.close()
    print("所有用户表已删除（系统表除外）。")
