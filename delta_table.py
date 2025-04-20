import sqlite3
import os

# 数据库路径
db_path = "coco_image_title_data/image_title_database.db"

# 检查数据库是否存在
if not os.path.exists(db_path):
    print(f"❌ 数据库文件不存在: {db_path}")
else:
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 删除表
        cursor.execute("DROP TABLE IF EXISTS image_features_clip;")
        conn.commit()
        print("✅ 表 'image_features_clip' 已成功删除。")
    except sqlite3.Error as e:
        print("⚠️ 删除表时出错:", e)
    finally:
        conn.close()