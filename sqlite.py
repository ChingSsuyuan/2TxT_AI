import sqlite3
import csv
import os

def export_sqlite_to_csv(db_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # 对每个表进行导出
    for table in tables:
        table_name = table[0]
        print(f"正在导出表: {table_name}")
        
        # 获取表数据
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        # 获取列名
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        
        # 将数据写入CSV
        output_file = os.path.join(output_folder, f"{table_name}.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)  # 写入列名
            writer.writerows(rows)    # 写入数据
        
        print(f"表 {table_name} 已导出到 {output_file}")
    
    # 关闭连接
    conn.close()
    print("所有表导出完成")

# 使用示例
db_path = "coco_image_title_data/image_title_database.db"
output_folder = "exported_csv"
export_sqlite_to_csv(db_path, output_folder)