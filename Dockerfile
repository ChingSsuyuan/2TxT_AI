# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装PyTorch和其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码
COPY *.py /app/
COPY model_output/best_model.pth /app/model/
COPY coco_image_title_data/image_title_database.db /app/data/

# 添加示例图片目录
RUN mkdir -p /app/images

# 设置环境变量
ENV MODEL_PATH=/app/model/best_model.pth
ENV DB_PATH=/app/data/image_title_database.db
ENV PYTHONPATH=/app

# 暴露API端口
EXPOSE 5000

# 启动API服务器
CMD ["python", "api_server.py"]