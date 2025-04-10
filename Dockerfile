# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置为中国国内源（针对 Bookworm/Debian 12）
# RUN rm -rf /etc/apt/sources.list.d/* && \
#     echo "deb http://mirrors.ustc.edu.cn/debian bookworm main" > /etc/apt/sources.list && \
#     echo "deb http://mirrors.ustc.edu.cn/debian bookworm-updates main" >> /etc/apt/sources.list && \
#     echo "deb http://mirrors.ustc.edu.cn/debian-security bookworm-security main" >> /etc/apt/sources.list

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 安装系统依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]