FROM continuumio/miniconda3

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制环境文件和代码到容器中
COPY environment.yml .


# 创建 Conda 环境
RUN conda env create -f environment.yml

# 验证 jupyter 是否已安装
RUN conda run -n DL jupyter --version
