# Базовый образ с поддержкой CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04


# Установка переменных окружения для подавления интерактивных запросов
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Установка всех системных зависимостей в одном слое
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    git \
    htop \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    nvtop \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

# Установка PyTorch с поддержкой CUDA 12.1
RUN pip3 install -r requirements.txt

# Установка Jupyter и дополнительных библиотек
RUN pip3 install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    seaborn \
    glances \
    ray

# Настройка рабочей директории
WORKDIR /workspace

# Открытие портов
EXPOSE 8888 61209 8265

# Настройка Jupyter Notebook
CMD ["ray", "start", "--head", "--include-dashboard=true", "--dashboard-host=0.0.0.0", "--num-gpus=1"]
