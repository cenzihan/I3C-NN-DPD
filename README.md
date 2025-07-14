# NN-DPD 项目文档

本项目旨在使用不同的深度学习模型，包括 **ARVTDNN** (全连接网络), **RVTDCNN** (卷积网络), 和 **RVTDSAN** (Transformer/自注意力网络)，来实现功率放大器 (PA) 的数字预失真 (DPD)。

---

## 目录结构

```
.
├── coefs/                # 存放训练好的模型权重 (.pth 文件)
├── data_xzr/             # 存放所有的 .mat 格式的原始数据
├── env/                  # 存放 Conda 环境配置文件和安装脚本
│   ├── environment.yml
│   └── setup_conda.sh
├── Predictions/          # 存放 DPD.py 推理后生成的预测结果 (.mat 文件)
├── ARVTDNN.py            # ARVTDNN 模型的训练脚本
├── RVTDCNN.py            # RVTDCNN 模型的训练脚本
├── RVTDSAN.py            # RVTDSAN 模型的训练脚本
├── DPD.py                # 加载预训练模型进行推理的脚本
├── enable_wandb.py       # 全局开关，用于启用或禁用 Weights & Biases
└── README.md             # 本文档
```

---

## 环境配置

本项目使用 Conda 来管理依赖环境。请按照以下步骤来创建和配置环境。

1.  **安装 Conda**
    如果您尚未安装 Conda，请先从 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载并安装。

2.  **创建 Conda 环境**
    我们提供了一个一键安装脚本来创建所需的环境。

    首先，进入项目根目录，为脚本添加执行权限：
    ```bash
    chmod +x env/setup_conda.sh
    ```

    然后，运行脚本来创建环境。脚本会自动创建一个名为 `nn-dpd-env` 的新环境。
    ```bash
    ./env/setup_conda.sh
    ```

3.  **激活环境**
    在运行任何代码之前，您必须激活此环境：
    ```bash
    conda activate nn-dpd-env
    ```
    当您看到终端提示符前出现 `(nn-dpd-env)` 时，表示环境已成功激活。

---

## 使用说明

请确保您已经激活了 `nn-dpd-env` 环境。

### 1. 配置 Weights & Biases (wandb)

在项目根目录有一个 `enable_wandb.py` 文件，用于控制如何集成 [Weights & Biases](https://wandb.ai/) 进行实验跟踪。打开此文件，修改 `WANDB_MODE` 变量来选择一种模式：

-   **`WANDB_MODE = "offline"`** (推荐/默认)
    此模式用于网络连接不稳定或受限的环境。它会将所有实验数据先保存在本地的 `wandb` 文件夹中，不会尝试实时上传。您可以在稍后网络良好的情况下手动同步数据。

-   **`WANDB_MODE = "online"`**
    此模式会将数据实时同步到 `wandb` 云端。需要稳定良好的网络连接。若选择此模式，首次使用时可能需要根据终端提示登录：
    ```bash
    wandb login
    ```

-   **`WANDB_MODE = "disabled"`**
    完全禁用 `wandb` 功能。程序将只在本地运行，不记录任何 `wandb` 相关数据。

### 2. 训练模型

您可以从多个训练脚本中选择一个来开始训练。例如，要训练 `RVTDSAN` 模型：

```bash
python RVTDSAN.py
```

训练过程中的损失等信息会打印在终端上。训练完成后，模型权重会自动保存在 `./coefs/` 目录下。

### 3. 进行推理

当您拥有一个训练好的模型后，可以使用 `DPD.py` 脚本来加载它并对测试数据进行预测。

1.  **修改 `DPD.py`**：打开该文件，根据您训练好的模型和想测试的数据集，修改文件顶部的 `Modelname`, `Dataname`, 和 `PAname` 变量。
2.  **运行推理脚本**：
    ```bash
    python DPD.py
    ```
    预测结果将会被保存为 `.mat` 文件到 `./Predictions/` 目录下。

### 4. (可选) 自动化评估

为了获得更可靠的模型性能，您可以使用 `run2.sh` 脚本来多次运行同一个训练脚本并计算性能指标的平均值。

1.  **修改 `run2.sh`**：打开该文件，修改 `python_file` 变量为您想要测试的脚本名称。
2.  **运行脚本**：
    ```bash
    bash run2.sh
    ```

### 5. 同步离线数据

如果您在 `offline` 模式下运行了实验，可以随时使用以下命令将本地保存的所有数据同步到云端：
```bash
wandb sync --sync-all
```

---
---

# NN-DPD Project Documentation

This project aims to implement Digital Pre-Distortion (DPD) for Power Amplifiers (PA) using various deep learning models, including **ARVTDNN** (Fully-Connected Network), **RVTDCNN** (Convolutional Network), and **RVTDSAN** (Transformer/Self-Attention Network).

---

## Directory Structure

```
.
├── coefs/                # Stores trained model weights (.pth files)
├── data_xzr/             # Stores all raw data in .mat format
├── env/                  # Contains Conda environment configuration and setup script
│   ├── environment.yml
│   └── setup_conda.sh
├── Predictions/          # Stores prediction results from DPD.py inference (.mat files)
├── ARVTDNN.py            # Training script for the ARVTDNN model
├── RVTDCNN.py            # Training script for the RVTDCNN model
├── RVTDSAN.py            # Training script for the RVTDSAN model
├── DPD.py                # Script to load a pre-trained model for inference
├── enable_wandb.py       # Global switch to enable or disable Weights & Biases
└── README.md             # This document
```

---

## Environment Setup

This project uses Conda to manage its dependencies. Please follow the steps below to set up the environment.

1.  **Install Conda**
    If you don't have Conda, please download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

2.  **Create Conda Environment**
    A setup script is provided for one-click environment creation.

    First, navigate to the project root and make the script executable:
    ```bash
    chmod +x env/setup_conda.sh
    ```

    Next, run the script. It will automatically create a new environment named `nn-dpd-env`.
    ```bash
    ./env/setup_conda.sh
    ```

3.  **Activate Environment**
    Before running any code, you must activate the environment:
    ```bash
    conda activate nn-dpd-env
    ```
    When you see `(nn-dpd-env)` prepended to your shell prompt, the environment is active.

---

## How to Run

Please ensure you have activated the `nn-dpd-env` environment.

### 1. Configure Weights & Biases (wandb)

There is a central configuration file `enable_wandb.py` in the project root to control experiment tracking with [Weights & Biases](https://wandb.ai/). Open this file and set the `WANDB_MODE` variable to one of the following modes:

-   **`WANDB_MODE = "offline"`** (Recommended/Default)
    Use this mode in environments with unstable or restricted internet access. It saves all experiment data locally to a `wandb` directory without attempting to upload it in real-time. You can sync the data manually later when you have a stable connection.

-   **`WANDB_MODE = "online"`**
    This mode syncs data to the `wandb` cloud in real-time and requires a stable internet connection. If you choose this mode, you may need to log in via the terminal on your first use:
    ```bash
    wandb login
    ```

-   **`WANDB_MODE = "disabled"`**
    This completely disables `wandb` integration. The scripts will run locally without logging any `wandb`-related data.

### 2. Train a Model

You can choose from several training scripts to start. For example, to train the `RVTDSAN` model:

```bash
python RVTDSAN.py
```

Training progress, such as loss, will be printed to the console. Upon completion, the model weights will be automatically saved to the `./coefs/` directory.

### 3. Run Inference

Once you have a trained model, you can use the `DPD.py` script to load it and make predictions on test data.

1.  **Modify `DPD.py`**: Open the file and update the `Modelname`, `Dataname`, and `PAname` variables at the top to match your trained model and the desired test dataset.
2.  **Run the inference script**:
    ```bash
    python DPD.py
    ```
    The prediction results will be saved as a `.mat` file in the `./Predictions/` directory.

### 4. (Optional) Automated Evaluation

For a more robust performance evaluation, you can use the `run2.sh` script to run a training script multiple times and average the performance metrics.

1.  **Modify `run2.sh`**: Open the file and set the `python_file` variable to the name of the script you want to test.
2.  **Run the script**:
    ```bash
    bash run2.sh
    ```

### 5. Sync Offline Data

If you have run experiments in `offline` mode, you can upload all locally saved data to the cloud at any time with the following command:
```bash
wandb sync --sync-all
```

---
--- 