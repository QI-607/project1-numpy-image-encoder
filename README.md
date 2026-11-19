# Project 1：用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器

本项目仅用 **NumPy** 复现 OpenCLIP ViT-B-32 视觉编码器的前向计算：输入预处理后的图像，
输出与独立 PyTorch 参考实现数值一致的 512 维图像特征向量。模型参数由教师提供，无需训练。

## 环境依赖

- Python 3.9+
- 依赖见 `requirements.txt`（NumPy、Pillow、imageio）

```bash
python -m pip install -r requirements.txt
```

## 复现步骤

### 1. 验证实现（逐阶段与参考答案比对）

```bash
python check.py --stage all
```

会依次检查 8 个阶段：`assets / preprocess / patches / tokens / block0 / last_tokens / output / batch`，
每一关打印 PASS 与最大/平均绝对误差。也可只看某一关，例如 `python check.py --stage output`。

### 2. 生成输出 token

对题目自带的 10 张图：

```bash
python run.py --output output_tokens.npz
```

对任意图片：

```bash
python run.py --images 路径/a.jpg 路径/b.png --output my_tokens.npz
```

结果保存为 `.npz`，内含每张图的 id 与对应的 `[B,512]` 输出向量。

### 3. 查看结果

```python
import numpy as np
with np.load("output_tokens.npz", allow_pickle=False) as data:
    print(data["ids"])                 # 图片 id
    print(data["output_tokens"].shape) # (10, 512)
```

## 验证结果

`python check.py --stage all` 八阶段全部 PASS，关键阶段最大绝对误差：

| 阶段 | 形状 | 最大绝对误差 |
|---|---|---|
| patches | (2,49,3072) | 0.0 |
| tokens | (2,50,768) | 4.01e-05 |
| block0 | (2,50,768) | 7.06e-06 |
| last_tokens | (2,50,768) | 4.77e-05 |
| output | (10,512) | 5.44e-07 |
| batch | (3,512) | 4.62e-07 |

## 文件说明

| 文件 | 说明 |
|---|---|
| `image_encoder.py` | 核心实现（本人在此完成 TODO-01 ~ TODO-06） |
| `helpers.py` | 教师提供的 LayerNorm、GELU、softmax、单位化等工具 |
| `preprocess_image.py` | 教师提供的图像预处理（缩放、裁剪、归一化） |
| `runtime.py` / `run.py` / `check.py` | 教师提供的数据加载、命令行入口与校验脚本 |
| `weights/image_encoder.npz` | 预训练参数（152 个视觉张量，未随提交上传） |
| `assets/` | 10 张测试图与独立参考输出 |
| `report.tex` / `report.pdf` | 实验报告（LaTeX 源码与编译产物） |
| `ASSIGNMENT.md` | 教师原版英文作业说明（保留备查） |

## 报告

实验报告见 `report.pdf`（Overleaf / XeLaTeX 编译，含数据流图、红色高亮本人实现部分、
数值误差表、shape 变化说明与调试记录）。源码为 `report.tex`。
