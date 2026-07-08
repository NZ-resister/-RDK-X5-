# RF-DETR 目标检测模型说明

本目录保存项目中用于物体识别的 RF-DETR 相关文件，包括训练脚本、导出脚本、ONNX 模型和地平线 BPU 编译产物。主程序通过摄像头采集图像，并调用 RF-DETR 模型识别常见学习物体，为后续多语言例句生成和发音跟读提供目标单词。

## 目录内容

```text
RF-DETR/
├── best.onnx             # 训练后导出的 ONNX 模型示例
├── rf_detr_nano.onnx     # 主程序使用的 ONNX 推理模型
├── rf_detr.bin           # 地平线 BPU 编译后的二进制模型
├── train.py              # RF-DETR 训练入口示例
├── export.py             # 模型导出脚本
├── bin.py                # BPU / bin 模型相关脚本
└── readme.cn.md          # 本说明文档
```

## 数据集格式

RF-DETR 训练推荐使用 COCO 格式数据集。目录结构如下：

```text
your_dataset/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── image_0001.jpg
│   └── ...
└── val2017/
    ├── image_0800.jpg
    └── ...
```

说明：

- `annotations/` 存放 COCO JSON 标注文件。
- `train2017/` 存放训练图片。
- `val2017/` 存放验证图片。
- COCO 标注中的 `bbox` 使用绝对像素坐标，格式为 `[x, y, width, height]`。

如果原始数据集是 YOLO 格式，可以先使用 Roboflow 转换为 COCO，也可以自行编写转换脚本。转换时需要特别注意类别编号是否从 0 或 1 开始，必须与训练配置保持一致。

## 训练模型

在 Ubuntu / CUDA / PyTorch 环境下安装 RF-DETR 后，可参考如下方式训练：

```python
from rfdetr import RFDETRNano


if __name__ == "__main__":
    model = RFDETRNano(pretrain_weights="/root/rf-detr-nano.pth")

    model.train(
        dataset_dir="/root/rf_detr_data",
        epochs=300,
        batch_size=64,
        grad_accum_steps=2,
        resolution=384,
        lr=1e-2,
        output_dir="runs/rf_detr_nano",
        device="cuda",
        use_ema=True,
    )
```

训练完成后通常会得到 `.pth` 权重文件，例如：

```text
runs/rf_detr_nano/checkpoint_best_total.pth
```

实际文件名以训练框架输出为准。

## 本地验证

导出或部署前，建议先在训练环境中用单张图片验证模型效果：

```python
from rfdetr import RFDETRNano


model = RFDETRNano(pretrain_weights="runs/rf_detr_nano/checkpoint_best_total.pth")
results = model.predict("test_image.jpg", threshold=0.5)
print(results)
```

确认类别、置信度和检测框基本正确后，再进入导出和部署阶段。

## 导出 ONNX

主程序和板端推理使用 ONNX 模型。导出时建议固定输入尺寸，避免动态 shape 在后续推理或 BPU 编译时出错。

示例命令：

```bash
python export.py \
  --checkpoint runs/rf_detr_nano/checkpoint_best_total.pth \
  --output rf_detr_nano.onnx \
  --input-size 384
```

注意事项：

- 输入尺寸要和 `app.py` 中视觉线程的 `input_size` 保持一致，目前为 `384 x 384`。
- 如果用于地平线 BPU 编译，建议使用静态输入尺寸。
- 如果工具链对 ONNX opset 有要求，优先使用兼容版本，例如 opset 11。

## 地平线 BPU 编译

如果需要在地平线开发板上使用 BPU 加速，需要将 ONNX 编译为 `.bin`。

典型流程：

1. 准备静态 shape 的 ONNX 模型。
2. 准备校准图片集。
3. 编写 `compile_config.yaml`。
4. 使用 OpenExplorer / `hb_mapper` 完成模型检查、量化和编译。

配置文件示例：

```yaml
model_parameters:
  onnx_model: "rf_detr_nano.onnx"
  output_model_file_prefix: "rf_detr"
  march: "bayes-e"

input_parameters:
  input_name: ""
  input_type_rt: "nv12"
  input_layout_rt: "NHWC"
  input_type_train: "rgb"
  input_layout_train: "NCHW"
  norm_type: "data_mean_and_scale"
  mean_value: 123.675 116.28 103.53
  scale_value: 0.01712475 0.017507 0.01742919

calibration_parameters:
  cal_data_dir: "./calibration_data"
  calibration_type: "max"

compiler_parameters:
  compile_mode: "latency"
  debug: false
```

具体字段需要根据开发板型号、OpenExplorer 版本和模型输入输出名称调整。

## 主程序集成

项目主程序 `app.py` 默认从以下路径加载 ONNX 模型：

```text
RF-DETR/rf_detr_nano.onnx
```

视觉线程完成以下步骤：

1. 从摄像头读取 BGR 图像。
2. 转为 RGB，并按 `384 x 384` 做 letterbox。
3. 调用 ONNX Runtime 推理。
4. 对输出框做置信度筛选和 NMS。
5. 将检测框映射回原图坐标。
6. 用户点击目标后，系统把类别标签传给语言学习模块。

如果更换模型，需要同步修改：

- `app.py` 中的类别表 `CLASSES`。
- `VisionThread.input_size`。
- ONNX 输出解析逻辑。
- 模型文件路径。

## 常见问题

### 1. ONNX 模型无法加载

检查文件是否存在：

```text
RF-DETR/rf_detr_nano.onnx
```

同时确认已安装：

```bash
pip install onnxruntime
```

### 2. 检测框位置偏移

通常是 letterbox 缩放比例、padding 或输入尺寸不一致导致。需要确认训练、导出和推理中的输入尺寸一致。

### 3. 检测类别错位

检查训练数据的类别编号、导出模型的类别顺序，以及 `app.py` 中 `CLASSES` 的映射是否一致。

### 4. BPU 编译失败

优先检查：

- ONNX 是否为静态 shape。
- opset 是否被工具链支持。
- 输入布局和颜色空间是否配置正确。
- 校准数据是否足够且路径正确。

## 发布建议

- 大模型权重和 `.onnx/.bin` 文件体积较大，公开到 GitHub 前建议使用 Git LFS，或在 README 中提供下载链接。
- 训练数据集通常不应直接提交，需要单独说明来源、格式和授权。
- 如果保留板端专用 `.bin`，请标注适用的芯片型号和工具链版本。

