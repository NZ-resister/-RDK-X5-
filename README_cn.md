# LXNND 多模态智慧语言学习助手

LXNND 是一个面向嵌入式 AI 竞赛项目的多模态语言学习系统。系统将摄像头物体识别、多语言单词展示、Edge TTS 标准发音、麦克风跟读录音、发音评分、波形回放和 DeepSeek 云端大模型建议整合到一个 PySide6 图形界面中。

## 主要功能

- 摄像头实时预览与 RF-DETR 目标检测。
- 点击识别框选择学习物体。
- 支持英语、俄语、法语单词展示，翻译统一显示中文。
- Edge TTS 发音示范，并带本地音频缓存。
- 麦克风按住跟读录音。
- 本地 `Whisper` 模块输出发音评分。
- 单词/音素块按绿、黄、红三色显示发音状态。
- DeepSeek 云端大模型生成例句和发音改进建议。
- 波形图展示标准音频和用户录音，点击波形可回放录音。
- 摄像头曝光滑杆，便于开发板现场调试。

## 项目结构

```text
LXNND/
├── app.py                  # 主程序入口
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── DeepSeek_API/           # DeepSeek API 接入说明和测试示例
├── Edge TTS/               # TTS 辅助代码
├── RF-DETR/                # 目标检测模型和说明
└── Whisper/                # 发音识别与评分模块
```

## 环境要求

建议环境：

- Python 3.10 或以上
- 摄像头与麦克风
- Linux 开发板或 Windows 开发环境
- Linux 下建议安装 `mpg123` 用于播放 TTS 音频
- 使用 DeepSeek 和在线 TTS 时需要网络

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

Linux 下如需安装系统音频工具：

```bash
sudo apt-get install mpg123 portaudio19-dev
```

## 环境变量

不要把真实密钥写入代码或提交到 GitHub。可参考 `.env.example` 设置环境变量。

DeepSeek 云端大模型功能需要：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

百度语音备用识别为可选项：

```bash
export BAIDU_SPEECH_APP_ID="your_app_id"
export BAIDU_SPEECH_API_KEY="your_api_key"
export BAIDU_SPEECH_SECRET_KEY="your_secret_key"
```

如果没有设置 `DEEPSEEK_API_KEY`，程序会回退到本地例句或本地发音建议。

## 运行方式

在项目根目录执行：

```bash
python app.py
```

主程序默认从以下位置加载 RF-DETR ONNX 模型：

```text
RF-DETR/rf_detr_nano.onnx
```

## 模块说明

### `app.py`

主界面和业务流程入口，负责摄像头、检测结果展示、语言选择、TTS、录音、评分和 DeepSeek 调用。

### `RF-DETR/`

目标检测模块，保存 ONNX 模型、BPU 模型和训练/导出说明。详细内容见 `RF-DETR/readme.cn.md`。

### `Whisper/`

发音评分模块，包含音频预处理、ASR 调用、音素块评分和诊断建议生成逻辑。

### `DeepSeek_API/`

DeepSeek API 的配置说明和最小测试脚本。详细内容见 `DeepSeek_API/readme.md`。

## GitHub 发布前检查

建议公开发布前补齐或处理：

- 删除 `__pycache__/`、`audio_cache/`、`user_audio/` 等生成文件。
- 不提交 `.env` 和任何真实 API Key。
- 大模型文件如 `.onnx`、`.bin` 体积较大，建议使用 Git LFS 或提供下载链接。
- 增加 `LICENSE` 文件，明确开源协议。
- 增加界面截图、演示视频或运行效果图。
- 说明测试过的开发板型号、系统版本、Python 版本和外设要求。
- 如果依赖专用硬件或 BPU 工具链，需要补充安装和部署步骤。

## 当前注意事项

- 在线 TTS 和 DeepSeek API 受网络影响，建议比赛或演示前预热缓存。
- 发音评分受麦克风、环境噪声和 ASR 结果影响，现场测试时建议保持麦克风距离稳定。
- `Whisper/recognizer.py` 中的百度语音参数已改为环境变量读取，发布时不要恢复硬编码密钥。

## 许可证

当前项目尚未声明开源许可证。正式发布到 GitHub 前建议添加 `LICENSE` 文件。

