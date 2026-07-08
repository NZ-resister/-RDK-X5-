# DeepSeek API 接入说明

本项目使用 DeepSeek 云端大模型完成两类任务：

1. 根据识别到的物体和目标语言生成简单例句。
2. 根据发音评分与音素/音节状态生成可执行的发音改进建议。

## 1. 准备 API Key

1. 登录 DeepSeek 开放平台并创建 API Key。
2. 不要把 API Key 写进代码或提交到 GitHub。
3. 在运行环境中设置环境变量 `DEEPSEEK_API_KEY`。

Linux / 开发板：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

如果希望长期生效，可以写入系统环境变量，或者使用 `.env` 文件并在启动脚本中加载。仓库只提供 `.env.example`，真实 `.env` 应被 `.gitignore` 忽略。

## 2. 当前项目中的调用位置

主程序 `app.py` 中包含两个线程类：

- `DeepSeekExampleThread`：调用 `https://api.deepseek.com/chat/completions`，模型为 `deepseek-chat`，要求返回 JSON，字段为 `example` 和 `translation`。
- `DeepSeekAdviceThread`：调用同一接口，输入总分、音素/音节状态和本地评分反馈，返回 1 到 3 条中文发音建议。

代码通过以下方式读取密钥：

```python
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
```

如果没有设置密钥，程序会回退到本地例句或本地评分建议，不会崩溃。

## 3. 请求格式

DeepSeek Chat Completions 的最小请求结构如下：

```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "你是语言学习助手。"},
    {"role": "user", "content": "请为 apple 生成一句简单英文例句，并给出中文翻译。"}
  ],
  "temperature": 0.7,
  "max_tokens": 180
}
```

请求头需要包含：

```text
Authorization: Bearer ${DEEPSEEK_API_KEY}
Content-Type: application/json
```

## 4. 本地测试

可以运行本文件夹中的 `deepseek_client_example.py` 验证 API Key 是否可用：

```bash
python DeepSeek_API/deepseek_client_example.py
```

成功时会输出一个简单例句 JSON。失败时优先检查：

- `DEEPSEEK_API_KEY` 是否已经设置。
- 开发板或电脑是否能访问外网。
- API Key 是否仍有效、额度是否充足。
- 系统时间是否正确。

## 5. GitHub 发布注意事项

- 不提交 `.env`。
- 不在 README、代码、截图中暴露真实 API Key。
- 建议在 README 中说明云端能力是可选功能：无 Key 时可使用本地回退逻辑。
- 如果面向公开用户，建议补充网络异常、超时、额度不足时的用户提示。

