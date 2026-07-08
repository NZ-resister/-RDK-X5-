# Independent and Experiential Language Learner

Independent and Experiential Language Learner is a multimodal language-learning assistant designed for an embedded AI competition project. It combines camera-based object recognition, multilingual word display, Edge TTS pronunciation demonstration, microphone recording, pronunciation scoring, waveform replay, and DeepSeek-powered example/advice generation.

## Features

- Real-time camera preview and object detection with RF-DETR.
- Click-to-select detected objects and start a language-learning workflow.
- English, Russian, and French word display with Chinese translation.
- Edge TTS pronunciation demonstration with local audio cache.
- Microphone recording for pronunciation practice.
- Pronunciation scoring through the local `Whisper` module.
- Visual word/phoneme feedback with green, yellow, and red highlighting.
- DeepSeek cloud LLM integration for example sentences and pronunciation advice.
- Waveform display and click-to-play user recording.
- Camera exposure control slider for embedded-board testing.

## Project Structure

```text
LXNND/
├── app.py                  # Main PySide6 desktop/board application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── DeepSeek_API/           # DeepSeek setup guide and test client
├── Edge TTS/               # TTS helper code
├── RF-DETR/                # Object detection model files and notes
└── Whisper/                # Pronunciation scoring module
```

## Requirements

Recommended environment:

- Python 3.10+
- Linux or Windows for development
- Camera and microphone
- `mpg123` command-line player for TTS playback on Linux
- Network access for DeepSeek and online TTS generation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On Linux, install system audio tools if needed:

```bash
sudo apt-get install mpg123 portaudio19-dev
```

## Environment Variables

Copy `.env.example` and set real credentials in your environment. Do not commit real keys.

Required for cloud LLM features:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

Optional Baidu ASR fallback:

```bash
export BAIDU_SPEECH_APP_ID="your_app_id"
export BAIDU_SPEECH_API_KEY="your_api_key"
export BAIDU_SPEECH_SECRET_KEY="your_secret_key"
```

If `DEEPSEEK_API_KEY` is not configured, the app falls back to local examples or local scoring advice.

## Run

From the project root:

```bash
python app.py
```

The RF-DETR ONNX model is expected at:

```text
RF-DETR/rf_detr_nano.onnx
```

## Model Notes

- Object detection uses RF-DETR ONNX through ONNX Runtime.
- Pronunciation scoring is implemented in `Whisper/recognizer.py`.
- Edge TTS audio files are cached under `audio_cache/`.
- User recordings are stored under `user_audio/`.

## GitHub Release Checklist

Before publishing publicly:

- Remove generated caches such as `__pycache__/`, `audio_cache/`, and `user_audio/`.
- Do not commit `.env` or any real API keys.
- Consider Git LFS for large model files such as `.onnx` and `.bin`.
- Add a license file if this project will be open-source.
- Add screenshots or a short demo video.
- Document tested hardware, OS, Python version, and board model.

## License

No license has been declared yet. Add a `LICENSE` file before public release.

