# Whisper/audio_utils.py
# Audio preprocessing for Whisper: convert to 16kHz mono WAV

import os
import wave
import struct


def prepare_audio(input_path, output_path=None):
    """
    Convert audio to Whisper standard format: 16kHz, mono, PCM_16.

    Args:
        input_path: Path to input audio file (wav, mp3, m4a, etc.)
        output_path: Optional output path. Defaults to input's dir + _16k.wav

    Returns:
        Path to the processed WAV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到音频文件: {input_path}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_16k.wav"

    # Use soundfile (supports wav/mp3/m4a/ogg/flac automatically)
    try:
        import soundfile as sf
        import numpy as np
        y, sr = sf.read(input_path, dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(output_path, y, 16000, subtype='PCM_16')
        return output_path
    except ImportError:
        pass

    # Fallback: pure stdlib wave module (wav files only)
    return _convert_wav_pcm(input_path, output_path)


def _convert_wav_pcm(input_path, output_path):
    """Convert WAV to 16kHz mono PCM using only stdlib."""
    with wave.open(input_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    # Convert to mono
    if n_channels == 1:
        mono = raw
    elif n_channels == 2:
        shorts = struct.unpack(f'<{n_frames * 2}h', raw)
        mono = struct.pack(f'<{n_frames}h', *[s >> 1 for s in shorts[::2]])
    else:
        raise ValueError(f"不支持的声道数: {n_channels}")

    # Resample from original rate to 16000 Hz
    if framerate == 16000:
        resampled = mono
        resampled_nframes = n_frames
    else:
        ratio = 16000 / framerate
        resampled_nframes = int(n_frames * ratio)
        old_short_count = len(mono) // 2
        old_indices = [i / ratio for i in range(resampled_nframes)]
        old_shorts = struct.unpack(f'<{old_short_count}h', mono)
        resampled = struct.pack(
            f'<{resampled_nframes}h',
            *[
                int(old_shorts[min(int(idx), old_short_count - 1)])
                for idx in old_indices
            ]
        )

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(resampled)

    return output_path


def get_audio_info(path):
    """Return basic info about an audio file."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "format": info.format,
        }
    except Exception:
        pass
    try:
        with wave.open(path, 'rb') as wf:
            return {
                "sample_rate": wf.getframerate(),
                "channels": wf.getnchannels(),
                "duration": wf.getnframes() / wf.getframerate(),
                "format": "WAV",
            }
    except Exception:
        return {}
