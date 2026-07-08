# Whisper/__init__.py
from .recognizer import WhisperRecognizer
from .config import WORD_SYLLABLES, SCORE_SETTINGS, TARGET_WORDS
from .audio_utils import prepare_audio, get_audio_info

__all__ = [
    "WhisperRecognizer",
    "WORD_SYLLABLES",
    "SCORE_SETTINGS",
    "TARGET_WORDS",
    "prepare_audio",
    "get_audio_info",
]
