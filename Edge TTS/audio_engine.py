import os
import sys

VOICES = {
    "us_female": "en-US-AriaNeural",
    "us_male":   "en-US-GuyNeural",
    "uk_female": "en-GB-SoniaNeural",
    "uk_male":   "en-GB-RyanNeural",
}

SAVE_DIR = "./audio_cache"

def generate_wavs(word):
    """为指定单词静默生成4种口音的慢速WAV文件"""
    word = word.strip().replace('"', "'")
    if not word:
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[TTS引擎] 正在处理单词: '{word}'")

    for style, code in VOICES.items():
        file_name = f"{word}_{style}.wav"
        file_path = os.path.join(SAVE_DIR, file_name)
        
        cmd = f'{sys.executable} -m edge_tts --voice {code} --rate=-20% --text "{word}" --write-media "{file_path}"'
        os.system(cmd)
        
        if os.path.exists(file_path):
            print(f"  √ 已生成并正在播放: {file_name}")
            os.system(f"mpg123 -q {file_path}")
        else:
            print(f"  x 失败: {file_name}")

if __name__ == "__main__":
    target_word = "Cup" 
    generate_wavs(target_word)
    print("\n [TTS引擎] 全部音频文件已生成完毕")