import os
import pyaudio
import wave
import time
import threading

# 导入底层打分引擎
from Whisperv4.recognizer import WhisperRecognizer

# 视觉显色切片字典，保持与 config.py 中的打分块数量对齐
VISUAL_SYLLABLES = {
    "glasses": ["g", "l", "a", "ss", "es"],
    "watch": ["w", "a", "tch"],
    "card": ["c", "ar", "d"],
    "apple": ["a", "pp", "le"],
    "key": ["k", "ey"],
    "earphone": ["ear", "ph", "o", "ne"],
    "bottle": ["b", "o", "tt", "le"],
    "phone": ["ph", "o", "ne"],
    "pen": ["p", "e", "n"],
    "hello": ["h", "e", "ll", "o"],
    "bicycle": ["b", "i", "c", "y", "cle"],
}


def record_audio_manual(filename, rate=16000, chunk=1024):
    """通过麦克风手动控制录制音频 (回车开始，回车结束)"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

    frames = []
    is_recording = [True]

    def capture():
        """后台录音子线程"""
        while is_recording[0]:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

    input("\n[准备录音] 请按【回车键】开始录音...")
    print("🔴 正在录音中... (朗读完毕后，请再按一次【回车键】结束)")

    t = threading.Thread(target=capture)
    t.start()

    input()

    is_recording[0] = False
    t.join()

    print("⏹️ 录音已结束，正在提交引擎分析...\n")
    stream.stop_stream()
    stream.close()
    p.terminate()

    if frames:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))


def test_my_pronunciation(target_word):
    print("正在初始化 AI 语音评测引擎 (首次加载可能需要几秒钟)...")
    recognizer = WhisperRecognizer(model_name="base")

    # 🌟 修复关键：引入时间戳，确保每次运行时生成全新独立的文件名，彻底避开底层缓存机制
    timestamp = int(time.time())
    audio_file = f"test_record_{timestamp}.wav"

    print(f"\n--- 当前测试单词: {target_word.upper()} ---")

    # 调用全新的手动录音函数
    record_audio_manual(audio_file)

    # 核心调用
    result = recognizer.score_word(audio_file, target_word)

    # 打印全局分析结果
    print("=== 全局诊断结果 ===")
    print(f"综合得分: {result.get('overall', 0)}")
    print(f"AI听成:   [{result.get('recognized', '')}]")
    print(f"匹配类型: {result.get('match_type', '')}\n")

    # 打印音节碎片分数
    syllables_data = result.get("syllables", [])
    visual_parts = VISUAL_SYLLABLES.get(target_word, [target_word])

    print("=== 字母级碎块得分详情 ===")
    for i, part in enumerate(visual_parts):
        if i < len(syllables_data):
            score = syllables_data[i].get("segment_score", 0)
            status = syllables_data[i].get("status", "improve")

            if status == "excellent":
                status_text = "🟩 优秀"
            elif status == "pass":
                status_text = "🟨 及格"
            else:
                status_text = "🟥 需改进"

            print(f"[{part.ljust(8)}] 得分: {score:5.2f} / 100  |  状态: {status_text}")
        else:
            print(f"[{part.ljust(8)}] 无底层数据返回")

    print("\n测试完成。你可以修改代码底部的目标单词进行其他测试。")

    # 测试结束后，清理掉产生的临时音频文件，保持目录整洁
    try:
        os.remove(audio_file)
        os.remove(audio_file.replace(".wav", "_16k.wav"))
    except Exception:
        pass


if __name__ == "__main__":
    test_my_pronunciation("glasses")
