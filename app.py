import sys
import cv2
import time
import numpy as np
import pyaudio
import wave
import subprocess
import os
import hashlib
import json
import urllib.request
import urllib.error
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 👇 Whisper 全局模型缓存，防止重复加载
_global_whisper_recognizer = None

# 隐藏 Linux 系统底层的 dbind-WARNING 刷屏提示
os.environ["NO_AT_BRIDGE"] = "1"

# 导入 PySide6 核心库
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

# ================= 核心知识库 (已升级为多语言多模态库，并新增了例句翻译) =================
CLASSES = {
    1: 'apple', 2: 'bottle', 3: 'card', 4: 'earphone',
    5: 'glasses', 6: 'key', 7: 'pen', 8: 'phone', 9: 'watch'
}

KNOWLEDGE_BASE = {
    'glasses': {
        '英语': ('Glasses', '/ˈɡlæsɪz/', 'She is wearing a pair of reading glasses.', '她戴着一副老花镜。'),
        '俄语': ('Очки', '/ɐˈt͡ɕkʲi/', 'Она носит очки для чтения.', '她戴着一副老花镜。'),
        '法语': ('Lunettes', '/ly.nɛt/', 'Elle porte des lunettes de lecture.', '她戴着一副老花镜。')
    },
    'watch': {
        '英语': ('Watch', '/wɒtʃ/', 'He looked at his smart watch.', '他看了看他的智能手表。'),
        '俄语': ('Часы', '/t͡ɕɪˈsɨ/', 'Он посмотрел на свои умные часы.', '他看了看他的智能手表。'),
        '法语': ('Montre', '/mɔ̃tʁ/', 'Il a regardé sa montre connectée.', '他看了看他的智能手表。')
    },
    'card': {
        '英语': ('Card', '/kɑːd/', 'Please swipe your access card.', '请刷您的门禁卡。'),
        '俄语': ('Карта', '/ˈkartə/', 'Пожалуйста, приложите карту доступа.', '请刷您的门禁卡。'),
        '法语': ('Carte', '/kaʁt/', 'Veuillez badger votre carte d\'accès.', '请刷您的门禁卡。')
    },
    'apple': {
        '英语': ('Apple', '/ˈæpl/', 'An apple a day keeps the doctor away.', '一天一苹果，医生远离我。'),
        '俄语': ('Яблоко', '/ˈjabləkə/', 'Одно яблоко в день спасает от врачей.', '一天一苹果，医生远离我。'),
        '法语': ('Pomme', '/pɔm/', 'Une pomme par jour éloigne le médecin.', '一天一苹果，医生远离我。')
    },
    'key': {
        '英语': ('Key', '/kiː/', 'I forgot my house key.', '我忘了房间钥匙。'),
        '俄语': ('Ключ', '/klʲʉt͡ɕ/', 'Я забыл ключ от дома.', '我忘了房间钥匙。'),
        '法语': ('Clé', '/kle/', 'J\'ai oublié la clé de ma maison.', '我忘了房间钥匙。')
    },
    'earphone': {
        '英语': ('Earphone', '/ˈɪəfəʊn/', 'I put on my earphone to listen to music.', '我戴上耳机听音乐。'),
        '俄语': ('Наушник', '/nɐˈuʂnʲɪk/', 'Я надел наушники, чтобы послушать музыку.', '我戴上耳机听音乐。'),
        '法语': ('Écouteur', '/e.ku.tœʁ/', 'J\'ai mis mes écouteurs pour écouter de la musique.', '我戴上耳机听音乐。')
    },
    'bottle': {
        '英语': ('Bottle', '/ˈbɒtl/', 'Fill the water bottle, please.', '请把水瓶倒满。'),
        '俄语': ('Бутылка', '/bʊˈtɨlkə/', 'Наполните бутылку водой, пожалуйста.', '请把水瓶倒满。'),
        '法语': ('Bouteille', '/bu.tɛj/', 'Remplissez la bouteille d\'eau, s\'il vous plaît.', '请把水瓶倒满。')
    },
    'phone': {
        '英语': ('Phone', '/fəʊn/', 'My phone is out of battery.', '我的手机没电了。'),
        '俄语': ('Телефон', '/tʲɪlʲɪˈfon/', 'Мой телефон разрядился.', '我的手机没电了。'),
        '法语': ('Téléphone', '/te.le.fɔn/', 'Mon téléphone n\'a plus de batterie.', '我的手机没电了。')
    },
    'pen': {
        '英语': ('Pen', '/pɛn/', 'Can I borrow your blue pen?', '我可以借一下你的蓝色圆珠笔吗？'),
        '俄语': ('Ручка', '/ˈrut͡ɕkə/', 'Можно я возьму твою синюю ручку?', 'Можно я возьму твою синюю ручку?'),
        '法语': ('Stylo', '/sti.lo/', 'Puis-je emprunter ton stylo bleu?', '我可以借一下你的蓝色圆珠笔吗？')
    }
}

CHINESE_TRANSLATIONS = {
    "apple": "苹果",
    "bottle": "水瓶",
    "card": "卡片",
    "earphone": "耳机",
    "glasses": "眼镜",
    "key": "钥匙",
    "pen": "钢笔",
    "phone": "手机",
    "watch": "手表"
}

VISUAL_SYLLABLES = {
    "glasses": ["g", "l", "a", "ss", "es"],
    "watch": ["w", "a", "tch"],
    "card": ["c", "ar", "d"],
    "apple": ["a", "pp", "le"],
    "key": ["k", "ey"],
    "earphone": ["ear", "ph", "o", "ne"],
    "bottle": ["b", "o", "tt", "le"],
    "phone": ["ph", "o", "ne"],
    "pen": ["p", "e", "n"]
}
# ===================================================

# ================= 全新现代化 QSS 样式表 =================
STYLE_SHEET = """
QMainWindow { background-color: #0b0f19; }

QFrame#Card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 15px;
}

QLabel#VideoScreen {
    background-color: #000000;
    border-radius: 15px;
}

QLabel#ResultTitle { color: #8b949e; font-size: 16px; }
QLabel#ResultWord { color: #ffffff; font-size: 16px; }
QLabel#TransLabel { color: #ffffff; font-size: 16px; }
QLabel#PhoneticLabel { color: #8b949e; font-size: 14px; font-family: "Segoe UI"; }
QLabel#ExampleLabel { color: #a5d6ff; font-size: 14px; font-style: italic; }

QComboBox { 
    background-color: #21262d; border: 1px solid #30363d; 
    border-radius: 8px; padding: 6px; color: white; font-size: 14px;
}
QComboBox::drop-down { border: none; }

QComboBox QAbstractItemView {
    background-color: #21262d;   
    color: #ffffff;              
    selection-background-color: #30363d; 
    border: 1px solid #30363d;
    outline: none;
}

QPushButton[class="OptionBtn"] { 
    background-color: #21262d; color: #8b949e; 
    border: 1px solid #30363d; border-radius: 12px; 
    min-height: 28px; font-size: 13px; font-weight: bold;
}
QPushButton[class="OptionBtn"]:checked { 
    background-color: #a5d6ff; color: #0b0f19; border: none; 
}

QToolButton {
    border-radius: 20px;
    font-weight: bold;
    font-size: 14px;
    color: #11111b;
    padding: 4px 8px;
    text-align: center;
}
QToolButton#AnalyzeBtn { background-color: #baddff; } 
QToolButton#AnalyzeBtn:hover { background-color: #a5ccff; }
QToolButton#AnalyzeBtn:disabled { background-color: #3b4252; color: #818896; }

QToolButton#RecordBtn { background-color: #ffc4d0; } 
QToolButton#RecordBtn:hover { background-color: #ffb3c2; }
QToolButton#RecordBtn:pressed { background-color: #e696a6; } 

QToolButton#PlayBtn { background-color: #bbfabb; } 
QToolButton#PlayBtn:hover { background-color: #aae8aa; }
QToolButton#PlayBtn:disabled { background-color: #8fcb8f; color: #11111b; } 

QPushButton#ExamplePlayBtn {
    background-color: transparent;
    border: none;
}
QPushButton#ExamplePlayBtn:hover {
    background-color: #21262d;
    border-radius: 4px;
}

QTextEdit { 
    background-color: transparent; border: none; 
    color: #c9d1d9; font-size: 14px;
}

QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #484f58;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #8b949e;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px; 
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none; 
}
"""
# ===================================================

class ClickableLabel(QLabel):
    clicked_signal = Signal(int, int)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.clicked_signal.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class WaveformWidget(QWidget):
    clicked_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedHeight(90)
        self.std_path = None
        self.user_path = None

    def update_waves(self, std_path, user_path):
        self.std_path = std_path
        self.user_path = user_path
        self.update()

    def mousePressEvent(self, event):
        self.clicked_signal.emit()
        super().mousePressEvent(event)

    def _get_trimmed_audio(self, path):
        if not path or not os.path.exists(path):
            return None, 0
        try:
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                if n_frames == 0:
                    return None, 0
                audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                if wf.getnchannels() == 2:
                    audio_data = audio_data[::2]
        except Exception:
            return None, 0

        if len(audio_data) == 0:
            return None, 0

        max_amp = np.max(np.abs(audio_data))
        if max_amp < 150:
            return None, 0

        threshold = max(max_amp * 0.08, 150)
        active_indices = np.where(np.abs(audio_data) > threshold)[0]
        if len(active_indices) == 0:
            return None, 0

        start_idx = active_indices[0]
        end_idx = active_indices[-1]

        padding = int(sr * 0.1)
        start_idx = max(0, start_idx - padding)
        end_idx = min(len(audio_data), end_idx + padding)

        trimmed_data = audio_data[start_idx:end_idx]
        trimmed_data = trimmed_data / max_amp
        return trimmed_data, sr

    def _get_raw_waveform_data(self, trimmed_data, sr, max_dur, width):
        mins = np.zeros(width)
        maxs = np.zeros(width)
        if trimmed_data is None or len(trimmed_data) == 0:
            return mins, maxs

        for x in range(width):
            t_start = x * max_dur / width
            t_end = (x + 1) * max_dur / width
            idx_start = int(t_start * sr)
            idx_end = int(t_end * sr)
            if idx_start == idx_end:
                idx_end = idx_start + 1

            if idx_start < len(trimmed_data):
                slice_data = trimmed_data[idx_start:min(idx_end, len(trimmed_data))]
                if len(slice_data) > 0:
                    mins[x] = np.min(slice_data)
                    maxs[x] = np.max(slice_data)

        return mins, maxs

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()
        mid_y = h / 2.0

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 15, 15)
        painter.fillPath(path, QColor("#000000"))
        painter.setClipPath(path)

        painter.setPen(QPen(QColor("#003300"), 1))
        grid_size = 15
        for i in range(0, w, grid_size):
            painter.drawLine(i, 0, i, h)
        for i in range(0, h, grid_size):
            painter.drawLine(0, i, w, i)

        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        text1 = "■ 示范单词"
        painter.setPen(QPen(QColor("#00e5ff")))
        painter.drawText(w - fm.horizontalAdvance(text1) - 10, 20, text1)

        text2 = "■ 我的音频"
        painter.setPen(QPen(QColor("#ff9900")))
        painter.drawText(w - fm.horizontalAdvance(text2) - 10, 40, text2)

        std_data, std_sr = self._get_trimmed_audio(self.std_path)
        user_data, user_sr = self._get_trimmed_audio(self.user_path)

        std_dur = len(std_data) / std_sr if std_data is not None else 0
        user_dur = len(user_data) / user_sr if user_data is not None else 0

        max_dur = max(std_dur, user_dur)
        if max_dur == 0:
            painter.setPen(QPen(QColor("#00aa00"), 1))
            painter.drawLine(0, int(mid_y), w, int(mid_y))
            return

        std_mins, std_maxs = self._get_raw_waveform_data(std_data, std_sr, max_dur, w)
        user_mins, user_maxs = self._get_raw_waveform_data(user_data, user_sr, max_dur, w)

        if std_data is not None and user_data is not None:
            if np.max(std_maxs) > 0 and np.max(user_maxs) > 0:
                correlation = np.correlate(std_maxs, user_maxs, mode='full')
                shift = np.argmax(correlation) - (len(user_maxs) - 1)

                aligned_user_mins = np.zeros_like(user_mins)
                aligned_user_maxs = np.zeros_like(user_maxs)
                if shift > 0:
                    aligned_user_mins[shift:] = user_mins[:-shift]
                    aligned_user_maxs[shift:] = user_maxs[:-shift]
                elif shift < 0:
                    aligned_user_mins[:shift] = user_mins[-shift:]
                    aligned_user_maxs[:shift] = user_maxs[-shift:]
                else:
                    aligned_user_mins = user_mins
                    aligned_user_maxs = user_maxs
                user_mins = aligned_user_mins
                user_maxs = aligned_user_maxs

        painter.setPen(QPen(QColor("#00ff00"), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        def draw_raw_wave(mins, maxs, color_hex):
            painter.setPen(QPen(QColor(color_hex), 1))
            for x in range(w):
                if mins[x] != 0 or maxs[x] != 0:
                    y1 = int(mid_y - maxs[x] * (h / 2.0 - 5))
                    y2 = int(mid_y - mins[x] * (h / 2.0 - 5))
                    painter.drawLine(x, y1, x, y2)

        if std_data is not None:
            draw_raw_wave(std_mins, std_maxs, "#00e5ff")
        if user_data is not None:
            draw_raw_wave(user_mins, user_maxs, "#ff9900")


# ----------------- 核心线程区 -----------------
class CameraThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.exposure_level = 62
        self._pending_exposure_level = 62

    def set_exposure(self, value):
        self._pending_exposure_level = int(value)

    def _apply_exposure_settings(self, cap):
        level = max(0, min(100, self.exposure_level))
        hw_exposure = int(round(-13 + level * 12 / 100))
        brightness = int(round(35 + level * 95 / 100))
        gain = int(round(level * 30 / 100))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, hw_exposure)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_GAIN, gain)

    def _adjust_frame_brightness(self, frame):
        level = max(0, min(100, self.exposure_level))
        alpha = 0.75 + level * 0.55 / 100
        beta = -35 + level * 75 / 100
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("未检测到可用摄像头！")
            return
            
        self._apply_exposure_settings(cap)

        for _ in range(10): cap.read(); time.sleep(0.03)
        while self._run_flag:
            if self._pending_exposure_level != self.exposure_level:
                self.exposure_level = self._pending_exposure_level
                self._apply_exposure_settings(cap)
            ret, cv_img = cap.read()
            if ret:
                cv_img = self._adjust_frame_brightness(cv_img)
                self.change_pixmap_signal.emit(cv_img)
            else:
                time.sleep(0.1)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VisionThread(QThread):
    finished_signal = Signal(bool, str, np.ndarray, list)

    def __init__(self, frame):
        super().__init__()
        self.frame = frame.copy()
        self.input_size = (384, 384)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def letterbox(self, img, new_shape=(384, 384), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def run(self):
        try:
            model_path = os.path.join(BASE_DIR, "RF-DETR", "rf_detr_nano.onnx")
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name

            h_orig, w_orig = self.frame.shape[:2]
            display_img = self.frame.copy()

            img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img_padded, r, (dw, dh) = self.letterbox(img_rgb, self.input_size)

            img_input = img_padded.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_input, axis=0)

            outputs = session.run(None, {input_name: input_tensor})

            raw_logits = outputs[4][0]
            raw_boxes = outputs[5][0]
            probs = self.sigmoid(raw_logits)

            num_classes = len(CLASSES)
            scores = np.max(probs[:, :num_classes], axis=-1)
            labels = np.argmax(probs[:, :num_classes], axis=-1)

            conf_threshold = 0.5
            nms_threshold = 0.45

            rects = []
            confidences = []
            class_ids = []

            for i in range(len(scores)):
                if scores[i] > conf_threshold:
                    cx, cy, bw, bh = raw_boxes[i]
                    px_cx = cx * self.input_size[0]
                    px_cy = cy * self.input_size[1]
                    px_w = bw * self.input_size[0]
                    px_h = bh * self.input_size[1]

                    orig_cx = (px_cx - dw) / r
                    orig_cy = (px_cy - dh) / r
                    orig_w = px_w / r
                    orig_h = px_h / r

                    w_box = int(orig_w)
                    h_box = int(orig_h)
                    x = int(orig_cx - orig_w / 2)
                    y = int(orig_cy - orig_h / 2)

                    rects.append([x, y, w_box, h_box])
                    confidences.append(float(scores[i]))
                    class_ids.append(int(labels[i]))

            indices = cv2.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)

            detected_objects = []
            if len(indices) > 0:
                final_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                for i in final_indices:
                    x, y, w_b, h_b = rects[i]
                    conf = confidences[i]
                    cls_id = class_ids[i]

                    x1 = max(0, min(w_orig - 1, int(x)))
                    y1 = max(0, min(h_orig - 1, int(y)))
                    x2 = max(0, min(w_orig, int(x + w_b)))
                    y2 = max(0, min(h_orig, int(y + h_b)))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue

                    detected_objects.append((cls_id, conf, x1, y1, x2, y2))

                    label_text = f"{CLASSES.get(cls_id, 'obj')} {conf:.2f}"
                    cv2.rectangle(display_img, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
                    cv2.putText(display_img, label_text, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not detected_objects:
                self.finished_signal.emit(True, "未发现目标", display_img, [])
            else:
                self.finished_signal.emit(True, f"识别到 {len(detected_objects)} 个目标", display_img, detected_objects)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.finished_signal.emit(False, f"识别失败: {str(e)}", np.zeros((100, 100, 3), dtype=np.uint8), [])


class AudioRecorderThread(QThread):
    recording_finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._is_recording = False
        self.filename = ""
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.frames = []

    def run(self):
        self._is_recording = True
        self.frames = []
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)
            while self._is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            self.error_signal.emit(f"录音设备异常: {str(e)}")
            self._is_recording = False
        finally:
            p.terminate()

        if self.frames:
            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            self.recording_finished_signal.emit(self.filename)

    def stop(self):
        self._is_recording = False
        self.wait()


class EvaluationThread(QThread):
    eval_finished_signal = Signal(int, str, list)

    def __init__(self, filename, target_word_en):
        super().__init__()
        self.filename = filename
        self.target_word_en = target_word_en

    def run(self):
        global _global_whisper_recognizer
        try:
            if _global_whisper_recognizer is None:
                from Whisper.recognizer import WhisperRecognizer
                _global_whisper_recognizer = WhisperRecognizer(model_name="base")

            word = self.target_word_en.lower()
            # 从全局引擎中获取评分结果
            result = _global_whisper_recognizer.score_word(self.filename, word)

            raw_score = int(result.get("overall", 0))
            match_type = result.get("match_type", "")
            recognized_text = result.get("recognized", "")
            syllables_data = result.get("syllables", [])
            diagnostic_advice = result.get("diagnostic_advice", "")

            # 👇 【终极真理：彻底信任底层的声学计算结果！】
            # 不再做任何阶梯打折，底层算出来多少波形分，我们就给多少分。
            score = raw_score
            
            # 引入细微随机数防机械感
            import random
            score += random.randint(-1, 1)
            score = max(0, min(100, score))

            # 依据真实的声学得分输出反馈
            if diagnostic_advice:
                feedback = diagnostic_advice
            else:
                if score >= 80:
                    feedback = f"Perfect！发音非常标准，完美匹配单词 [{word}]！"
                elif score >= 60:
                    feedback = f"发音不错，但部分音节的清晰度还可以提高。"
                else:
                    if recognized_text and match_type == "not_found":
                        feedback = f"发音偏差较大，AI听成了 [{recognized_text}]。请注意口型再试一次。"
                    else:
                        feedback = f"未检测到有效声音，请靠近麦克风大声朗读。"

            self.eval_finished_signal.emit(score, feedback, syllables_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.eval_finished_signal.emit(0, f"AI评分引擎异常: {str(e)}", [])


class DeepSeekExampleThread(QThread):
    finished_signal = Signal(bool, str, str, str)

    def __init__(self, word_en, target_word, target_lang, fallback_example, fallback_translation):
        super().__init__()
        self.word_en = word_en
        self.target_word = target_word
        self.target_lang = target_lang
        self.fallback_example = fallback_example
        self.fallback_translation = fallback_translation

    def run(self):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            self.finished_signal.emit(False, self.fallback_example, self.fallback_translation,
                                      "DEEPSEEK_API_KEY is not configured.")
            return

        prompt = (
            "请为语言学习系统生成一个简单、自然、适合初学者跟读的例句。"
            f"识别到的英文物体标签是 {self.word_en}，当前目标语言是 {self.target_lang}，"
            f"目标语言词汇是 {self.target_word}。"
            "只返回严格 JSON，字段为 example 和 translation。"
            "example 使用目标语言，translation 使用简体中文。"
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是多语种语言学习助手，只输出用户要求的 JSON。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.75,
            "max_tokens": 180
        }
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            generated = json.loads(content)
            example = str(generated.get("example", "")).strip()
            translation = str(generated.get("translation", "")).strip()
            if not example:
                raise ValueError("DeepSeek response missing example.")
            self.finished_signal.emit(True, example, translation, "")
        except Exception as e:
            self.finished_signal.emit(False, self.fallback_example, self.fallback_translation, str(e))


class DeepSeekAdviceThread(QThread):
    finished_signal = Signal(bool, str)

    def __init__(self, word_en, score, syllables_data, original_feedback):
        super().__init__()
        self.word_en = word_en
        self.score = score
        self.syllables_data = syllables_data
        self.original_feedback = original_feedback

    def run(self):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            self.finished_signal.emit(False, self.original_feedback)
            return

        syllables_payload = []
        for item in self.syllables_data or []:
            syllables_payload.append({
                "syllable": item.get("syllable") or item.get("text") or "",
                "score": item.get("score"),
                "status": item.get("status", "")
            })

        prompt = (
            "你是英语发音教练。请根据评分结果给出简洁但具体可执行的中文改进建议。"
            "建议必须对应具体音节或音素，例如爆破音气流缺失、元音饱满度不足、鼻音共鸣不够、"
            "双唇或舌尖动作不够干脆等。"
            "输出 1 到 3 条建议即可，不要输出 JSON。"
            f"\n单词: {self.word_en}\n总分: {self.score}\n音节数据: "
            + json.dumps(syllables_payload, ensure_ascii=False)
            + f"\n本地评分反馈: {self.original_feedback}"
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是严谨的发音纠错老师，反馈要短、准、可执行。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.35,
            "max_tokens": 220
        }
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
            advice = result["choices"][0]["message"]["content"].strip()
            self.finished_signal.emit(True, advice)
        except Exception:
            self.finished_signal.emit(False, self.original_feedback)


class EdgeTTSThread(QThread):
    finished_signal = Signal(bool, str, str)

    def __init__(self, text, lang, voice_style, skip_wav_emit=False):
        super().__init__()
        self.text = text
        self.lang = lang
        self.voice_style = voice_style
        self.skip_wav_emit = skip_wav_emit  
        self._stop_requested = False
        self._player_process = None

    def stop(self):
        self._stop_requested = True
        if self._player_process and self._player_process.poll() is None:
            try:
                self._player_process.terminate()
            except Exception:
                pass

    def _play_mp3(self, file_path_mp3):
        if self._stop_requested:
            return False
        self._player_process = subprocess.Popen(["mpg123", "-q", file_path_mp3])
        while self._player_process.poll() is None:
            if self._stop_requested:
                self.stop()
                try:
                    self._player_process.wait(timeout=1)
                except Exception:
                    pass
                return False
            self.msleep(30)
        return self._player_process.returncode == 0

    def run(self):
        if self.lang == "英语":
            if "美式女" in self.voice_style: voice_code = "en-US-AriaNeural"
            elif "美式男" in self.voice_style: voice_code = "en-US-GuyNeural"
            elif "英式女" in self.voice_style: voice_code = "en-GB-SoniaNeural"
            elif "英式男" in self.voice_style: voice_code = "en-GB-RyanNeural"
            else: voice_code = "en-US-AriaNeural"
        elif self.lang == "俄语":
            voice_code = "ru-RU-SvetlanaNeural" if "女" in self.voice_style else "ru-RU-DmitryNeural"
        elif self.lang == "法语":
            voice_code = "fr-FR-DeniseNeural" if "女" in self.voice_style else "fr-FR-HenriNeural"
        elif self.lang == "中文":  
            voice_code = "zh-CN-XiaoxiaoNeural"
        else:
            voice_code = "en-US-AriaNeural"

        os.makedirs("./audio_cache", exist_ok=True)
        cache_key = hashlib.sha256(
            f"{voice_code}|{self.lang}|{self.voice_style}|{self.text}".encode("utf-8")
        ).hexdigest()[:24]
        file_path_mp3 = f"./audio_cache/tts_{cache_key}.mp3"
        file_path_wav = f"./audio_cache/tts_{cache_key}.wav"

        mp3_ready = os.path.exists(file_path_mp3) and os.path.getsize(file_path_mp3) > 100
        wav_ready = os.path.exists(file_path_wav) and os.path.getsize(file_path_wav) > 100
        if mp3_ready:
            if not self.skip_wav_emit and not wav_ready:
                subprocess.run(["mpg123", "-w", file_path_wav, file_path_mp3], capture_output=True)
                wav_ready = os.path.exists(file_path_wav) and os.path.getsize(file_path_wav) > 100
            played = self._play_mp3(file_path_mp3)
            if self._stop_requested:
                return
            if self.skip_wav_emit or wav_ready:
                self.finished_signal.emit(played, "Loaded from TTS cache" if played else "Playback stopped",
                                          "" if self.skip_wav_emit else file_path_wav)
                return

        cmd = [sys.executable, "-m", "edge_tts", "--voice", voice_code, "--rate=-15%", "--text", self.text,
               "--write-media", file_path_mp3]

        success = False
        error_msg = ""

        for attempt in range(3):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0 and os.path.exists(file_path_mp3) and os.path.getsize(file_path_mp3) > 100:
                    if not self.skip_wav_emit:
                        subprocess.run(["mpg123", "-w", file_path_wav, file_path_mp3], capture_output=True)
                    success = self._play_mp3(file_path_mp3)
                    if self._stop_requested:
                        return
                    break
                else:
                    error_msg = "API未返回有效数据"
                    time.sleep(0.5)
            except Exception as e:
                error_msg = str(e)
                time.sleep(0.5)

        if success:
            self.finished_signal.emit(True, "发音完毕", "" if self.skip_wav_emit else file_path_wav)
        else:
            self.finished_signal.emit(False, f"失败: {error_msg}", "")


# ----------------------------------------------------------------------

class LanguageLearnerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模态智慧助教 - 视觉增强版")
        self.resize(800, 480)
        self.setMinimumSize(800, 480)
        self.setStyleSheet(STYLE_SHEET)

        self.current_frame = None
        self.saved_raw_frame = None
        self.current_boxes = []
        self.is_showing_result = False
        self.is_zoomed_in = False
        self.result_pixmap_full = None
        self.current_word_en = ""
        self.current_target_text = ""
        self.current_example_text = ""  
        self._example_request_token = 0
        self._deepseek_threads = []
        self._deepseek_advice_threads = []

        self.std_wav_path = None
        self.user_wav_path = None
        self.user_audio_play_process = None

        self.initUI()

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_frame)
        self.camera_thread.error_signal.connect(self.on_system_error)
        self.camera_thread.start()

        self.audio_thread = AudioRecorderThread()
        self.audio_thread.recording_finished_signal.connect(self.on_recording_finished)
        self.audio_thread.error_signal.connect(self.on_system_error)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(5, 5, 5, 5)

        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(6)

        exposure_panel = QVBoxLayout()
        exposure_panel.setContentsMargins(0, 4, 0, 4)
        exposure_label = QLabel("曝光", styleSheet="color:#8b949e; font-size:10px;")
        exposure_label.setAlignment(Qt.AlignCenter)
        self.exposure_slider = QSlider(Qt.Vertical)
        self.exposure_slider.setRange(0, 100)
        self.exposure_slider.setValue(62)
        self.exposure_slider.setFixedWidth(24)
        self.exposure_slider.setToolTip("向下更暗，向上更亮")
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)
        exposure_panel.addWidget(exposure_label)
        exposure_panel.addWidget(self.exposure_slider, 1)

        self.video_label = ClickableLabel("摄像头连接中...")
        self.video_label.setObjectName("VideoScreen")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.clicked_signal.connect(self.on_video_clicked)
        camera_row.addLayout(exposure_panel)
        camera_row.addWidget(self.video_label, 1)
        left_layout.addLayout(camera_row, 1)

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.clicked_signal.connect(self.play_user_recording)
        left_layout.addWidget(self.waveform_widget)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        result_card = QFrame()
        result_card.setObjectName("Card")
        result_layout = QVBoxLayout(result_card)

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("识别结果: ", objectName="ResultTitle"))
        self.word_label = QLabel("Pending", objectName="ResultWord")
        title_layout.addWidget(self.word_label)
        title_layout.addStretch()

        self.trans_label = QLabel("翻译: --", objectName="TransLabel")
        self.phonetic_label = QLabel("音标: --", objectName="PhoneticLabel")
        
        example_box = QHBoxLayout()
        self.example_label = QLabel("例句: --", objectName="ExampleLabel")
        self.example_label.setWordWrap(True)
        
        self.btn_play_example = QPushButton()
        self.btn_play_example.setObjectName("ExamplePlayBtn")
        self.btn_play_example.setIcon(QIcon("./icons/speaker.png")) 
        self.btn_play_example.setIconSize(QSize(16, 16))
        self.btn_play_example.setFixedSize(24, 24)
        self.btn_play_example.setVisible(False) 
        self.btn_play_example.clicked.connect(self.play_example_tts)

        colorize_effect = QGraphicsColorizeEffect(self.btn_play_example)
        colorize_effect.setColor(QColor("#FFFFFF"))  
        colorize_effect.setStrength(1.0)              
        self.btn_play_example.setGraphicsEffect(colorize_effect)

        example_box.addWidget(self.example_label, 1)
        example_box.addWidget(self.btn_play_example, 0, Qt.AlignVCenter)

        result_layout.addLayout(title_layout)
        result_layout.addWidget(self.trans_label)
        result_layout.addWidget(self.phonetic_label)
        result_layout.addLayout(example_box)

        settings_card = QFrame()
        settings_card.setObjectName("Card")
        settings_layout = QVBoxLayout(settings_card)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("目标语言:", styleSheet="color:#8b949e;"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["英语", "俄语", "法语"])
        self.lang_combo.currentTextChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_combo)

        voice_label_layout = QHBoxLayout()
        voice_label_layout.addWidget(QLabel("发音人声:", styleSheet="color:#8b949e;"))

        self.voice_group_box = QWidget()
        self.voice_group_box.setMinimumHeight(62)
        self.voice_layout = QGridLayout(self.voice_group_box)
        self.voice_layout.setContentsMargins(0, 0, 0, 0)
        self.voice_layout.setSpacing(6)
        self.voice_group = QButtonGroup(self)

        settings_layout.addLayout(lang_layout)
        settings_layout.addLayout(voice_label_layout)
        settings_layout.addWidget(self.voice_group_box)
        self.update_voice_buttons(self.lang_combo.currentText())

        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)

        self.btn_snapshot = QToolButton()
        self.btn_snapshot.setObjectName("AnalyzeBtn")
        self.btn_snapshot.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_snapshot.setText("拍摄\n分析")
        self.btn_snapshot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_snapshot.setFixedHeight(60)
        self.btn_snapshot.clicked.connect(self.handle_snapshot_click)
        self.btn_snapshot.setIcon(QIcon("./icons/camera.png"))
        self.btn_snapshot.setIconSize(QSize(30, 30))

        self.btn_record = QToolButton()
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_record.setText("按住\n跟读")
        self.btn_record.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_record.setFixedHeight(60)
        self.btn_record.pressed.connect(self.start_recording)
        self.btn_record.released.connect(self.stop_recording)
        self.btn_record.setIcon(QIcon("./icons/mic.png"))
        self.btn_record.setIconSize(QSize(30, 30))

        self.btn_play = QToolButton()
        self.btn_play.setObjectName("PlayBtn")
        self.btn_play.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_play.setText("发音\n示范")
        self.btn_play.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_play.setFixedHeight(60)
        self.btn_play.clicked.connect(self.play_demo)
        self.btn_play.setIcon(QIcon("./icons/speaker.png"))
        self.btn_play.setIconSize(QSize(30, 30))

        action_layout.addWidget(self.btn_snapshot)
        action_layout.addWidget(self.btn_record)
        action_layout.addWidget(self.btn_play)

        feedback_card = QFrame()
        feedback_card.setObjectName("Card")
        feedback_layout = QVBoxLayout(feedback_card)
        feedback_layout.setContentsMargins(8, 8, 8, 6)
        self.feedback_text = QTextEdit()
        self.feedback_text.setPlaceholderText("AI Feedback: Ready to explore...")
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setMinimumHeight(130)
        self.feedback_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.feedback_text.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.feedback_text.document().setDocumentMargin(8)
        self.feedback_text.textChanged.connect(self.scroll_feedback_to_bottom)
        feedback_layout.addWidget(self.feedback_text)

        right_panel.addWidget(result_card, 1)
        right_panel.addWidget(settings_card, 3)
        right_panel.addLayout(action_layout, 1)
        right_panel.addWidget(feedback_card, 6)

        main_layout.addWidget(left_card, 6)
        main_layout.addLayout(right_panel, 4)

    def scroll_feedback_to_bottom(self):
        QTimer.singleShot(0, self._scroll_feedback_now)
        QTimer.singleShot(80, self._scroll_feedback_now)

    def _scroll_feedback_now(self):
        self.feedback_text.moveCursor(QTextCursor.End)
        bar = self.feedback_text.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.feedback_text.ensureCursorVisible()

    def on_exposure_changed(self, value):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.set_exposure(value)

    def clear_waveforms(self):
        self.std_wav_path = None
        self.user_wav_path = None
        if hasattr(self, 'waveform_widget'):
            self.waveform_widget.update_waves(None, None)

    def update_voice_buttons(self, lang):
        for btn in self.voice_group.buttons():
            self.voice_group.removeButton(btn)
            self.voice_layout.removeWidget(btn)
            btn.deleteLater()

        if lang == "英语":
            voice_list = ["美式男声", "美式女声", "英式男声", "英式女声"]
            default_idx = 1
        else:
            voice_list = ["男声 (Male)", "女声 (Female)"]
            default_idx = 1

        for i, name in enumerate(voice_list):
            btn = QPushButton(name)
            btn.setProperty("class", "OptionBtn")
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            btn.setMaximumHeight(30)
            self.voice_group.addButton(btn, i)
            if i == default_idx:
                btn.setChecked(True)

            if lang == "英语":
                self.voice_layout.addWidget(btn, i // 2, i % 2)
            else:
                self.voice_layout.addWidget(btn, 0, i)

            btn.clicked.connect(self.clear_waveforms)

    def on_language_changed(self, new_lang):
        self.update_voice_buttons(new_lang)
        self.refresh_translation()
        self.clear_waveforms()

    @Slot(np.ndarray)
    def update_frame(self, cv_img):
        self.current_frame = cv_img
        if self.is_showing_result: return
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def handle_snapshot_click(self):
        if self.is_showing_result:
            self.is_showing_result = False
            self.is_zoomed_in = False
            self.current_boxes = []
            self.result_pixmap_full = None
            self.btn_snapshot.setText("拍摄\n分析")
            self.word_label.setText("Pending")
            self.phonetic_label.setText("音标: --")
            self.trans_label.setText("翻译: --")
            self.example_label.setText("例句: --")
            self.btn_play_example.setVisible(False)
            self.clear_waveforms()
            self.feedback_text.append("<b style='color:#baddff;'>[System]</b> Switched to live view.")
        else:
            self.take_vision_snapshot()

    def take_vision_snapshot(self):
        if self.current_frame is not None:
            if hasattr(self, 'vision_thread') and self.vision_thread.isRunning():
                return
            self.saved_raw_frame = self.current_frame.copy()
            self.is_showing_result = True
            self.btn_snapshot.setEnabled(False)
            self.btn_snapshot.setText("分析\n中...")
            self.feedback_text.append("<b style='color:#aae8aa;'>[Vision]</b> Running AI inference...")
            self.clear_waveforms()

            self.vision_thread = VisionThread(self.saved_raw_frame)
            self.vision_thread.finished_signal.connect(self.on_vision_finished)
            self.vision_thread.start()
        else:
            self.on_system_error("Camera not ready.")

    @Slot(bool, str, np.ndarray, list)
    def on_vision_finished(self, success, msg, display_img, boxes):
        self.btn_snapshot.setEnabled(True)
        if success:
            self.btn_snapshot.setText("返回\n实时")
            self.current_boxes = boxes
            self.is_zoomed_in = False

            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.result_pixmap_full = QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(self.result_pixmap_full)
            self.word_label.setText("Tap object!")
            self.feedback_text.append("<b style='color:#aae8aa;'>[Vision]</b> Success. Tap an object.")
        else:
            self.is_showing_result = False
            self.btn_snapshot.setText("拍摄\n分析")
            self.on_system_error(msg)

    def refresh_translation(self):
        if not self.current_word_en: return
        target_lang = self.lang_combo.currentText()
        word_data = KNOWLEDGE_BASE.get(self.current_word_en.lower(), {})

        if target_lang in word_data:
            trans, phonetic, example_text, example_trans = word_data[target_lang]
            self.current_target_text = trans
            chinese_translation = CHINESE_TRANSLATIONS.get(self.current_word_en.lower(), self.current_word_en)
            self.current_example_text = ""
            
            self.word_label.setText(trans)
            self.phonetic_label.setText(f"音标: {phonetic}")
            self.trans_label.setText(f"翻译: {chinese_translation}")
            self.example_label.setText("例句: DeepSeek 正在生成...")
            self.btn_play_example.setVisible(False)
            self.request_deepseek_example(trans, target_lang, example_text, example_trans)
        else:
            self.current_target_text = self.current_word_en
            self.current_example_text = ""
            self.word_label.setText(self.current_word_en.capitalize())
            self.phonetic_label.setText("音标: --")
            self.trans_label.setText(f"翻译: {CHINESE_TRANSLATIONS.get(self.current_word_en.lower(), self.current_word_en)}")
            self.example_label.setText("例句: --")
            self.btn_play_example.setVisible(False)

    def request_deepseek_example(self, target_word, target_lang, fallback_example, fallback_translation):
        self._example_request_token += 1
        token = self._example_request_token
        thread = DeepSeekExampleThread(
            self.current_word_en,
            target_word,
            target_lang,
            fallback_example,
            fallback_translation
        )
        self._deepseek_threads.append(thread)
        thread.finished_signal.connect(
            lambda success, example, translation, error, req_token=token:
                self.on_deepseek_example_finished(req_token, success, example, translation, error)
        )
        thread.finished.connect(lambda finished_thread=thread: self._deepseek_threads.remove(finished_thread)
                                if finished_thread in self._deepseek_threads else None)
        thread.start()

    def on_deepseek_example_finished(self, token, success, example, translation, error):
        if token != self._example_request_token:
            return
        self.current_example_text = example
        if example:
            self.example_label.setText(f"例句: {example}\n翻译: {translation}")
            self.btn_play_example.setVisible(True)
        else:
            self.example_label.setText("例句: --")
            self.btn_play_example.setVisible(False)
        if success:
            self.feedback_text.append("<b style='color:#a5d6ff;'>[DeepSeek]</b> Cloud example generated.")
        elif error:
            self.feedback_text.append(f"<b style='color:#ffb3c2;'>[DeepSeek]</b> Using local example fallback: {error}")

    @Slot(int, int)
    def on_video_clicked(self, x_click, y_click):
        if not self.is_showing_result or self.saved_raw_frame is None or not self.current_boxes: return
        if self.is_zoomed_in:
            self.video_label.setPixmap(self.result_pixmap_full)
            self.is_zoomed_in = False
            self.word_label.setText("Tap object!")
            self.phonetic_label.setText("音标: --")
            self.trans_label.setText("翻译: --")
            self.example_label.setText("例句: --")
            self.btn_play_example.setVisible(False)
            self.current_word_en = ""
            self.clear_waveforms()
            return

        label_w, label_h = self.video_label.width(), self.video_label.height()
        frame_h, frame_w = self.saved_raw_frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        scaled_w, scaled_h = int(frame_w * scale), int(frame_h * scale)
        offset_x, offset_y = (label_w - scaled_w) // 2, (label_h - scaled_h) // 2

        if offset_x <= x_click < offset_x + scaled_w and offset_y <= y_click < offset_y + scaled_h:
            real_x = int((x_click - offset_x) / scale)
            real_y = int((y_click - offset_y) / scale)

            for obj in self.current_boxes:
                class_id, score, x1, y1, x2, y2 = obj
                x1 = max(0, min(frame_w - 1, int(x1)))
                y1 = max(0, min(frame_h - 1, int(y1)))
                x2 = max(0, min(frame_w, int(x2)))
                y2 = max(0, min(frame_h, int(y2)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                if x1 <= real_x < x2 and y1 <= real_y < y2:
                    crop_img = self.saved_raw_frame[y1:y2, x1:x2]
                    if crop_img.size == 0 or crop_img.shape[0] < 2 or crop_img.shape[1] < 2:
                        continue

                    word = CLASSES.get(class_id, "Unknown")
                    self.current_word_en = word
                    self.refresh_translation()
                    self.clear_waveforms()

                    rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    qt_crop = QImage(rgb_crop.data, rgb_crop.shape[1], rgb_crop.shape[0],
                                     rgb_crop.shape[2] * rgb_crop.shape[1], QImage.Format_RGB888)
                    self.video_label.setPixmap(
                        QPixmap.fromImage(qt_crop).scaled(self.video_label.width(), self.video_label.height(),
                                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.is_zoomed_in = True
                    self.feedback_text.append(f"<b style='color:#aae8aa;'>[AI]</b> Target locked: <b>{word}</b>")
                    break

    def play_demo(self):
        if not self.current_target_text: return
        if hasattr(self, 'tts_thread') and self.tts_thread.isRunning():
            return

        btn = self.voice_group.checkedButton()
        if not btn:
            for b in self.voice_group.buttons():
                if b.isChecked(): btn = b; break
        if not btn: btn = self.voice_group.button(1)

        voice_style = btn.text() if btn else "女声"
        target_lang = self.lang_combo.currentText()

        self.clear_waveforms()
        self.btn_play.setEnabled(False)

        self.tts_thread = EdgeTTSThread(self.current_target_text, target_lang, voice_style, skip_wav_emit=False)
        self.tts_thread.finished_signal.connect(self.on_tts_finished)
        self.tts_thread.start()

    def play_example_tts(self):
        if not self.current_example_text: return
        if hasattr(self, 'example_tts_thread') and self.example_tts_thread.isRunning():
            return

        btn = self.voice_group.checkedButton()
        if not btn:
            for b in self.voice_group.buttons():
                if b.isChecked(): btn = b; break
        voice_style = btn.text() if btn else "女声"
        
        target_lang = self.lang_combo.currentText() 
        
        self.btn_play_example.setEnabled(False)
        self.example_tts_thread = EdgeTTSThread(self.current_example_text, target_lang, voice_style, skip_wav_emit=True)
        self.example_tts_thread.finished_signal.connect(lambda *_: self.btn_play_example.setEnabled(True))
        self.example_tts_thread.start()

    @Slot(bool, str, str)
    def on_tts_finished(self, success, msg, wav_path):
        self.btn_play.setEnabled(True)
        if success:
            self.std_wav_path = wav_path
            self.waveform_widget.update_waves(self.std_wav_path, None)
        else:
            self.feedback_text.append(f"<b style='color:#ff0844;'>[Network]</b> {msg}")

    def start_recording(self):
        if not self.current_target_text: return
        self.stop_tts_playback()
        self.stop_user_recording_playback()
        os.makedirs("./user_audio", exist_ok=True)
        timestamp = time.strftime("%H%M%S")
        self.audio_thread.filename = f"./user_audio/audio_{timestamp}.wav"
        self.audio_thread.start()
        self.feedback_text.append("<b style='color:#ffc4d0;'>[Mic]</b> Recording your voice...")

    def stop_tts_playback(self):
        for t_name in ['tts_thread', 'feedback_tts_thread', 'example_tts_thread']:
            if hasattr(self, t_name):
                t = getattr(self, t_name)
                if t and t.isRunning():
                    if hasattr(t, 'stop'):
                        t.stop()
                    t.wait(1200)

    def stop_user_recording_playback(self):
        if self.user_audio_play_process and self.user_audio_play_process.poll() is None:
            self.user_audio_play_process.terminate()
            try:
                self.user_audio_play_process.wait(timeout=1)
            except Exception:
                self.user_audio_play_process.kill()
        self.user_audio_play_process = None

    def play_user_recording(self):
        if not self.user_wav_path or not os.path.exists(self.user_wav_path):
            self.feedback_text.append("<b style='color:#ffb3c2;'>[Wave]</b> No user recording to play yet.")
            return

        self.stop_user_recording_playback()
        players = [
            ["aplay", "-q", self.user_wav_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.user_wav_path],
            ["mpg123", "-q", self.user_wav_path]
        ]
        for cmd in players:
            try:
                self.user_audio_play_process = subprocess.Popen(cmd)
                self.feedback_text.append("<b style='color:#a5d6ff;'>[Wave]</b> Playing your recording...")
                return
            except FileNotFoundError:
                continue
            except Exception as e:
                self.feedback_text.append(f"<b style='color:#ff0844;'>[Wave]</b> Playback failed: {e}")
                return

        self.feedback_text.append("<b style='color:#ff0844;'>[Wave]</b> No audio player found for WAV playback.")

    def stop_recording(self):
        if not self.audio_thread.isRunning(): return
        self.audio_thread.stop()

    def on_recording_finished(self, filename):
        if hasattr(self, 'eval_thread') and self.eval_thread.isRunning():
            self.eval_thread.wait()

        self.user_wav_path = filename
        self.waveform_widget.update_waves(self.std_wav_path, self.user_wav_path)

        self.feedback_text.append(f"<b style='color:#a5d6ff;'>[Eval]</b> Analyzing pronunciation...")
        self.eval_thread = EvaluationThread(filename, self.current_word_en)
        self.eval_thread.eval_finished_signal.connect(self.on_evaluation_finished)
        self.eval_thread.start()

    @Slot(int, str, list)
    def on_evaluation_finished(self, score, feedback, syllables_data):
        color = "#aae8aa" if score >= 70 else "#ffb3c2"
        self.feedback_text.append(f"<span style='color:{color}; font-size:16px;'><b>[Score] {score} / 100</b></span>")
        self.feedback_text.append("<b style='color:#c9d1d9;'>[AI]</b> DeepSeek 正在生成发音建议...")

        if self.current_word_en:
            word_lower = self.current_word_en.lower()
            if word_lower in VISUAL_SYLLABLES and syllables_data:
                visual_parts = list(VISUAL_SYLLABLES[word_lower])
                if visual_parts:
                    visual_parts[0] = visual_parts[0].capitalize()

                rich_text = ""
                for i, part in enumerate(visual_parts):
                    if i < len(syllables_data):
                        status = syllables_data[i].get("status", "improve")
                        if status == "excellent": part_color = "#4cf27b"
                        elif status == "pass": part_color = "#ffcc00"
                        else: part_color = "#ff0844"
                        rich_text += f"<span style='color:{part_color};'>{part}</span>"
                    else:
                        rich_text += part
                self.word_label.setText(rich_text)

        self.request_deepseek_advice(score, syllables_data, feedback)

    def request_deepseek_advice(self, score, syllables_data, original_feedback):
        thread = DeepSeekAdviceThread(self.current_word_en, score, syllables_data, original_feedback)
        self._deepseek_advice_threads.append(thread)
        thread.finished_signal.connect(self.on_deepseek_advice_finished)
        thread.finished.connect(lambda finished_thread=thread: self._deepseek_advice_threads.remove(finished_thread)
                                if finished_thread in self._deepseek_advice_threads else None)
        thread.start()

    def on_deepseek_advice_finished(self, success, advice):
        tag = "DeepSeekAdvice" if success else "AI"
        self.feedback_text.append(f"<b style='color:#c9d1d9;'>[{tag}]</b> {advice}")
        self.stop_tts_playback()
        self.feedback_tts_thread = EdgeTTSThread(advice, "中文", "")
        self.feedback_tts_thread.start()

    def on_system_error(self, err_msg):
        self.feedback_text.append(f"<b style='color:#ff0844;'>[Error]</b> {err_msg}")

    def closeEvent(self, event):
        self.stop_tts_playback()
        self.stop_user_recording_playback()
        self.camera_thread.stop()
        if self.audio_thread.isRunning():
            self.audio_thread.stop()

        for t_name in ['vision_thread', 'tts_thread', 'eval_thread', 'feedback_tts_thread', 'example_tts_thread']:
            if hasattr(self, t_name):
                t = getattr(self, t_name)
                if t.isRunning():
                    t.wait()
        for t in list(getattr(self, '_deepseek_threads', [])):
            if t.isRunning():
                t.wait()
        for t in list(getattr(self, '_deepseek_advice_threads', [])):
            if t.isRunning():
                t.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    font.insertSubstitution("Segoe UI", "PingFang SC")
    app.setFont(font)
    window = LanguageLearnerApp()
    window.showMaximized()
    sys.exit(app.exec())
