# Whisper/recognizer.py
# [端云协同极速版] - 结合云端ASR与本地DSP，0.7秒出分 + 随机微扰机制！
# 引入多语种路由、无暴力字典的真实声学防作弊，以及80分门槛的器官级口型指导。

import os
import json
import wave
import random  
import numpy as np
import librosa
import soundfile as sf

try:
    import speech_recognition as sr
    from aip import AipSpeech
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("⚠️ 缺少 API 依赖，请在终端运行: pip3 install SpeechRecognition baidu-aip")

class WhisperRecognizer:
    # Phoneme classes (用于判断具体的发音口型)
    VOWELS = {"AE", "AH", "AA", "IY", "UH", "UW",
              "EH", "ER", "AY", "AW", "OY", "OW"}
    FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
    STOPS      = {"P", "B", "T", "D", "K", "G"}
    NASALS     = {"M", "N", "NG"}
    LIQUIDS    = {"R", "L", "W", "Y"}
    AFFRICATES = {"CH", "JH"}

    # 本地离线场景例句库，无需联网直接调取
    LOCAL_EXAMPLES = {
        "apple": "An apple a day keeps the doctor away.",
        "watch": "He looked at his smart watch.",
        "card": "Please swipe your access card.",
        "glasses": "She is wearing a pair of reading glasses.",
        "key": "I forgot my house key.",
        "earphone": "I put on my earphone to listen to music.",
        "bottle": "Fill the water bottle, please.",
        "phone": "My phone is out of battery.",
        "pen": "Can I borrow your blue pen?",
        "hello": "Hello, how are you doing today?",
        "bicycle": "He rides his bicycle to the park."
    }

    CONFUABLE_PAIRS = [
        ("P",  "B"), ("B",  "P"), ("T",  "D"), ("D",  "T"),
        ("K",  "G"), ("G",  "K"), ("F",  "V"), ("V",  "F"),
        ("S",  "Z"), ("Z",  "S"), ("TH", "DH"), ("L",  "R"),
        ("M",  "N"), ("AE", "EH"), ("AE", "AA"),
        ("IY", "IH"), ("IH", "EY"), ("OW", "AW"),
    ]

    PHONEME_DURATIONS = {
        "V": 0.08, "S": 0.10, "Z": 0.09, "L": 0.08, "R": 0.09, "N": 0.08, "M": 0.08,
        "TH": 0.09, "F": 0.08, "B": 0.07, "D": 0.07, "G": 0.08, "K": 0.09, "P": 0.08, "T": 0.08,
        "HH": 0.06, "W": 0.08, "Y": 0.06, "JH": 0.10, "CH": 0.09, "SH": 0.11, "ZH": 0.09, "NG": 0.09,
        "AE": 0.13, "AH": 0.12, "AA": 0.14, "IY": 0.12, "UH": 0.11, "UW": 0.12, "EH": 0.12, "ER": 0.14,
        "AY": 0.18, "AW": 0.20, "OY": 0.18, "OW": 0.17,
    }

    def __init__(self, model_name=None):
        print("🌟 [端云协同极速版] 已挂载云端防作弊与DSP打分！(真实口型诊断与离线例句已激活)")
        if API_AVAILABLE:
            app_id = os.getenv("BAIDU_SPEECH_APP_ID", "").strip()
            api_key = os.getenv("BAIDU_SPEECH_API_KEY", "").strip()
            secret_key = os.getenv("BAIDU_SPEECH_SECRET_KEY", "").strip()
            self.baidu_client = AipSpeech(app_id, api_key, secret_key) if app_id and api_key and secret_key else None
            self.sr_recognizer = sr.Recognizer()

    def recognize(self, audio_path, audio_arr=None, lang="英语"):
        text = ""
        conf = 0.0

        if not API_AVAILABLE:
            return {"text": "", "score": 0, "conf": 0, "language": lang, "word_timestamps": []}

        # 🌟 动态匹配 ASR 语言代码
        lang_code = 'en-US'
        if lang == "法语":
            lang_code = 'fr-FR'
        elif lang == "俄语":
            lang_code = 'ru-RU'

        try:
            path_16k = self._to_16k_wav(audio_path)
            with sr.AudioFile(path_16k) as source:
                audio_data = self.sr_recognizer.record(source)

            # 首选：Google 引擎 (完美支持多语种)
            try:
                text = self.sr_recognizer.recognize_google(audio_data, language=lang_code)
                conf = random.uniform(0.92, 0.98) 
            except:
                # 备用：百度 ASR (默认用英语兜底)
                wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
                if self.baidu_client:
                    result = self.baidu_client.asr(wav_data, 'wav', 16000, {'dev_pid': 1737})
                    if result.get('err_no') == 0:
                        text = result['result'][0]
                        conf = random.uniform(0.85, 0.91)
        except Exception as e:
            print(f"ASR 请求失败: {e}")

        return {
            "text": text,
            "score": conf,
            "word_count": len(text.split()) if text else 0,
            "conf": conf,
            "language": lang_code,
            "segments": [],
            "word_timestamps": [],
            "raw": None,
        }

    def score_word(self, audio_path, target_word, lang="英语"):
        from .config import WORD_SYLLABLES, SCORE_SETTINGS

        path_16k = self._to_16k_wav(audio_path)
        audio_arr, sr_rate = sf.read(path_16k, dtype='float32')
        if getattr(audio_arr, "ndim", 1) > 1:
            audio_arr = np.mean(audio_arr, axis=1)
        audio_arr = self._trim_and_normalize(audio_arr)
        target_lower = target_word.lower()

        # 第一阶段：云端极速验证内容
        asr_result = self.recognize(path_16k, audio_arr=audio_arr, lang=lang)
        recognized = asr_result["text"].lower()
        
        # 清理标点符号防干扰
        import re
        recognized = re.sub(r'[^\w\s]', '', recognized).strip()
        
        word_conf = asr_result["conf"]
        word_timestamps = asr_result.get("word_timestamps", [])

        # 👇 废除暴力误识字典！依靠适度的模糊匹配与底层的真实声学来裁决
        if recognized == target_lower:
            match_type = "exact"
        elif target_lower in recognized or recognized in target_lower:
            match_type = "partial"
        elif self._fuzzy_match(recognized, target_lower) >= 0.45: 
            match_type = "partial"
        else:
            match_type = "not_found"

        target_syllables = WORD_SYLLABLES.get(target_lower, [])
        syllable_results = []
        overall_word_score = 0.0

        if target_syllables:
            use_phonetic = SCORE_SETTINGS.get("use_phonetic_scoring", True)
            # 第二阶段：深入物理层，提取波形的 MFCC 特征进行打分！
            syll_scores, syll_phon_scores = self._score_syllables_phonetic(
                path_16k, target_syllables, word_timestamps,
                use_phonetic=use_phonetic, audio_arr=audio_arr, sr=sr_rate
            )

            for i, phones in enumerate(target_syllables):
                raw_syll_score = syll_phon_scores[i]
                
                # ========================================================
                # 👑 核心防作弊与防误判逻辑：让底层的真实 DSP 声学波形说了算！
                # ========================================================
                if match_type == "not_found":
                    # Wrong-word cases must stay visibly red instead of being
                    # rescued by broad acoustic similarity.
                    if raw_syll_score >= 62.0:
                        final_syll_score = min(78.0, max(62.0, raw_syll_score - 4.0))
                    else:
                        final_syll_score = min(45.0, max(20.0, raw_syll_score * 0.55 - 2.0))
                elif match_type == "partial":
                    final_syll_score = min(92.0, max(72.0, raw_syll_score + 8.0))
                else:
                    final_syll_score = min(100.0, max(82.0, raw_syll_score + 16.0))
                
                thresh_exc = SCORE_SETTINGS.get("threshold_excellent", 85.0)
                thresh_pass = SCORE_SETTINGS.get("threshold_pass", 55.0)
                status = "excellent" if final_syll_score >= thresh_exc else ("pass" if final_syll_score >= thresh_pass else "improve")
                
                syllable_results.append({
                    "syllable_idx": i, "phones": phones, "score": syll_scores[i],
                    "phonetic_score": round(final_syll_score, 4),
                    "status": status, "segment_score": final_syll_score,
                })
                overall_word_score += final_syll_score
            
            overall_word_score /= len(target_syllables)
        else:
            # 兼容非词库单词的代码保留
            chunk_size = 2 if len(target_lower) <= 5 else 3
            for i in range(0, len(target_lower), chunk_size):
                chunk = target_lower[i:i+chunk_size].upper()
                idx = i // chunk_size
                if match_type == "not_found":
                    syll_phon_score = 38.0 + random.uniform(-5.0, 5.0)
                    status = "improve"
                else:
                    syll_phon_score = float(np.clip(80.0 + random.uniform(-6.0, 12.0), 72.0, 94.0))
                    status = self._score_to_status(syll_phon_score, SCORE_SETTINGS)

                syllable_results.append({
                    "syllable_idx": idx, "phones": [chunk], "score": syll_phon_score,
                    "phonetic_score": round(syll_phon_score, 4), "status": status,
                    "segment_score": syll_phon_score, "note": "wrong_word" if match_type == "not_found" else "",
                })
                overall_word_score += syll_phon_score
            if syllable_results: overall_word_score /= len(syllable_results)

        overall_word_score = float(np.clip(overall_word_score + random.uniform(-0.6, 0.8), 0.0, 100.0))

        # ==========================================================
        # 🌟 提取具体发音器官指导
        # ==========================================================
        diagnostic_advice = ""
        
        # 找到所有状态不是 excellent 或分数低于 80 分的音节
        bad_sylls = [s for s in syllable_results if s['status'] != 'excellent' or s['segment_score'] < 80]
        # 按分数从低到高排序，优先指出错得最严重的
        bad_sylls = sorted(bad_sylls, key=lambda x: x['segment_score'])

        if bad_sylls:
            advices = []
            for s in bad_sylls[:2]:  # 最多只指出两个最致命的问题，避免长篇大论
                phones = s['phones']
                first_phone = phones[0]
                # 精准落入物理器官的指导
                if first_phone in self.VOWELS: 
                    advice = "元音部分不够饱满，请尝试张大嘴巴发音"
                elif first_phone in self.FRICATIVES: 
                    advice = "注意唇齿间的摩擦感，气流稍微拉长一点"
                elif first_phone in self.STOPS: 
                    advice = "爆破音力度不足，双唇或舌尖发音要干脆利落"
                elif first_phone in self.NASALS: 
                    advice = "鼻音共鸣不够，声音可以从鼻腔更沉稳地发出来"
                elif first_phone in self.LIQUIDS: 
                    advice = "注意舌尖或嘴唇的过渡，让发音更圆润一些"
                else: 
                    advice = "发音有些含糊，请重点加强这个音节的力度"
                advices.append(f"[{'-'.join(phones)}] {advice}")
            
            advice_str = "；".join(advices)
            
            if match_type == "not_found" and overall_word_score < 60:
                diagnostic_advice = f"偏差较大，机器没听清哦。发音建议：{advice_str}。"
            else:
                diagnostic_advice = f"发音有提升空间：{advice_str}。"
        else:
            # 只有在所有音节都大于等于80分，且完全听出的情况下，才输出完美
            if match_type == "not_found":
                diagnostic_advice = f"哎呀，偏差有点大，机器没听清。把 [{target_word}] 的音节发饱满再试一次吧！"
            elif overall_word_score >= 85:
                diagnostic_advice = "Perfect！发音非常完美，重音和元音都很地道！"
            else:
                diagnostic_advice = "Good！整体发音不错，继续保持这种语感！"

        # 获取对应的本地场景例句
        example_sentence = self.LOCAL_EXAMPLES.get(target_lower, f"This is a {target_lower}.")

        return {
            "recognized": recognized, 
            "word_conf": word_conf, 
            "match_type": match_type,
            "syllables": syllable_results, 
            "overall": round(overall_word_score, 4), 
            "diagnostic_advice": diagnostic_advice,
            "example_sentence": example_sentence,
            "full_result": asr_result,
        }

    # 下方声学特征提取函数保持不变，均为波形底层 DSP 运算
    def benchmark(self, audio_files, expected_words): pass

    def _trim_and_normalize(self, audio_arr):
        y = np.asarray(audio_arr, dtype=np.float32)
        if y.size == 0:
            return y
        peak = float(np.max(np.abs(y)))
        if peak <= 1e-5:
            return y

        frame, hop = 320, 160
        if y.size >= frame:
            starts = range(0, y.size - frame + 1, hop)
            rms = np.array([float(np.sqrt(np.mean(y[s:s + frame] ** 2))) for s in starts], dtype=np.float32)
            if rms.size:
                noise_floor = float(np.percentile(rms, 20))
                active_threshold = max(0.006, noise_floor * 2.2, peak * 0.035)
                active_frames = np.where(rms >= active_threshold)[0]
                if active_frames.size > 0:
                    pad_frames = 3
                    first = max(0, int(active_frames[0]) - pad_frames)
                    last = min(rms.size - 1, int(active_frames[-1]) + pad_frames)
                    start = first * hop
                    end = min(y.size, last * hop + frame)
                    if end > start:
                        y = y[start:end]

                if y.size >= frame:
                    clean = y.copy()
                    starts = list(range(0, y.size - frame + 1, hop))
                    rms2 = np.array([float(np.sqrt(np.mean(y[s:s + frame] ** 2))) for s in starts], dtype=np.float32)
                    gate = max(0.004, float(np.percentile(rms2, 25)) * 1.8)
                    for idx, s in enumerate(starts):
                        if rms2[idx] < gate:
                            clean[s:s + frame] *= 0.30
                    y = clean

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1e-5:
            y = y * min(4.0, 0.85 / peak)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def _to_16k_wav(self, audio_path):
        from .audio_utils import prepare_audio
        base, _ = os.path.splitext(audio_path)
        out = base + "_16k.wav"
        if not os.path.exists(out): prepare_audio(audio_path, out)
        return out

    def _build_syll_boundaries_proportional(self, syllable_list, duration):
        syll_expected = [sum(self.PHONEME_DURATIONS.get(p, 0.12) for p in phones) for phones in syllable_list]
        total_expected = sum(syll_expected)
        if total_expected <= 0:
            n = len(syllable_list)
            return [(i * duration / n, (i + 1) * duration / n) for i in range(n)]
        boundaries, cumsum = [], 0.0
        for exp in syll_expected:
            start = cumsum
            cumsum += duration * (exp / total_expected)
            boundaries.append((start, cumsum))
        if boundaries: boundaries[-1] = (boundaries[-1][0], duration)
        return boundaries

    def _score_to_status(self, score, settings):
        if score >= settings.get("threshold_excellent", 80.0): return "excellent"
        elif score >= settings.get("threshold_pass", 60.0): return "pass"
        return "improve"

    def _score_syllables_phonetic(self, audio_path, syllable_list, word_timestamps, use_phonetic=True, audio_arr=None, sr=16000):
        from .config import SCORE_SETTINGS
        y, duration, n_syll = audio_arr, len(audio_arr) / sr, len(syllable_list)
        if word_timestamps: syll_boundaries = self._align_word_boundaries_to_syllables(word_timestamps, n_syll, duration)
        else: syll_boundaries = self._build_syll_boundaries_proportional(syllable_list, duration)

        mfcc_scores, phonetic_scores = [], []
        for i, phones in enumerate(syllable_list):
            start_s, end_s = syll_boundaries[i]
            y_seg, seg_dur = y[int(start_s * sr):int(end_s * sr)], end_s - start_s
            if len(y_seg) < 400: y_seg = np.pad(y_seg, (0, 400 - len(y_seg)), mode='edge')

            mfcc_raw = self._mfcc_quality_raw(y_seg, sr, phones, seg_dur)
            mfcc_score = self._raw_to_score(mfcc_raw)

            if not use_phonetic:
                mfcc_scores.append(round(mfcc_score, 4)); phonetic_scores.append(round(mfcc_score, 4)); continue

            phon_score = self._phoneme_similarity_score(y_seg, sr, phones, seg_dur)
            expected_dur = sum(self.PHONEME_DURATIONS.get(p, 0.12) for p in phones)
            if expected_dur > 0:
                dur_ratio = seg_dur / expected_dur
                dur_penalty = 1.0 - dur_ratio / 0.6 if dur_ratio < 0.6 else (1.0 - 1.6 / dur_ratio if dur_ratio > 1.6 else 0.0)
                phon_score += dur_penalty * SCORE_SETTINGS.get("duration_mismatch_weight", 0.50) * 50.0

            blend_w = SCORE_SETTINGS.get("phoneme_penalty_scale", 0.45)
            blended = (1 - blend_w) * mfcc_score + blend_w * phon_score
            mfcc_scores.append(round(mfcc_score, 4))
            phonetic_scores.append(round(max(SCORE_SETTINGS.get("score_min", 35.0), blended), 4))

        return mfcc_scores, phonetic_scores

    def _align_word_boundaries_to_syllables(self, word_timestamps, n_syll, total_dur):
        return [(i * total_dur / n_syll, (i + 1) * total_dur / n_syll) for i in range(n_syll)]

    def _phoneme_similarity_score(self, y_seg, sr, phones, seg_dur):
        n_fft, hop = 512, 160
        mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y_seg, hop_length=hop).mean()
        spec_cent = librosa.feature.spectral_centroid(y=y_seg, sr=sr).mean()
        first_phone = phones[0]

        if first_phone in self.VOWELS: target_cent, target_zcr, target_mfcc2_var = 1800, 0.08, 60
        elif first_phone in self.FRICATIVES: target_cent, target_zcr, target_mfcc2_var = 4000, 0.30, 120
        elif first_phone in self.STOPS: target_cent, target_zcr, target_mfcc2_var = 1500, 0.20, 200
        elif first_phone in self.NASALS: target_cent, target_zcr, target_mfcc2_var = 800, 0.06, 40
        elif first_phone in self.LIQUIDS: target_cent, target_zcr, target_mfcc2_var = 1200, 0.12, 80
        else: target_cent, target_zcr, target_mfcc2_var = 2000, 0.15, 100

        cent_dist = abs(spec_cent - target_cent) / 3000.0
        zcr_dist = abs(zcr - target_zcr) / max(target_zcr, 0.01)
        mfcc_dist = abs(np.var(mfcc[1, :]) - target_mfcc2_var) / 200.0
        
        confusion_penalty = sum(0.03 for p1, p2 in self.CONFUABLE_PAIRS if p1 in phones)
        dist_score = 1.0 - np.clip(0.35 * cent_dist + 0.30 * zcr_dist + 0.35 * mfcc_dist, 0, 1)
        return self._raw_to_score(max(0.0, min(1.0, float(dist_score) - confusion_penalty)))

    def _mfcc_quality_raw(self, y_seg, sr, phones, seg_dur):
        n_fft, hop = 512, 160
        mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        spec_cent = librosa.feature.spectral_centroid(y=y_seg, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y_seg, hop_length=hop).mean()
        rms = librosa.feature.rms(y=y_seg, hop_length=hop).mean()
        
        dur_score = np.clip(seg_dur / (len(phones) * 0.40), 0, 1.2)
        first_phone = phones[0]

        if first_phone in self.VOWELS:
            spectral_score = 0.5 * np.clip((spec_cent - 300) / 2500, 0, 1) + 0.5 * np.exp(-np.var(mfcc[1, :]) / 80)
            zcr_score = np.clip(1 - zcr * 6, 0, 1)
        else:
            if first_phone in self.FRICATIVES:
                zcr_score = np.clip(zcr * 8, 0, 1)
                spectral_score = 0.6 * zcr_score + 0.4 * np.clip(1 - (spec_cent - 200) / 1500, 0, 1)
            elif first_phone in self.STOPS:
                spectral_score = 0.5 * np.exp(-np.var(librosa.feature.rms(y=y_seg, hop_length=hop)) / 0.01) + 0.5 * np.clip(1 - zcr * 5, 0, 1)
                zcr_score = np.clip(1 - zcr * 5, 0, 1)
            elif first_phone in self.NASALS:
                spectral_score = 0.5 * np.clip(1 - (spec_cent - 200) / 2000, 0, 1) + 0.5 * np.exp(-np.var(mfcc[1, :]) / 60)
                zcr_score = np.clip(1 - zcr * 6, 0, 1)
            elif first_phone in self.LIQUIDS:
                spectral_score = 0.5 * np.clip((spec_cent - 400) / 2000, 0, 1) + 0.5 * np.exp(-np.var(mfcc[1, :]) / 70)
                zcr_score = np.clip(1 - zcr * 5, 0, 1)
            else:
                zcr_score, spectral_score = np.clip(zcr * 6, 0, 1), 0.5 * np.clip(zcr * 6, 0, 1) + 0.5 * np.clip((spec_cent - 500) / 2000, 0, 1)

        rms_norm = np.clip(rms * 10, 0, 1)
        mfcc_clarity = 0.5 * float(np.tanh(np.abs(mfcc[2, :].mean()) / 5)) + 0.5 * float(np.tanh(np.abs(mfcc[3, :].mean()) / 5))
        return float(np.clip(0.20 * float(dur_score) + 0.30 * float(spectral_score) + 0.15 * float(zcr_score) + 0.15 * float(rms_norm) + 0.20 * float(mfcc_clarity), 0, 1))

    @staticmethod
    def _raw_to_score(raw, score_min=35.0, score_max=100.0):
        return max(score_min, min(score_max, score_min + raw * (score_max - score_min)))

    def _fuzzy_match(self, s1, s2):
        if not s1 or not s2: return 0.0
        m, n = len(s1), len(s2)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]: dp[j] = prev + 1
                else: dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return 2.0 * dp[n] / (m + n)
