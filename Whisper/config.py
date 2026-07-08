# Whisper/config.py
# Configuration data for Whisper-based pronunciation scoring

TARGET_WORDS = [
    "glasses", "watch", "card", "apple",
    "key", "earphone", "bottle", "phone", "pen",
    "hello", "bicycle"
]

# 🌟 核心修改：为了前端实现字母级高亮，将原本的音节(Syllable)拆分为更细的“视觉发音块(Grapheme block)”
WORD_SYLLABLES = {
    # 比如 glasses 原本是2块，现在拆成 g-l-a-ss-es 5块
    "glasses": [["G"], ["L"], ["AE"], ["S"], ["IH", "Z"]],

    # watch 拆成 w-a-tch 3块
    "watch": [["W"], ["AA"], ["CH"]],

    # card 拆成 c-ar-d 3块 (完美解决读错成 carm 时，独立标红尾音 d 的需求)
    "card": [["K"], ["AA", "R"], ["D"]],

    # apple 拆成 a-pp-le 3块
    "apple": [["AE"], ["P"], ["AH", "L"]],

    # key 拆成 k-ey 2块
    "key": [["K"], ["IY"]],

    # earphone 拆成 ear-ph-o-ne 4块
    "earphone": [["IY", "R"], ["F"], ["OW"], ["N"]],

    # bottle 拆成 b-o-tt-le 4块
    "bottle": [["B"], ["AA"], ["T"], ["AH", "L"]],

    # phone 拆成 ph-o-ne 3块
    "phone": [["F"], ["OW"], ["N"]],

    # pen 拆成 p-e-n 3块
    "pen": [["P"], ["EH"], ["N"]],

    # hello 拆成 h-e-ll-o 4块
    "hello": [["HH"], ["AH"], ["L"], ["OW"]],

    # bicycle 拆成 b-i-c-y-cle 5块
    "bicycle": [["B"], ["AY"], ["S"], ["IH"], ["K", "AH", "L"]],
}

# Phoneme -> word mapping for fuzzy matching
PHONEME_TO_GRAPHEME = {
    "AE": ["a"], "AH": ["u", "o"], "AA": ["a", "o"], "IY": ["ee", "ea", "e"],
    "UH": ["oo", "u"], "UW": ["oo", "ou"], "EH": ["e"], "ER": ["er", "ir", "ur"],
    "AY": ["i", "y"], "AW": ["ou", "ow"], "OY": ["oi", "oy"], "OW": ["o", "oa"],
    "P": ["p"], "B": ["b"], "T": ["t"], "D": ["d"], "K": ["c", "k"],
    "G": ["g"], "M": ["m"], "N": ["n"], "NG": ["ng", "n"], "F": ["f"],
    "V": ["v"], "TH": ["th"], "DH": ["th", "d"], "S": ["s"], "Z": ["z"],
    "SH": ["sh", "ch"], "ZH": ["s", "ge"], "HH": ["h"], "R": ["r"], "L": ["l"],
    "W": ["w"], "Y": ["y"], "CH": ["ch", "t"], "JH": ["j", "g"],
}

# Scoring thresholds for syllable-level MFCC scoring
SCORE_SETTINGS = {
    "threshold_excellent": 75.0,
    "threshold_pass": 55.0,
    "score_min": 35.0,
    "score_max": 100.0,
    "sample_rate": 16000,

    # 🌟 核心提速参数：指定纯英文版极速模型
    "whisper_model": "tiny.en",

    # Route-A phonetic scoring settings
    "phoneme_penalty_scale": 0.35,
    "mismatch_penalty": 35.0,
    "duration_mismatch_weight": 0.20,
    "word_boundary_weight": 0.20,
    "use_phonetic_scoring": True,

    # 🌟 核心提速参数：彻底关闭外部环境的 DTW 召唤
    "use_word_timestamps": False,
}

CONFUABLE_PAIRS = [
    ("P", "B"), ("B", "P"), ("T", "D"), ("D", "T"),
    ("K", "G"), ("G", "K"), ("F", "V"), ("V", "F"),
    ("S", "Z"), ("Z", "S"), ("TH", "DH"), ("L", "R"),
    ("M", "N"), ("AE", "EH"), ("AE", "AA"),
    ("IY", "IH"), ("IH", "EY"), ("OW", "AW"),
]

PHONEME_DURATIONS = {
    "V": 0.08, "S": 0.10, "Z": 0.09, "L": 0.08,
    "R": 0.09, "N": 0.08, "M": 0.08, "DH": 0.07,
    "TH": 0.09, "F": 0.08, "B": 0.07, "D": 0.07,
    "G": 0.08, "K": 0.09, "P": 0.08, "T": 0.08,
    "HH": 0.06, "W": 0.08, "Y": 0.06,
    "JH": 0.10, "CH": 0.09, "SH": 0.11, "ZH": 0.09,
    "NG": 0.09,
    "AE": 0.13, "AH": 0.12, "AA": 0.14, "IY": 0.12,
    "UH": 0.11, "UW": 0.12, "EH": 0.12, "ER": 0.14,
    "AY": 0.18, "AW": 0.20, "OY": 0.18, "OW": 0.17,
}
