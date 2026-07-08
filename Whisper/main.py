# Whisper/main.py
# Entry point for Whisper-based pronunciation scoring
# Supports interactive mode and batch benchmark mode

import os
import sys
import glob


def get_whisper_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_audio_path(raw_path):
    """Resolve a user-provided audio path to an absolute path."""
    path = raw_path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if os.path.isabs(path):
        return path

    whisper_root = get_whisper_root()
    # Try relative to GOPV5 root
    for base in [whisper_root, os.getcwd()]:
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Try all wav files matching the stem
    base_name = os.path.basename(path)
    stem, _ = os.path.splitext(base_name)
    for root, _, files in os.walk(whisper_root):
        for f in files:
            fstem, fext = os.path.splitext(f)
            if stem.lower() in fstem.lower() and fext.lower() == '.wav':
                return os.path.join(root, f)
        for f in files:
            fstem, fext = os.path.splitext(f)
            if fstem.lower() == stem.lower() and fext.lower() == '.wav':
                return os.path.join(root, f)

    return raw_path


def print_report(word, result):
    """Print a formatted pronunciation report."""
    from .config import SCORE_SETTINGS

    print()
    print(f"=== {word.upper()} 发音报告 ===")
    print(f"识别结果: \"{result['full_result']['text']}\"")
    conf = result['word_conf']
    conf_str = f"{conf:.2%}" if conf > 0 else "N/A"
    print(f"置信度:   {conf_str}")
    print(f"匹配类型: {result['match_type']}")

    if result['syllables']:
        print(f"音节评分:")
        for s in result['syllables']:
            phone_str = '-'.join(s['phones'])
            # Use phonetic_score (Route-A) if available, fall back to score (MFCC-only)
            score = s.get('phonetic_score', s['score'])
            thresh_exc = SCORE_SETTINGS['threshold_excellent']
            thresh_pass = SCORE_SETTINGS['threshold_pass']
            if score >= thresh_exc:
                status = "[EXCELLENT]"
            elif score >= thresh_pass:
                status = "[PASS]"
            else:
                status = "[IMPROVE]"
            note = f"  ← {s['note']}" if s.get('note') else ""
            print(f"  音节 {s['syllable_idx']+1} ({phone_str}): {score:.2f} {status}{note}")

    print(f"综合得分: {result['overall']:.2f}")
    print()


def print_benchmark_report(bench):
    """Print a formatted benchmark report with per-word breakdowns."""
    print()
    print("=" * 65)
    print(f"  Whisper 识别准确率测试报告")
    print("=" * 65)
    print(f"  总样本数:  {bench['total']}")
    print(f"  完全正确:  {bench['correct']}  ({bench['correct']/bench['total']*100:.1f}%)")
    print(f"  部分匹配:  {bench['partial']}  ({bench['partial']/bench['total']*100:.1f}%)")
    print(f"  识别错误:  {bench['wrong']}  ({bench['wrong']/bench['total']*100:.1f}%)")
    print(f"  单词准确率: {bench['accuracy']:.1f}%")
    print("=" * 65)

    # Per-word breakdown
    if bench.get('by_word'):
        print()
        print(f"  {'单词':<12} {'正确':<8} {'部分':<8} {'错误':<8} {'准确率':<8}")
        print(f"  {'-'*48}")
        for word, stats in sorted(bench['by_word'].items()):
            total = stats['correct'] + stats['partial'] + stats['wrong']
            acc = stats['correct'] / total * 100 if total > 0 else 0
            print(f"  {word:<12} {stats['correct']:<8} {stats['partial']:<8} "
                  f"{stats['wrong']:<8} {acc:.1f}%")

    print()
    print(f"  {'文件':<45} {'期望':<12} {'识别结果':<18} {'状态'}")
    print(f"  {'-'*100}")
    for d in bench['details']:
        text = d['recognized'][:16] + '..' if len(d['recognized']) > 18 else d['recognized']
        conf = d['conf']
        conf_str = f"{conf:.2%}" if conf > 0 else "N/A"
        print(f"  {d['file']:<45} {d['expected']:<12} {text:<18} {d['label']}")
    print()


def interactive_mode():
    """Interactive mode: prompt user for word and audio file."""
    from .recognizer import WhisperRecognizer

    recognizer = WhisperRecognizer()

    while True:
        print()
        word = input("请输入正在练习的单词 (输入 q 退出): ").strip().lower()
        if word in ('q', 'quit', 'exit'):
            break
        if not word:
            continue

        audio_file = input("请输入录音文件名 (例如 apple_1.wav): ").strip()
        if not audio_file:
            continue

        resolved = resolve_audio_path(audio_file)
        print(f"[INFO] 正在处理: {resolved}")

        if not os.path.exists(resolved):
            print(f"[ERROR] 文件不存在: {resolved}")
            stem = os.path.splitext(os.path.basename(resolved))[0]
            print(f"[HINT] 尝试搜索包含 '{stem}' 的文件 ...")
            whisper_root = get_whisper_root()
            for root, _, files in os.walk(whisper_root):
                for f in files:
                    if stem.lower() in f.lower() and f.endswith('.wav'):
                        print(f"  -> 找到: {os.path.join(root, f)}")
            continue

        try:
            result = recognizer.score_word(resolved, word)
            print_report(word, result)
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


def benchmark_mode():
    """Batch benchmark mode: recursively scan test_records subfolders and report accuracy."""
    import re
    from .recognizer import WhisperRecognizer
    from .config import TARGET_WORDS

    recognizer = WhisperRecognizer()

    # Scan for test audio recursively
    whisper_root = get_whisper_root()
    test_records_dir = os.path.join(whisper_root, "test_records")

    audio_files = []
    expected_words = []

    if os.path.isdir(test_records_dir):
        for dirpath, dirnames, filenames in os.walk(test_records_dir):
            for fname in filenames:
                if not fname.lower().endswith('.wav'):
                    continue
                if '_16k.wav' in fname:
                    continue

                fpath = os.path.join(dirpath, fname)

                rel_dir = os.path.relpath(dirpath, test_records_dir)
                folder_parts = [p for p in rel_dir.split(os.sep) if p != '.']
                expected = None

                # Try folder names first
                for part in reversed(folder_parts):
                    part_clean = re.sub(r'[_\s\-]+', '', part.lower())
                    for tw in TARGET_WORDS:
                        tw_clean = re.sub(r'[_\s\-]+', '', tw.lower())
                        if part_clean == tw_clean:
                            expected = tw
                            break
                    if expected:
                        break

                # Priority 2: filename contains a target word
                if not expected:
                    fname_clean = re.sub(r'[_\s\-\(\)]+', '', fname.lower())
                    fname_stripped = re.sub(r'\s*\(\d+\)\s*$', '',
                                            os.path.splitext(fname)[0])
                    fname_lower = fname_stripped.lower()
                    for tw in TARGET_WORDS:
                        if tw in fname_lower:
                            expected = tw
                            break
                        if any(p.startswith(tw) or tw.startswith(p)
                               for p in re.split(r'[_\s\-]+', fname_lower)):
                            expected = tw
                            break

                # Priority 3: fuzzy match on filename
                if not expected:
                    fname_stripped2 = re.sub(r'\s*\d+\s*$', '',
                                             re.sub(r'\s*\(\d+\)\s*$', '',
                                                    os.path.splitext(fname)[0])).lower()
                    for tw in TARGET_WORDS:
                        if any(part.startswith(tw) or tw.startswith(part)
                               for part in re.split(r'[_\s\-\(\)]+', fname_stripped2)):
                            expected = tw
                            break

                if expected:
                    audio_files.append(fpath)
                    expected_words.append(expected)

    if not audio_files:
        print("[ERROR] 未找到任何测试音频文件。")
        print(f"[HINT] 请将测试音频放在以下目录结构中:")
        print(f"  test_records/")
        for tw in TARGET_WORDS:
            print(f"    {tw}/")
            print(f"      {tw}_correct_1.wav")
            print(f"      {tw}_wrong_1.wav")
        return

    print(f"[INFO] 找到 {len(audio_files)} 个测试音频文件")

    # Run benchmark
    bench = recognizer.benchmark(audio_files, expected_words)

    # Build per-word breakdown
    by_word = {}
    for d in bench['details']:
        w = d['expected']
        if w not in by_word:
            by_word[w] = {'correct': 0, 'partial': 0, 'wrong': 0}
        match_type = d['type']
        count_key = 'correct' if match_type == 'exact' else \
                    'partial' if match_type == 'partial' else 'wrong'
        by_word[w][count_key] += 1
    bench['by_word'] = by_word

    print_benchmark_report(bench)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
