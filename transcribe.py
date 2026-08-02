"""
transcribe.py — Транскрибация аудио через faster-whisper.

Работает с одним файлом (обычно mp3), который передаёт app.py.
Длинные файлы (> 30 мин) режутся ffmpeg на 15-минутные WAV-чанки,
транскрибируются по частям и склеиваются — чтобы не держать всё в памяти.

Результат:
  <output_dir>/<stem>.txt          — чистый текст
  --progress-file                  — JSON с прогрессом (читает app.py)

Запуск:
  python transcribe.py audio.mp3 -o transcripts/ --progress-file p.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import WhisperModel

# ─── Конфиг ────────────────────────────────────────────────────────

CHUNK_SECONDS = 900         # 15 минут на чанк (макс аудио в память за раз)
LONG_FILE_THRESHOLD = 1800  # 30 минут — порог, после которого режем на чанки


def write_txt(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"  ✍️  {path.name}")


# ─── Прогресс ──────────────────────────────────────────────────────


def _write_progress(progress_file: Path, data: dict):
    tmp = progress_file.with_name(progress_file.name + ".tmp")
    data["_ts"] = time.time()
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(progress_file)


# ─── Вспомогательные ffmpeg ────────────────────────────────────────


def ffmpeg_run(cmd: list[str], desc: str = ""):
    print(f"  🎞️  {desc or ' '.join(cmd[:3])}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ❌ ffmpeg ошибка: {r.stderr[-300:]}")
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-200:]}")


def get_audio_duration(audio_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def extract_audio_wav(filepath: Path, dst: Path):
    """Извлекает/перекодирует дорожку в 16 кГц моно WAV."""
    ffmpeg_run([
        "ffmpeg", "-y", "-i", str(filepath),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(dst),
    ], desc=f"Извлечение аудио: {filepath.name}")


# ─── Потоковая транскрибация (весь файл целиком) ────────────────────


def transcribe_stream(
    model: WhisperModel, filepath: Path, language: str, vad: bool,
    progress_file: Path | None, t0: float, text_parts: list,
):
    segments, info = model.transcribe(
        str(filepath),
        language=language,
        beam_size=5,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
    )

    total_duration = info.duration
    print(f"  📊  Длительность: {total_duration:.0f} с ({total_duration / 3600:.1f} ч)")
    print(f"      Язык:         {info.language} (prob: {info.language_probability:.2f})")

    seg_count = 0
    last_progress_save = 0.0

    if progress_file:
        _write_progress(progress_file, {
            "status": "processing", "progress_pct": 0,
            "total_duration_sec": round(total_duration, 1),
            "processed_sec": 0, "elapsed_sec": 0, "segments_count": 0, "text": "",
        })

    for seg in segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue
        text_parts.append(seg_text + " ")
        seg_count += 1

        now = time.time()
        processed_sec = seg.end
        progress_pct = min((processed_sec / total_duration) * 100, 99.9) if total_duration else 0

        if progress_file and now - last_progress_save > 3:
            _write_progress(progress_file, {
                "status": "processing", "progress_pct": round(progress_pct, 1),
                "total_duration_sec": round(total_duration, 1),
                "processed_sec": round(processed_sec, 1),
                "elapsed_sec": round(now - t0, 1),
                "segments_count": seg_count, "text": seg_text,
            })
            last_progress_save = now

    return seg_count


# ─── Чанковая транскрибация (длинные файлы) ─────────────────────────


def transcribe_chunked(
    model: WhisperModel, filepath: Path, language: str, vad: bool,
    progress_file: Path | None, t0: float, text_parts: list,
):
    """Режем аудио на WAV-чанки, транскрибируем по одному, склеиваем."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))

    try:
        audio_wav = tmp_dir / "audio.wav"
        extract_audio_wav(filepath, audio_wav)

        total_duration = get_audio_duration(audio_wav)
        total_chunks = max(1, int(total_duration // CHUNK_SECONDS) +
                           (1 if total_duration % CHUNK_SECONDS > 0 else 0))
        print(f"  📊  Длительность: {total_duration:.0f} с ({total_duration/3600:.1f} ч)")
        print(f"  🧩  Чанков: {total_chunks} по {CHUNK_SECONDS} с")

        chunk_pattern = str(tmp_dir / "chunk_%04d.wav")
        ffmpeg_run([
            "ffmpeg", "-y", "-i", str(audio_wav),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            chunk_pattern,
        ], desc="Нарезка на чанки")

        chunk_files = sorted(tmp_dir.glob("chunk_*.wav"))
        if not chunk_files:
            chunk_files = [audio_wav]

        print(f"  🧩  Фактических чанков: {len(chunk_files)}")

        all_seg_count = 0
        processed_chunk_sec = 0.0

        for ci, chunk_path in enumerate(chunk_files):
            chunk_dur = get_audio_duration(chunk_path)
            print(f"\n  [{ci+1}/{len(chunk_files)}] 🎯 {chunk_path.name} ({chunk_dur:.0f} с)")

            segments, info = model.transcribe(
                str(chunk_path),
                language=language,
                beam_size=5,
                vad_filter=vad,
                vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
            )

            last_seg_text = ""
            seg_count = 0
            last_save = 0.0

            for seg in segments:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue
                text_parts.append(seg_text + " ")
                last_seg_text = seg_text
                seg_count += 1
                all_seg_count += 1

                now = time.time()
                chunk_progress = seg.end / chunk_dur if chunk_dur > 0 else 1
                overall_pct = ((ci + chunk_progress) / total_chunks) * 100
                processed_total_sec = processed_chunk_sec + seg.end

                if progress_file and now - last_save > 3:
                    _write_progress(progress_file, {
                        "status": "processing", "progress_pct": round(overall_pct, 1),
                        "total_duration_sec": round(total_duration, 1),
                        "processed_sec": round(processed_total_sec, 1),
                        "elapsed_sec": round(now - t0, 1),
                        "segments_count": all_seg_count,
                        "chunk": f"{ci+1}/{len(chunk_files)}", "text": seg_text,
                    })
                    last_save = now

            processed_chunk_sec += chunk_dur

            if progress_file:
                _write_progress(progress_file, {
                    "status": "processing",
                    "progress_pct": round(((ci + 1) / total_chunks) * 100, 1),
                    "total_duration_sec": round(total_duration, 1),
                    "processed_sec": round(processed_chunk_sec, 1),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "segments_count": all_seg_count,
                    "chunk": f"{ci+1}/{len(chunk_files)}",
                    "text": last_seg_text[:200],
                })

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── Основная логика ────────────────────────────────────────────────


def transcribe_file(
    model: WhisperModel,
    filepath: Path,
    output_dir: Path,
    language: str = "ru",
    vad: bool = True,
    progress_file: Path | None = None,
):
    """Транскрибирует один файл. Если длинный — с нарезкой на чанки."""
    print(f"\n{'=' * 60}")
    print(f"🎧  {filepath.name}")
    print(f"    размер: {filepath.stat().st_size / 1024 / 1024:.0f} МБ")
    print(f"    язык:   {language}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    stem = filepath.stem
    text_parts: list[str] = []
    total_dur = 0.0

    try:
        do_chunk = False
        try:
            total_dur = get_audio_duration(filepath)
            if total_dur > LONG_FILE_THRESHOLD:
                do_chunk = True
                print(f"  ⏱️   {total_dur:.0f} с ({total_dur/3600:.1f} ч) — чанковый режим")
        except Exception:
            pass  # ffprobe не сработал — транскрибируем как есть

        if do_chunk:
            transcribe_chunked(model, filepath, language, vad,
                               progress_file, t0, text_parts)
        else:
            transcribe_stream(model, filepath, language, vad,
                              progress_file, t0, text_parts)

    except IndexError as e:
        print(f"\n  ❌  В файле нет аудиодорожки: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error",
                            "error": f"Нет аудиодорожки: {e}"})
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌  Ошибка: {e}")
        if progress_file:
            _write_progress(progress_file, {"status": "error", "error": str(e)})
        sys.exit(1)

    elapsed = time.time() - t0

    # ── Финальное сохранение ──
    full_text = "".join(text_parts).rstrip()
    write_txt(output_dir / f"{stem}.txt", full_text)

    if progress_file:
        _write_progress(progress_file, {
            "status": "done", "progress_pct": 100,
            "total_duration_sec": round(total_dur, 1),
            "processed_sec": round(total_dur, 1),
            "elapsed_sec": round(elapsed, 1),
            "segments_count": 0, "text": "",
        })

    speed = total_dur / elapsed if elapsed > 0 else 0
    print(f"\n  ⏱️   Обработано за {elapsed/60:.1f} мин ({speed:.1f}x realtime)")
    print(f"  ✅  {stem} — готово!\n")


# ─── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Транскрибация аудио через faster-whisper"
    )
    parser.add_argument("input", help="Аудиофайл (mp3/wav/m4a/…)")
    parser.add_argument("-o", "--output", default="transcripts",
                        help="Папка для результатов")
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "base"),
                        help="tiny / base / small / medium / large-v3")
    parser.add_argument("--lang", default=os.environ.get("WHISPER_LANG", "ru"))
    parser.add_argument("--device", default=os.environ.get("WHISPER_DEVICE", "cpu"),
                        choices=["cpu", "cuda"])
    parser.add_argument("--compute", default=os.environ.get("WHISPER_COMPUTE", "int8"),
                        choices=["float16", "int8_float16", "int8", "float32"])
    parser.add_argument("--no-vad", action="store_false", dest="vad",
                        help="Отключить VAD-фильтр")
    parser.add_argument("--progress-file", default=None,
                        help="Путь к JSON-файлу для записи прогресса")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌  Файл не найден: {input_path}")
        sys.exit(1)

    print(f"⏳  Загружаю модель '{args.model}' ({args.device}/{args.compute})...")
    t_load = time.time()
    model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute,
        cpu_threads=os.cpu_count() or 2, num_workers=1,
    )
    print(f"   ✅  Загрузка за {time.time() - t_load:.0f} с\n")

    output_path.mkdir(parents=True, exist_ok=True)
    progress_file = Path(args.progress_file) if args.progress_file else None
    transcribe_file(model, input_path, output_path, args.lang, args.vad,
                    progress_file=progress_file)


if __name__ == "__main__":
    main()
