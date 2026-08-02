#!/usr/bin/env python3
"""
bulk_upload.py — массовая заливка mp3 в API с ограниченной параллельностью.

Заливает все аудиофайлы из папки, пишет карту «файл → uuid» в JSON,
чтобы ничего не потерялось и можно было потом забрать результаты по uuid.
Повторный запуск пропускает уже загруженные файлы (по имени).

Требует: pip install requests

Пример:
    python bulk_upload.py ./mp3_folder \
        --url http://SERVER:8080 --api-key SECRET --workers 4
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma", ".aiff"}


def upload_one(path: Path, url: str, api_key: str) -> dict:
    with open(path, "rb") as f:
        resp = requests.post(
            f"{url}/api/transcriptions",
            headers={"X-API-Key": api_key},
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=3600,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser(description="Массовая заливка mp3 в транскрайбер")
    ap.add_argument("folder", help="Папка с аудиофайлами")
    ap.add_argument("--url", required=True, help="Базовый URL API, напр. http://host:8080")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--workers", type=int, default=4, help="Параллельных загрузок")
    ap.add_argument("--map", default="upload_map.json",
                    help="Файл карты «имя → uuid»")
    args = ap.parse_args()

    folder = Path(args.folder)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"Нет аудиофайлов в {folder}")
        sys.exit(1)

    map_path = Path(args.map)
    done = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}

    todo = [p for p in files if p.name not in done]
    print(f"Всего файлов: {len(files)}, к заливке: {len(todo)}, уже загружено: {len(done)}")

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_one, p, args.url, args.api_key): p for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                res = fut.result()
                done[p.name] = res["id"]
                ok += 1
                print(f"  ✅ {p.name} → {res['id']}")
            except Exception as e:
                print(f"  ❌ {p.name}: {e}")
            # Сохраняем карту после каждого файла — ничего не теряется
            map_path.write_text(json.dumps(done, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    print(f"\nЗагружено сейчас: {ok}. Карта: {map_path}")


if __name__ == "__main__":
    main()
