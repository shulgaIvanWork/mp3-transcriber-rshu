#!/usr/bin/env python3
"""
app.py — HTTP API транскрибации аудио (mp3) → текст.

Поток:
  POST /api/transcriptions  → стрим файла на диск, строка queued, сразу отдаём uuid
  фоновый пул воркеров      → берёт queued, гоняет transcribe.py, пишет done/error
  GET  /api/transcriptions/{id}[/text]  → статус и результат по uuid

Состояние — в SQLite (db.py), переживает рестарт. Защита от перегрузки —
семафор MAX_CONCURRENT: заливай хоть тысячи файлов, обрабатываются по N за раз.

Запуск (обязательно один worker uvicorn — фоновый пул живёт в процессе):
    uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query
from fastapi.responses import PlainTextResponse

import db

# ─── Конфиг ────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

for d in (UPLOADS_DIR, TRANSCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("API_KEY", "change-me")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_GB", "5")) * 1024 * 1024 * 1024
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SEC", str(48 * 3600)))
DELETE_AUDIO_AFTER_DONE = os.environ.get("DELETE_AUDIO_AFTER_DONE", "false").lower() == "true"
PROGRESS_POLL_SEC = 5

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma", ".aiff",
}

app = FastAPI(title="Audio Transcriber API", version="1.0",
              docs_url="/docs", redoc_url=None)


# ─── Auth (статичный API-ключ) ──────────────────────────────────────

def require_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ─── Фоновый пул воркеров ──────────────────────────────────────────

def _audio_path(job_id: str) -> Optional[Path]:
    matches = list(UPLOADS_DIR.glob(f"{job_id}.*"))
    return matches[0] if matches else None


def _sync_progress(job_id: str, progress_file: Path):
    """Читает JSON-прогресс от transcribe.py и переносит в БД."""
    if not progress_file.exists():
        return
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return
    db.update_progress(
        job_id,
        progress_pct=round(data.get("progress_pct", 0), 1),
        segments_count=data.get("segments_count", 0),
        duration_sec=data.get("total_duration_sec") or None,
    )


async def process_job(job: dict):
    job_id = job["id"]
    audio = _audio_path(job_id)
    if audio is None:
        db.mark_error(job_id, "Загруженный файл не найден на диске")
        return

    txt_path = TRANSCRIPTS_DIR / f"{job_id}.txt"
    progress_file = TRANSCRIPTS_DIR / f"{job_id}.progress.json"
    log_path = TRANSCRIPTS_DIR / f"{job_id}.log"

    cmd = [
        "python", "transcribe.py",
        "--model", os.environ.get("WHISPER_MODEL", "base"),
        "--lang", os.environ.get("WHISPER_LANG", "ru"),
        "--device", os.environ.get("WHISPER_DEVICE", "cpu"),
        "--compute", os.environ.get("WHISPER_COMPUTE", "int8"),
        "-o", str(TRANSCRIPTS_DIR),
        "--progress-file", str(progress_file),
        str(audio),
    ]

    print(f"🎯  Старт {job_id} ({audio.name})")
    logf = open(log_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=logf, stderr=logf, cwd=str(PROJECT_DIR),
        )
        waiter = asyncio.ensure_future(proc.wait())
        started = time.time()
        while not waiter.done():
            await asyncio.sleep(PROGRESS_POLL_SEC)
            _sync_progress(job_id, progress_file)
            if time.time() - started > JOB_TIMEOUT:
                proc.kill()
                await waiter
                db.mark_error(job_id, f"Таймаут (> {JOB_TIMEOUT} с)")
                print(f"❌  Таймаут {job_id}")
                return
        returncode = await waiter
    except Exception as e:
        db.mark_error(job_id, f"Не удалось запустить обработку: {e}")
        print(f"❌  {job_id}: {e}")
        return
    finally:
        logf.close()

    if returncode == 0 and txt_path.exists():
        duration = None
        try:
            pdata = json.loads(progress_file.read_text(encoding="utf-8"))
            duration = pdata.get("total_duration_sec")
        except Exception:
            pass
        db.mark_done(job_id, duration_sec=duration)
        print(f"✅  Готово {job_id}")
        _cleanup(progress_file, log_path)
        if DELETE_AUDIO_AFTER_DONE:
            _cleanup(audio)
    else:
        err = ""
        try:
            err = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
        except Exception:
            pass
        db.mark_error(job_id, err or f"Обработка завершилась с кодом {returncode}")
        print(f"❌  Ошибка {job_id} (код {returncode})")


def _cleanup(*paths: Path):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


async def worker_loop(n: int):
    while True:
        job = db.claim_next()
        if job:
            await process_job(job)
        else:
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup():
    db.init_db()
    requeued = db.requeue_stale()
    if requeued:
        print(f"🔄  Восстановлено зависших задач: {requeued}")
    for i in range(MAX_CONCURRENT):
        asyncio.create_task(worker_loop(i))
    print(f"🎧  API запущен: MAX_CONCURRENT={MAX_CONCURRENT}, "
          f"модель={os.environ.get('WHISPER_MODEL', 'base')}")


# ─── Сериализация записи для ответа ─────────────────────────────────

def _serialize(job: dict) -> dict:
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress_pct": job["progress_pct"],
        "segments_count": job["segments_count"],
        "duration_sec": job["duration_sec"],
        "error": job["error"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }


# ─── API ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "queue": db.counts()}


@app.post("/api/transcriptions", status_code=201)
async def create_transcription(
    file: UploadFile = File(...), _=Depends(require_key)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Формат {ext} не поддерживается. Разрешено: "
                   f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    job_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{job_id}{ext}"

    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Файл слишком большой")
                f.write(chunk)
    except HTTPException:
        _cleanup(dest)
        raise
    except Exception as e:
        _cleanup(dest)
        raise HTTPException(status_code=500, detail=f"Ошибка записи файла: {e}")

    db.create(job_id, file.filename)
    print(f"📥  Загружен {job_id} ({file.filename}, {size/1024/1024:.1f} МБ)")

    return {"id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/transcriptions")
async def list_transcriptions(
    status: Optional[str] = Query(default=None,
                                  pattern="^(queued|processing|done|error)$"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _=Depends(require_key),
):
    items, total = db.list_jobs(status=status, limit=limit, offset=offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_serialize(j) for j in items],
    }


@app.get("/api/transcriptions/{job_id}")
async def get_transcription(job_id: str, _=Depends(require_key)):
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    result = _serialize(job)
    if job["status"] == "done":
        txt = TRANSCRIPTS_DIR / f"{job_id}.txt"
        if txt.exists():
            result["text"] = txt.read_text(encoding="utf-8")
    return result


@app.get("/api/transcriptions/{job_id}/text", response_class=PlainTextResponse)
async def get_transcription_text(job_id: str, _=Depends(require_key)):
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Ещё не готово (status={job['status']})")
    txt = TRANSCRIPTS_DIR / f"{job_id}.txt"
    if not txt.exists():
        raise HTTPException(status_code=404, detail="Файл результата не найден")
    return txt.read_text(encoding="utf-8")


@app.delete("/api/transcriptions/{job_id}")
async def delete_transcription(job_id: str, _=Depends(require_key)):
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    audio = _audio_path(job_id)
    if audio:
        _cleanup(audio)
    _cleanup(
        TRANSCRIPTS_DIR / f"{job_id}.txt",
        TRANSCRIPTS_DIR / f"{job_id}.progress.json",
        TRANSCRIPTS_DIR / f"{job_id}.log",
    )
    db.delete(job_id)
    return {"ok": True, "id": job_id}
