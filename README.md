# Audio Transcriber API 🎧→📄

HTTP API для транскрибации аудио (mp3 и др.) в текст на базе
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Загружаешь файл → сразу получаешь `uuid` → потом по `uuid` забираешь текст.
Состояние хранится в SQLite и переживает рестарт. Обработка ограничена
пулом воркеров (`MAX_CONCURRENT`), поэтому можно залить тысячи файлов, не
перегружая сервер — они спокойно проходят очередь по одному.

## Быстрый старт

```bash
API_KEY=секрет docker compose up -d --build
```

## Как работает

1. `POST /api/transcriptions` с mp3 → файл стримится на диск, создаётся
   запись `queued`, в ответ приходит `uuid` (мгновенно).
2. Фоновый пул берёт задачи из очереди (`MAX_CONCURRENT` одновременно) и
   гоняет `transcribe.py`. Длинные файлы (> 30 мин) режутся на чанки.
3. `GET /api/transcriptions/{uuid}` показывает статус и прогресс, а когда
   `done` — отдаёт текст.
4. При падении/рестарте зависшие `processing` возвращаются в `queued`.

## API

Все запросы (кроме `/health`) требуют заголовок `X-API-Key: <ключ>`.

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/transcriptions` | Загрузить аудио (multipart `file`). Ответ: `{id, status, filename}` |
| `GET` | `/api/transcriptions/{id}` | Запись: статус, прогресс, метаданные (+ `text`, если `done`) |
| `GET` | `/api/transcriptions/{id}/text` | Сырой `.txt` (только когда `done`) |
| `GET` | `/api/transcriptions?status=&limit=&offset=` | Список с пагинацией |
| `DELETE` | `/api/transcriptions/{id}` | Удалить запись и файлы |
| `GET` | `/health` | Живость + счётчики очереди |

Статусы: `queued` → `processing` → `done` \| `error`.

### Примеры (curl)

```bash
# Загрузить
curl -X POST http://SERVER:8080/api/transcriptions \
  -H "X-API-Key: секрет" \
  -F "file=@lecture.mp3"
# → {"id":"a1b2c3...","status":"queued","filename":"lecture.mp3"}

# Статус / результат
curl http://SERVER:8080/api/transcriptions/a1b2c3... -H "X-API-Key: секрет"

# Готовый текст
curl http://SERVER:8080/api/transcriptions/a1b2c3.../text -H "X-API-Key: секрет"

# Список необработанных
curl "http://SERVER:8080/api/transcriptions?status=queued&limit=100" \
  -H "X-API-Key: секрет"
```

### Массовая заливка

Для ~2500 файлов — скрипт с ограниченной параллельностью, который
сохраняет карту «имя → uuid» (ничего не теряется, повторный запуск
докачивает остаток):

```bash
pip install requests
python bulk_upload.py ./mp3_folder --url http://SERVER:8080 \
  --api-key секрет --workers 4
```

## Параметры (env)

| Переменная | По умолч. | Описание |
|-----------|-----------|----------|
| `API_KEY` | `change-me` | Статичный ключ для `X-API-Key` |
| `MAX_CONCURRENT` | `1` | Сколько файлов обрабатывать одновременно |
| `WHISPER_MODEL` | `base` | tiny / base / small / medium / large-v3 |
| `WHISPER_LANG` | `ru` | Язык распознавания |
| `WHISPER_DEVICE` | `cpu` | cpu / cuda |
| `WHISPER_COMPUTE` | `int8` | int8 (cpu) / float16 (cuda) |
| `DELETE_AUDIO_AFTER_DONE` | `false` | Удалять mp3 после готовности (экономия диска) |
| `MAX_FILE_SIZE_GB` | `5` | Лимит размера загрузки |

> На CPU скорость ~4–6x realtime (модель `base`). 4-часовой файл ≈ 40–60 мин
> обработки. При большой очереди это долго — увеличивайте `MAX_CONCURRENT`
> под число ядер или переходите на GPU (`WHISPER_DEVICE=cuda`).

## Структура

```
app.py           # FastAPI: API + фоновый пул воркеров
transcribe.py    # движок faster-whisper (+ чанкинг длинных файлов)
db.py            # SQLite: очередь и записи
bulk_upload.py   # клиент массовой заливки
data/            # SQLite + загруженные аудио + тексты (в .gitignore)
old-project/     # прежняя версия (видео + веб-UI), как референс
```
