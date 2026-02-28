"""
server.py  –  Live Japanese caption web server

Captures audio (mic or PulseAudio monitor for TV system audio), transcribes
Japanese speech with Faster-Whisper, translates to English, and streams the
results to every connected browser in real time via WebSocket.

Quick start
-----------
  # Microphone (default):
      python server.py

  # List audio devices (Linux):
      python server.py --audio_source list

  # TV system audio via PulseAudio loopback (Linux):
      python server.py --audio_source monitor

  # Larger model for better accuracy:
      python server.py --model large

  # CPU-only:
      python server.py --device cpu

Then open the URL printed in the terminal on any phone/tablet/browser on the
same network.
"""

import argparse
import asyncio
import io
import re
import socket
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep

import nltk
import speech_recognition as sr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from translatepy.translators.google import GoogleTranslate

# ── Sentence splitting (Japanese-aware) ──────────────────────────────────────

_JA_SENT_END = re.compile(r"(?<=[。！？\.\!\?])\s*")


def split_sentences(text: str) -> list[str]:
    parts = _JA_SENT_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Shared state (all async, single event loop – no locking needed) ───────────

_caption_queue: Queue = Queue()   # audio thread → async broadcast loop
_clients: set[WebSocket] = set()  # connected WebSocket clients


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_broadcast_loop())
    yield


app = FastAPI(title="Live Captions", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            # Keep the connection alive; data only flows server→client
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


# ── Broadcast loop (async task) ───────────────────────────────────────────────

async def _broadcast_loop():
    """
    Drain the caption queue, translate each item, and push to all clients.
    Translation is run in a thread so it doesn't block the event loop.
    """
    gtranslate = GoogleTranslate()

    while True:
        try:
            payload = _caption_queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.1)
            continue

        original    = payload.get("original", "")
        target_lang = payload.get("target_lang", "English")

        if not original:
            continue

        # Run blocking translate call off the event loop
        try:
            translation = await asyncio.to_thread(
                lambda: str(gtranslate.translate(original, target_lang))
            )
        except Exception:
            translation = ""

        data = {"original": original, "translation": translation}
        dead: set[WebSocket] = set()

        for ws in list(_clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)

        _clients.difference_update(dead)


# ── Audio + transcription (daemon thread) ─────────────────────────────────────

def _audio_loop(args: argparse.Namespace, source: sr.Microphone,
                model: WhisperModel) -> None:
    """
    Continuously capture audio, transcribe with Faster-Whisper (forced to
    Japanese), and put the latest sentence into _caption_queue for the async
    broadcast loop to pick up.
    """
    recorder = sr.Recognizer()
    recorder.energy_threshold    = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    data_queue:   Queue = Queue()
    phrase_time:  datetime | None = None
    last_sample:  bytes = b""
    transcription: list[str] = [""]
    temp_file = NamedTemporaryFile(suffix=".wav", delete=False).name

    with source:
        recorder.adjust_for_ambient_noise(source)

    def _enqueue(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())

    recorder.listen_in_background(source, _enqueue,
                                  phrase_time_limit=args.record_timeout)
    print("Audio capture started.\n")

    while True:
        try:
            now = datetime.utcnow()

            if data_queue.empty():
                sleep(0.25)
                continue

            phrase_complete = False
            if phrase_time and (now - phrase_time) > timedelta(seconds=args.phrase_timeout):
                last_sample   = b""
                phrase_complete = True
            phrase_time = now

            while not data_queue.empty():
                last_sample += data_queue.get()

            audio_data = sr.AudioData(last_sample,
                                      source.SAMPLE_RATE,
                                      source.SAMPLE_WIDTH)
            wav_bytes = io.BytesIO(audio_data.get_wav_data())
            with open(temp_file, "w+b") as f:
                f.write(wav_bytes.read())

            # Transcribe – force Japanese (skips language detection, faster & more accurate)
            segments, _ = model.transcribe(
                temp_file,
                language=args.source_lang,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = "".join(seg.text for seg in segments).strip()
            if not text:
                continue

            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            recent    = "".join(transcription[-10:])
            sentences = split_sentences(recent)

            if sentences:
                _caption_queue.put({
                    "original":    sentences[-1],
                    "target_lang": args.target_lang,
                })

        except Exception as exc:
            print(f"[audio loop error] {exc}")
            sleep(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_ip() -> str:
    """Return the LAN IP so the user knows what URL to open on their device."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"
    finally:
        s.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live Japanese caption web server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="medium",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (default: medium)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute_type", default="auto",
                   choices=["auto", "int8", "int8_float16", "float16", "int16", "float32"])
    p.add_argument("--threads", default=0, type=int,
                   help="CPU inference threads; 0 = all cores (default: 0)")
    p.add_argument("--energy_threshold", default=300, type=int,
                   help="Audio sensitivity; lower = picks up quieter sounds (default: 300)")
    p.add_argument("--record_timeout", default=2, type=float,
                   help="Audio chunk length in seconds (default: 2)")
    p.add_argument("--phrase_timeout", default=2, type=float,
                   help="Silence gap before new caption line in seconds (default: 2)")
    p.add_argument("--source_lang", default="ja",
                   help="Whisper source language code (default: ja)")
    p.add_argument("--target_lang", default="English",
                   help="Translation target language (default: English)")
    p.add_argument("--port", default=8000, type=int,
                   help="Port to serve on (default: 8000)")

    if "linux" in platform:
        p.add_argument("--audio_source", default="pulse",
                       help=("Partial name of the audio input device. "
                             "Use 'list' to print available devices. "
                             "For TV audio use your PulseAudio monitor source, "
                             "e.g. 'monitor'. (default: pulse)"))
    return p


def main() -> None:
    import uvicorn

    args = _build_parser().parse_args()

    # ── Audio source ──────────────────────────────────────────────────────────
    if "linux" in platform:
        if args.audio_source == "list":
            print("Available audio devices:\n")
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  [{i:2d}]  {name}")
            print("\nTip: look for a device with 'monitor' for TV system audio.")
            print("     Or run:  pactl list short sources | grep monitor")
            return

        source = None
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            if args.audio_source.lower() in name.lower():
                source = sr.Microphone(sample_rate=16000, device_index=i)
                print(f"Audio source: [{i}] {name}")
                break

        if source is None:
            print(f"No audio device matching '{args.audio_source}'. "
                  "Run with --audio_source list to see options.")
            return
    else:
        source = sr.Microphone(sample_rate=16000)

    # ── Whisper model ─────────────────────────────────────────────────────────
    model_size   = "large-v2" if args.model == "large" else args.model
    compute_type = "int8" if args.device == "cpu" else args.compute_type

    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print(f"Loading Whisper '{model_size}' …")
    whisper_model = WhisperModel(model_size, device=args.device,
                                 compute_type=compute_type,
                                 cpu_threads=args.threads)

    # ── Start audio thread ────────────────────────────────────────────────────
    audio_thread = threading.Thread(
        target=_audio_loop, args=(args, source, whisper_model), daemon=True
    )
    audio_thread.start()

    # ── Print access URLs ─────────────────────────────────────────────────────
    ip = _local_ip()
    print(f"\nOpen on any device on your network:")
    print(f"  http://{ip}:{args.port}")
    print(f"  http://localhost:{args.port}  (this PC only)")
    print("\nPress Ctrl+C to stop.\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
