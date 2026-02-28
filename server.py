"""
server.py  –  Live Japanese caption web server

Captures audio (mic or PulseAudio monitor for TV system audio), transcribes
Japanese speech with Faster-Whisper, translates to English, and streams the
results to every connected browser in real time via WebSocket.

Clients can also stream audio FROM their phone/tablet mic through the browser
(requires HTTPS – use --https to auto-generate a self-signed certificate).

Quick start
-----------
  # Microphone on this PC (default):
      python server.py

  # TV system audio via PulseAudio loopback (Linux):
      python server.py --audio_source monitor

  # List audio devices (Linux):
      python server.py --audio_source list

  # Enable HTTPS so phones/tablets can use THEIR mic in the browser:
      python server.py --https
      # First visit: browser will warn about the self-signed cert –
      # click Advanced → Proceed to accept it (one-time per device).

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
import json
import os
import re
import socket
import struct
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep

import nltk
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from translatepy.translators.google import GoogleTranslate

# ── Sentence splitting (Japanese-aware) ──────────────────────────────────────

_JA_SENT_END = re.compile(r"(?<=[。！？\.\!\?])\s*")


def split_sentences(text: str) -> list[str]:
    parts = _JA_SENT_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Module-level model + args (set in main, used by client-mic handler) ──────

_model: WhisperModel | None = None
_args: argparse.Namespace | None = None

# ── Shared state ──────────────────────────────────────────────────────────────

_caption_queue: Queue = Queue()   # server audio thread → async broadcast loop
_clients: set[WebSocket] = set()  # all connected WebSocket clients


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a minimal WAV container."""
    channels, sample_width = 1, 2
    data_size = len(pcm)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1,                                          # PCM
        channels, sample_rate,
        sample_rate * channels * sample_width,      # byte rate
        channels * sample_width,                    # block align
        sample_width * 8,                           # bits per sample
        b"data", data_size,
    ) + pcm


async def _broadcast(data: dict) -> None:
    """Send a JSON payload to all connected clients, pruning dead ones."""
    dead: set[WebSocket] = set()
    for ws in list(_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_broadcast_loop())
    yield


app = FastAPI(title="Live Captions", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


# ── Client-mic: per-connection audio processing ───────────────────────────────

# How many bytes of 16kHz/16-bit mono PCM to accumulate before processing:
# 16000 Hz × 2 bytes × 3 seconds = 96 000 bytes
_CLIENT_CHUNK_BYTES = 96_000


async def _process_client_pcm(
    ws: WebSocket,
    pcm: bytes,
    transcription: list[str],
    sample_rate: int,
    gtranslate: GoogleTranslate,
) -> None:
    """Convert a PCM chunk to WAV, transcribe, translate, and broadcast."""
    wav = _pcm_to_wav(pcm, sample_rate)
    tmp = NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav)
    tmp.close()

    try:
        def _run_whisper():
            segs, _ = _model.transcribe(
                tmp.name,
                language=_args.source_lang,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return "".join(s.text for s in segs).strip()

        text = await asyncio.to_thread(_run_whisper)
    except Exception as exc:
        print(f"[client transcribe error] {exc}")
        return
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if not text:
        return

    transcription[-1] = text
    recent = "".join(transcription[-10:])
    sentences = split_sentences(recent)
    if not sentences:
        return

    original = sentences[-1]
    try:
        translation = await asyncio.to_thread(
            lambda: str(gtranslate.translate(original, _args.target_lang))
        )
    except Exception:
        translation = ""

    print(f"[client mic] [JA] {original}")
    print(f"[client mic] [EN] {translation}\n")

    await _broadcast({"original": original, "translation": translation})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)

    # Per-client state for browser-mic audio
    audio_buf = bytearray()
    transcription: list[str] = [""]
    sample_rate = 16000        # updated when client sends audio_config
    gtranslate = GoogleTranslate()

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("bytes"):
                # Binary frame = raw 16-bit mono PCM from the browser mic
                audio_buf.extend(msg["bytes"])
                if len(audio_buf) >= _CLIENT_CHUNK_BYTES:
                    pcm = bytes(audio_buf)
                    audio_buf.clear()
                    asyncio.create_task(
                        _process_client_pcm(
                            websocket, pcm, transcription, sample_rate, gtranslate
                        )
                    )

            elif msg.get("text"):
                # Text frame: either audio config metadata or a keep-alive ping
                try:
                    meta = json.loads(msg["text"])
                    if meta.get("type") == "audio_config":
                        sample_rate = int(meta.get("sample_rate", 16000))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


# ── Server-audio broadcast loop (async task) ─────────────────────────────────

async def _broadcast_loop():
    """
    Drain the server-audio caption queue, translate, and push to all clients.
    Translation runs in a thread to avoid blocking the event loop.
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

        try:
            translation = await asyncio.to_thread(
                lambda: str(gtranslate.translate(original, target_lang))
            )
        except Exception:
            translation = ""

        await _broadcast({"original": original, "translation": translation})


# ── Server-audio capture thread ───────────────────────────────────────────────

def _audio_loop(args: argparse.Namespace, source, model: WhisperModel) -> None:
    """
    Continuously capture audio, transcribe with Faster-Whisper (forced to
    Japanese), and put the latest sentence into _caption_queue for the async
    broadcast loop to pick up.
    """
    import speech_recognition as sr
    recorder = sr.Recognizer()
    recorder.energy_threshold         = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    data_queue:  Queue         = Queue()
    phrase_time: datetime | None = None
    last_sample: bytes         = b""
    transcription: list[str]   = [""]
    temp_file = NamedTemporaryFile(suffix=".wav", delete=False).name

    with source:
        recorder.adjust_for_ambient_noise(source)

    def _enqueue(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())

    recorder.listen_in_background(source, _enqueue,
                                  phrase_time_limit=args.record_timeout)
    print("Server-side audio capture started.\n")

    while True:
        try:
            now = datetime.utcnow()

            if data_queue.empty():
                sleep(0.25)
                continue

            phrase_complete = False
            if phrase_time and (now - phrase_time) > timedelta(seconds=args.phrase_timeout):
                last_sample     = b""
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
            print(f"[server audio error] {exc}")
            sleep(1)


# ── TLS cert generation ───────────────────────────────────────────────────────

def _ensure_cert(ip: str) -> tuple[str, str]:
    """
    Generate a self-signed TLS certificate using openssl (if not already
    present). Returns (cert_path, key_path).
    """
    import shutil
    cert, key = "server.crt", "server.key"
    if Path(cert).exists() and Path(key).exists():
        print("Using existing server.crt / server.key")
        return cert, key

    if not shutil.which("openssl"):
        print(
            "ERROR: 'openssl' not found. Install it, or manually create "
            "server.crt and server.key and re-run without --https."
        )
        sys.exit(1)

    import subprocess
    # Build a SAN extension so iOS/Android accept the cert for the LAN IP
    san = f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key, "-out", cert,
            "-days", "365", "-nodes",
            "-subj", "/CN=Live Captions",
            "-addext", san,
        ],
        check=True,
        capture_output=True,
    )
    print(f"Generated {cert} and {key} (valid 365 days)")
    return cert, key


# ── Network helpers ───────────────────────────────────────────────────────────

def _local_ip() -> str:
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
                   help="Audio sensitivity; lower = more sensitive (default: 300)")
    p.add_argument("--record_timeout", default=2, type=float,
                   help="Server audio chunk length in seconds (default: 2)")
    p.add_argument("--phrase_timeout", default=2, type=float,
                   help="Silence gap before new caption line in seconds (default: 2)")
    p.add_argument("--source_lang", default="ja",
                   help="Whisper source language code (default: ja)")
    p.add_argument("--target_lang", default="English",
                   help="Translation target language (default: English)")
    p.add_argument("--port", default=8000, type=int,
                   help="Port to serve on (default: 8000)")
    p.add_argument("--https", action="store_true",
                   help=(
                       "Serve over HTTPS with a self-signed certificate. "
                       "Required for phone/tablet mic access in the browser. "
                       "Needs openssl installed."
                   ))
    p.add_argument("--no_server_audio", action="store_true",
                   help="Disable server-side audio capture (client mic only).")

    if "linux" in platform:
        p.add_argument("--audio_source", default="pulse",
                       help=("Partial name of the audio input device. "
                             "Use 'list' to print available devices. "
                             "For TV audio use your PulseAudio monitor source, "
                             "e.g. 'monitor'. (default: pulse)"))
    return p


def main() -> None:
    global _model, _args
    import uvicorn

    args = _build_parser().parse_args()
    _args = args

    # ── Audio source ──────────────────────────────────────────────────────────
    source = None
    if not args.no_server_audio:
        import speech_recognition as sr
        if "linux" in platform:
            if args.audio_source == "list":
                print("Available audio devices:\n")
                for i, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"  [{i:2d}]  {name}")
                print("\nTip: look for a device with 'monitor' for TV system audio.")
                print("     Or run:  pactl list short sources | grep monitor")
                return

            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                if args.audio_source.lower() in name.lower():
                    source = sr.Microphone(sample_rate=16000, device_index=i)
                    print(f"Server audio source: [{i}] {name}")
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
    _model = WhisperModel(model_size, device=args.device,
                          compute_type=compute_type,
                          cpu_threads=args.threads)

    # ── Start server-side audio thread (if not disabled) ─────────────────────
    if source is not None:
        threading.Thread(
            target=_audio_loop, args=(args, source, _model), daemon=True
        ).start()
    else:
        print("Server-side audio disabled – waiting for browser mic input.")

    # ── TLS ───────────────────────────────────────────────────────────────────
    ip       = _local_ip()
    ssl_cert = ssl_key = None
    scheme   = "http"

    if args.https:
        ssl_cert, ssl_key = _ensure_cert(ip)
        scheme = "https"

    # ── Print access URLs ─────────────────────────────────────────────────────
    print(f"\nOpen on any device on your network:")
    print(f"  {scheme}://{ip}:{args.port}")
    print(f"  {scheme}://localhost:{args.port}  (this PC only)")
    if args.https:
        print("\n  First visit: browser will warn about the self-signed cert.")
        print("  Click 'Advanced' → 'Proceed' to accept it (one-time per device).")
    else:
        print("\n  Phone/tablet mic: restart with --https to enable it.")
    print("\nPress Ctrl+C to stop.\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )


if __name__ == "__main__":
    main()
