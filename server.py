"""
server.py  –  Live Japanese caption web server

Captures audio (mic or PulseAudio monitor for TV system audio), transcribes
Japanese speech with OpenAI Whisper (PyTorch), translates to English, and
streams the results to every connected browser in real time via WebSocket.

AMD GPU (ROCm) is supported – PyTorch exposes ROCm under the same "cuda"
device name, so --device cuda works on both NVIDIA and AMD cards.

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

  # Force CPU:
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

# Windows: prevent crash when PyTorch (libiomp5md.dll) and CTranslate2
# (libomp140.x86_64.dll) both try to initialise their own OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import socket
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep

import numpy as np
import torch
import nltk
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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


# ── Hallucination filter ──────────────────────────────────────────────────────

# Phrases Whisper commonly fabricates on silence / low-energy audio.
# Checked case-insensitively after stripping punctuation/spaces.
_HALLUCINATION_PHRASES: frozenset[str] = frozenset({
    # Japanese
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "ありがとうございました",
    "お疲れ様でした",
    "ご覧いただきありがとうございました",
    "最後までご覧いただきありがとうございました",
    "字幕は自動生成されました",
    "字幕",
    "翻訳",
    "チャンネル登録よろしくお願いします",
    "高評価をお願いします",
    # English
    "thank you for watching",
    "thanks for watching",
    "thank you for your hard work",
    "thank you very much",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "[music]",
    "[applause]",
    "[silence]",
})


def _is_hallucination(segments: list, text: str, no_speech_threshold: float = 0.6) -> bool:
    """
    Return True when the Whisper result should be discarded as a hallucination.

    Two checks:
    • Every segment has a high no_speech_prob  (model itself doubts there is speech)
    • The full transcript exactly matches a known filler phrase
    """
    if segments and all(seg.no_speech_prob > no_speech_threshold for seg in segments):
        return True
    return text.strip().lower() in _HALLUCINATION_PHRASES


# ── Helpers ───────────────────────────────────────────────────────────────────


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
# 16000 Hz × 2 bytes × 1.5 seconds = 48 000 bytes
_CLIENT_CHUNK_BYTES = 48_000


async def _process_client_pcm(
    ws: WebSocket,
    pcm: bytes,
    transcription: list[str],
    sample_rate: int,
    gtranslate: GoogleTranslate,
) -> None:
    """Convert a PCM chunk to float32 and transcribe directly (no ffmpeg needed)."""
    # 16-bit signed mono PCM → float32 in [-1, 1] (Whisper's native format)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        def _run_whisper():
            segments_gen, _ = _model.transcribe(
                audio,
                language=_args.source_lang,
                beam_size=_args.beam_size,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            segments = list(segments_gen)
            text = "".join(seg.text for seg in segments).strip()
            if _is_hallucination(segments, text):
                return ""
            return text

        text = await asyncio.to_thread(_run_whisper)
    except Exception as exc:
        print(f"[client transcribe error] {exc}")
        return

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

        print(f"[server audio] [JA] {original}")
        print(f"[server audio] [EN] {translation}\n")
        await _broadcast({"original": original, "translation": translation})


# ── Server-audio capture thread ───────────────────────────────────────────────

def _audio_loop(args: argparse.Namespace, source, model: WhisperModel) -> None:
    """
    Continuously capture audio, transcribe with faster-whisper (forced to
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
            now = datetime.now(timezone.utc)

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

            segments_gen, _ = model.transcribe(
                temp_file,
                language=args.source_lang,
                beam_size=args.beam_size,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                no_speech_threshold=0.5,
                log_prob_threshold=-0.5,
            )
            segments = list(segments_gen)
            text = "".join(
                seg.text for seg in segments
                if seg.no_speech_prob < 0.5
            ).strip()
            if not text or _is_hallucination(segments, text):
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
    Generate a self-signed TLS certificate (if not already present).
    Tries the `cryptography` library first (no external tools needed),
    then falls back to the openssl CLI. Returns (cert_path, key_path).
    """
    cert, key = "server.crt", "server.key"
    if Path(cert).exists() and Path(key).exists():
        print("Using existing server.crt / server.key")
        return cert, key

    # ── Try pure-Python generation via `cryptography` ────────────────────
    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Live Captions")])
        san_entries = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_entries.insert(0, x509.IPAddress(ipaddress.IPv4Address(ip)))
        except ValueError:
            pass  # ip was a hostname, skip

        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(private_key, hashes.SHA256())
        )

        Path(cert).write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        Path(key).write_bytes(private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        print(f"Generated {cert} and {key} (valid 365 days)")
        return cert, key

    except ImportError:
        pass  # fall through to openssl CLI

    # ── Fallback: openssl CLI ─────────────────────────────────────────────
    import shutil
    import subprocess
    if not shutil.which("openssl"):
        print(
            "ERROR: could not generate a TLS certificate.\n"
            "  Install the 'cryptography' Python package:  pip install cryptography\n"
            "  or install the openssl CLI tool and re-run.\n"
            "  Alternatively, place server.crt and server.key next to server.py\n"
            "  and re-run without --https."
        )
        sys.exit(1)

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
                   help="Whisper model size (default: medium). "
                        "Try 'small' for a good speed/accuracy trade-off.")
    p.add_argument("--beam_size", default=1, type=int,
                   help="Beam search width. 1 = greedy (fastest, default). "
                        "5 = more accurate but ~2–3× slower.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device. 'cuda' works for both NVIDIA and AMD/ROCm. (default: auto)")
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
    model_size = "large-v2" if args.model == "large" else args.model

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper '{model_size}' on {device} ({compute_type}) …")
    try:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if device == "cuda":
            print(f"[warning] CUDA init failed ({exc}); falling back to CPU.")
            device, compute_type = "cpu", "int8"
            _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
            raise
    print(f"Model loaded on: {device}")

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
