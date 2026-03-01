"""
live_caption_ja.py  –  Live Japanese TV caption + English translation

Captures audio (microphone or system loopback), transcribes Japanese speech
with Faster-Whisper, and displays the original + English translation in a
TV-style subtitle bar anchored to the bottom of the screen.

Quick start
-----------
# Microphone (default):
    python live_caption_ja.py

# System audio (TV through speakers) – Linux/PulseAudio:
    # 1. List devices and find your monitor source:
    python live_caption_ja.py --audio_source list
    # or:  pactl list short sources | grep monitor

    # 2. Run with the monitor source (usually contains "monitor"):
    python live_caption_ja.py --audio_source monitor

# Larger model for better accuracy (needs more VRAM/RAM):
    python live_caption_ja.py --model large

# CPU-only mode:
    python live_caption_ja.py --device cpu
"""

import argparse
import io
import re
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep

import nltk
import speech_recognition as sr
from faster_whisper import WhisperModel

from SubtitleWindow import SubtitleWindow


# ---------------------------------------------------------------------------
# Japanese sentence splitting
# ---------------------------------------------------------------------------

_JA_SENTENCE_END = re.compile(r"(?<=[。！？\.\!\?])\s*")


def split_sentences(text: str) -> list[str]:
    """Split text on Japanese and ASCII sentence-ending punctuation."""
    parts = _JA_SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Japanese TV caption and English translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size. Larger = more accurate but slower. (default: medium)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. (default: auto)",
    )
    parser.add_argument(
        "--compute_type", default="auto",
        choices=["auto", "int8", "int8_float16", "float16", "int16", "float32"],
        help="Quantization type. (default: auto)",
    )
    parser.add_argument(
        "--threads", default=0, type=int,
        help="CPU inference threads. 0 = use all cores. (default: 0)",
    )
    parser.add_argument(
        "--energy_threshold", default=300, type=int,
        help="Mic/audio sensitivity threshold. Lower = picks up quieter sounds. (default: 300)",
    )
    parser.add_argument(
        "--record_timeout", default=2, type=float,
        help="Audio chunk length in seconds. (default: 2)",
    )
    parser.add_argument(
        "--phrase_timeout", default=2, type=float,
        help="Silence gap (seconds) before a new caption line starts. (default: 2)",
    )
    parser.add_argument(
        "--source_lang", default="ja", type=str,
        help="Source language code passed to Whisper. (default: ja)",
    )
    parser.add_argument(
        "--target_lang", default="English", type=str,
        help="Translation target language. (default: English)",
    )

    if "linux" in platform:
        parser.add_argument(
            "--audio_source", default="pulse", type=str,
            help=(
                "Audio input device name (partial match). "
                "Use 'list' to print available devices. "
                "For TV audio use your PulseAudio monitor source, e.g. 'monitor'. "
                "(default: pulse)"
            ),
        )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Audio source
    # ------------------------------------------------------------------
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False  # keep threshold stable

    if "linux" in platform:
        audio_source_name = args.audio_source
        if not audio_source_name or audio_source_name == "list":
            print("Available audio devices:\n")
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  [{idx:2d}]  {name}")
            print(
                "\nTip: For TV audio (system loopback) look for a device with 'monitor' in its name."
                "\n     You can also run:  pactl list short sources | grep monitor"
            )
            return

        source = None
        for idx, name in enumerate(sr.Microphone.list_microphone_names()):
            if audio_source_name.lower() in name.lower():
                source = sr.Microphone(sample_rate=16000, device_index=idx)
                print(f"Audio source: [{idx}] {name}")
                break

        if source is None:
            print(f"No audio device matching '{audio_source_name}' found.")
            print("Run with '--audio_source list' to see available devices.")
            return
    else:
        source = sr.Microphone(sample_rate=16000)

    # ------------------------------------------------------------------
    # Whisper model
    # ------------------------------------------------------------------
    model_size = "large-v2" if args.model == "large" else args.model
    # Japanese requires the multilingual model – never append ".en"

    compute_type = "int8" if args.device == "cpu" else args.compute_type

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print(f"Loading Whisper model '{model_size}' …")
    audio_model = WhisperModel(
        model_size,
        device=args.device,
        compute_type=compute_type,
        cpu_threads=args.threads,
    )

    window = SubtitleWindow()

    # ------------------------------------------------------------------
    # Recording loop
    # ------------------------------------------------------------------
    phrase_time = None
    last_sample = bytes()
    data_queue: Queue = Queue()
    temp_file = NamedTemporaryFile(suffix=".wav", delete=False).name

    transcription: list[str] = [""]

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    print(
        f"\nReady! Listening for {args.source_lang.upper()} → translating to {args.target_lang}."
        "\nPress Ctrl+C to stop.\n"
    )

    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True

                phrase_time = now

                while not data_queue.empty():
                    last_sample += data_queue.get()

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, "w+b") as f:
                    f.write(wav_data.read())

                # Transcribe – force Japanese so Whisper skips language detection
                segments, _info = audio_model.transcribe(
                    temp_file,
                    language=args.source_lang,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                text = "".join(seg.text for seg in segments).strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Build recent context and split into sentences
                recent = "".join(transcription[-10:])
                sentences = split_sentences(recent)

                window.update_text(sentences, args.target_lang)

            sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nFull transcription:")
    for line in transcription:
        if line.strip():
            print(line)


if __name__ == "__main__":
    main()
