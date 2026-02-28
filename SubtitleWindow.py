# SubtitleWindow.py
# TV-style subtitle overlay: Japanese original on top, English translation below.
import tkinter as tk
from translatepy.translators.google import GoogleTranslate


class SubtitleWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Captions")

        # Semi-transparent black background
        self.root.attributes("-alpha", 0.85)
        self.root.configure(bg="black")
        self.root.lift()
        self.root.attributes("-topmost", True)

        # Remove window decorations for a clean subtitle bar
        self.root.overrideredirect(True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Full-width bar anchored to the bottom of the screen
        window_width = screen_width
        window_height = 160
        self.root.geometry(f"{window_width}x{window_height}+0+{screen_height - window_height}")

        # Japanese original (white, larger)
        self.original_label = tk.Label(
            self.root,
            text="",
            wraplength=screen_width - 20,
            font=("Gothic", 22, "bold"),
            bg="black",
            fg="white",
            justify=tk.CENTER,
            anchor="center",
        )
        self.original_label.pack(fill=tk.X, padx=10, pady=(12, 2))

        # English translation (yellow — standard subtitle colour)
        self.translation_label = tk.Label(
            self.root,
            text="",
            wraplength=screen_width - 20,
            font=("Gothic", 20),
            bg="black",
            fg="#FFE066",
            justify=tk.CENTER,
            anchor="center",
        )
        self.translation_label.pack(fill=tk.X, padx=10, pady=(2, 12))

        self._gtranslate = GoogleTranslate()
        self.root.update()

    def update_text(self, sentences, target_lang):
        """Display the most recent sentence and its translation."""
        if not sentences:
            self.root.update()
            return

        latest = sentences[-1].strip()
        if not latest:
            self.root.update()
            return

        self.original_label.config(text=latest)

        try:
            translation = str(self._gtranslate.translate(latest, target_lang))
        except Exception:
            translation = ""

        self.translation_label.config(text=translation)

        print(f"[JA] {latest}")
        print(f"[EN] {translation}\n")

        self.root.update()

    def mainloop(self):
        self.root.mainloop()
