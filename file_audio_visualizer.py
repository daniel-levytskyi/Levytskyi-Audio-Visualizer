import tkinter as tk
from tkinter import filedialog
import subprocess
import numpy as np
import sounddevice as sd
import threading

# ----------------------
# Config
# ----------------------
WIDTH = 1200
HEIGHT = 500
BANDS = 16
FPS = 144
BLOCK_SIZE = 1024

HEADROOM = 0.8
DECAY = 0.95

RISE_SPEED = 0.35  # how fast bars go up
FALL_SPEED = 0.08   # how slow bars fall

fft_target = np.zeros(BANDS)
fft_display = np.zeros(BANDS)

lock = threading.Lock()
stop_flag = threading.Event()
running_max = 1e-6

# ----------------------
# Audio playback (FFmpeg)
# ----------------------
def play_file(path):
    stop_flag.clear()

    cmd = [
        "ffmpeg",
        "-i", path,
        "-f", "f32le",
        "-ac", "2",
        "-ar", "44100",
        "pipe:1"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def callback(outdata, frames, time, status):
        if stop_flag.is_set():
            raise sd.CallbackStop

        raw = proc.stdout.read(frames * 2 * 4)
        if len(raw) < frames * 2 * 4:
            raise sd.CallbackStop

        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
        outdata[:] = data

        mono = data.mean(axis=1)
        fft = np.abs(np.fft.rfft(mono))
        fft = fft[:BANDS * 4]
        bands = fft.reshape(BANDS, -1).mean(axis=1)

        bands = np.log1p(bands * 10)

        global running_max
        max_val = np.max(bands)
        running_max = max(max_val, running_max * DECAY)
        bands /= running_max + 1e-6

        with lock:
            fft_target[:] = bands

    with sd.OutputStream(
        samplerate=44100,
        channels=2,
        blocksize=BLOCK_SIZE,
        callback=callback
    ):
        while not stop_flag.is_set():
            sd.sleep(50)

# ----------------------
# Visualizer
# ----------------------
class Visualizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, bg="#121212", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.left = []
        self.right = []
        self.create_bars()
        self.update()

    def create_bars(self):
        center = WIDTH // 2
        bar_width = (WIDTH // 2) / BANDS

        for i in range(BANDS):
            l = self.canvas.create_rectangle(
                center - (i + 1) * bar_width + 1, HEIGHT,
                center - i * bar_width - 1, HEIGHT,
                fill="#4dd0e1", width=0
            )
            r = self.canvas.create_rectangle(
                center + i * bar_width + 1, HEIGHT,
                center + (i + 1) * bar_width - 1, HEIGHT,
                fill="#4dd0e1", width=0
            )
            self.left.append(l)
            self.right.append(r)

    def update(self):
        with lock:
            target = fft_target.copy()

        # Smooth interpolation
        for i in range(BANDS):
            if target[i] > fft_display[i]:
                fft_display[i] += (target[i] - fft_display[i]) * RISE_SPEED
            else:
                fft_display[i] += (target[i] - fft_display[i]) * FALL_SPEED

        max_height = HEIGHT * HEADROOM

        for i, v in enumerate(fft_display):
            h = v * max_height

            self.canvas.coords(
                self.left[i],
                self.canvas.coords(self.left[i])[0],
                HEIGHT - h,
                self.canvas.coords(self.left[i])[2],
                HEIGHT
            )
            self.canvas.coords(
                self.right[i],
                self.canvas.coords(self.right[i])[0],
                HEIGHT - h,
                self.canvas.coords(self.right[i])[2],
                HEIGHT
            )

        self.root.after(int(1000 / FPS), self.update)

# ----------------------
# File picker
# ----------------------
def open_file():
    path = filedialog.askopenfilename(
        filetypes=[
            ("Audio files", "*.wav *.flac *.mp3 *.mp4 *.m4a"),
        ]
    )
    if not path:
        return

    stop_flag.set()
    threading.Thread(target=play_file, args=(path,), daemon=True).start()

# ----------------------
# Main
# ----------------------
root = tk.Tk()
root.title("Smooth Audio Visualizer")
root.geometry(f"{WIDTH}x{HEIGHT + 50}")
root.configure(bg="#1a1a1a")

btn = tk.Button(
    root,
    text="Open Audio File",
    command=open_file,
    bg="#2a2a2a",
    fg="white",
    relief="flat"
)
btn.pack(fill="x")

Visualizer(root)
root.mainloop()
