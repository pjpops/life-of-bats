#!/usr/bin/env python3
"""
Life of Bats — Live bat sonogram viewer with BatDetect2 species detection.

Connect your pippyg USB bat detector, select the device from the dropdown,
and click Start. Clips are saved to recordings/ when ultrasonic energy is
detected and automatically identified by BatDetect2.
"""

import sys
import os
import re
import json as _json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt, sosfilt_zi, iirnotch
from math import gcd

# ── BSG-BAT ───────────────────────────────────────────────────────────────────
from bsg_bat import BSG, COMMON_NAMES as _BSG_NAMES

# ── BatDetect2 path ────────────────────────────────────────────────────────────
_BD2_PATH = str(Path(__file__).resolve().parent.parent / "bat_detector")
if _BD2_PATH not in sys.path:
    sys.path.insert(0, _BD2_PATH)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFileDialog, QSlider, QDoubleSpinBox,
    QGraphicsRectItem,
)
from PyQt5.QtCore import Qt, QTimer, QEvent, pyqtSignal, QObject
from PyQt5.QtGui import QColor
import pyqtgraph as pg

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 384_000        # pippyg USB mic
CHUNK_SIZE      = 8_192          # audio callback block size (~21 ms)
FFT_SIZE        = 8_192          # FFT window — use full chunk for best SNR and resolution
FREQ_MIN_HZ     = 15_000         # display range low
FREQ_MAX_HZ     = 120_000        # display range high
WATERFALL_COLS  = 600            # ~12 s of history
CLIP_SECS       = 3              # seconds recorded per trigger
CLIP_SAMPLES    = SAMPLE_RATE * CLIP_SECS
DEBOUNCE_SECS   = 2.0            # minimum gap between triggers
RMS_THRESH      = 0.0173         # minimum RMS to check for ultrasonics (level 7 default)
ULTRA_RATIO     = 0.05           # fraction of energy that must be above ULTRA_MIN_HZ
ULTRA_MIN_HZ    = 20_000         # ultrasonic threshold

RECORDINGS_DIR  = Path(__file__).resolve().parent / "recordings"
BAT_DATA_DIR    = Path.home() / "Desktop" / "Bat data"

# Survey browser — species considered "not interesting" (same logic as cleanup_wavs.py)
_SURVEY_COMMON  = frozenset({"Pipistrellus pipistrellus", "Pipistrellus pygmaeus"})
_SURVEY_THRESH  = 0.3

# Latin → English common names for UK bat species returned by BatDetect2
_COMMON_NAMES: dict[str, str] = {
    "Barbastella barbastellus":    "Barbastelle",
    "Barbastellus barbastellus":   "Barbastelle",
    "Eptesicus serotinus":         "Serotine",
    "Myotis alcathoe":             "Alcathoe's Bat",
    "Myotis bechsteinii":          "Bechstein's Bat",
    "Myotis brandtii":             "Brandt's Bat",
    "Myotis daubentonii":          "Daubenton's Bat",
    "Myotis mystacinus":           "Whiskered Bat",
    "Myotis nattereri":            "Natterer's Bat",
    "Myotis myotis":               "Greater Mouse-eared Bat",
    "Nyctalus leisleri":           "Leisler's Bat",
    "Nyctalus noctula":            "Noctule",
    "Pipistrellus nathusii":       "Nathusius' Pipistrelle",
    "Pipistrellus pipistrellus":   "Common Pipistrelle",
    "Pipistrellus pygmaeus":       "Soprano Pipistrelle",
    "Plecotus auritus":            "Brown Long-eared Bat",
    "Plecotus austriacus":         "Grey Long-eared Bat",
    "Rhinolophus ferrumequinum":   "Greater Horseshoe Bat",
    "Rhinolophus hipposideros":    "Lesser Horseshoe Bat",
    "Tadarida teniotis":           "European Free-tailed Bat",
}

# ── Call detail view — high-resolution spectrogram parameters ─────────────────
# Small FFT window + 75 % overlap stretches each call laterally so FM sweeps
# appear as clear curved lines rather than thin slivers.
CLIP_FFT     = 512                                         # ~1.33 ms per window
CLIP_HOP     = 128                                         # ~0.33 ms time step
_CLIP_WIN    = np.hanning(CLIP_FFT).astype(np.float32)
_CLIP_FREQS  = np.fft.rfftfreq(CLIP_FFT, 1.0 / SAMPLE_RATE)
_CLIP_BIN_LO = int(np.searchsorted(_CLIP_FREQS, FREQ_MIN_HZ))
_CLIP_BIN_HI = int(np.searchsorted(_CLIP_FREQS, FREQ_MAX_HZ))
N_CLIP_BINS  = _CLIP_BIN_HI - _CLIP_BIN_LO


def _compute_clip_spec(audio: np.ndarray, max_secs: float = 10.0) -> np.ndarray:
    """Return a high-resolution spectrogram array (n_frames, N_CLIP_BINS)."""
    audio = audio[: int(max_secs * SAMPLE_RATE)]
    frames = []
    for s in range(0, len(audio) - CLIP_FFT, CLIP_HOP):
        seg = audio[s : s + CLIP_FFT] * _CLIP_WIN
        mag = np.abs(np.fft.rfft(seg))[_CLIP_BIN_LO:_CLIP_BIN_HI]
        frames.append(np.log1p(mag))
    if not frames:
        return np.zeros((1, N_CLIP_BINS), dtype=np.float32)
    return np.array(frames, dtype=np.float32)


# Pre-compute frequency bin indices once
_FREQS   = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
_BIN_LO  = int(np.searchsorted(_FREQS, FREQ_MIN_HZ))
_BIN_HI  = int(np.searchsorted(_FREQS, FREQ_MAX_HZ))
N_FREQ_BINS = _BIN_HI - _BIN_LO
FREQ_AXIS_LO = float(_FREQS[_BIN_LO])
FREQ_AXIS_HI = float(_FREQS[_BIN_HI - 1])

# Pre-compute Hanning window
_WINDOW = np.hanning(FFT_SIZE).astype(np.float32)

# ── Frequency-division audio output ──────────────────────────────────────────
# Divides all frequencies by 8: a 45 kHz bat call becomes 5.6 kHz, a 110 kHz
# lesser horseshoe becomes 13.75 kHz — all species audible simultaneously,
# no tuning required.  Equivalent to the pippyg iPhone app approach.
OUTPUT_RATE  = 48_000                       # standard speaker sample rate
_DIV         = SAMPLE_RATE // OUTPUT_RATE  # decimation factor = 8
_OUT_BLOCK   = CHUNK_SIZE // _DIV          # output frames per input chunk = 1024

# High-pass filter above 15 kHz applied before decimation.
# This strips out room noise / handling noise in the audible band so you only
# hear the bat calls, not wind or background sounds folded into the output.
_HP_SOS = butter(6, 15_000 / (SAMPLE_RATE / 2), btype="high", output="sos")


# ── Cross-thread Qt signals ────────────────────────────────────────────────────
class _Signals(QObject):
    detection       = pyqtSignal(str, str, float)    # timestamp, species, confidence
    status          = pyqtSignal(str)                # status bar message
    file_done       = pyqtSignal()                   # file load finished → re-enable button
    clip_spec       = pyqtSignal(object, str, object)  # spec array, timestamp, annotations
    bsg_result      = pyqtSignal(str, object)        # timestamp, {species: prob}
    bsg_file_result = pyqtSignal(str, object)        # wav_path_str, {species: prob} (survey)

SIGNALS = _Signals()


# ── Queues ─────────────────────────────────────────────────────────────────────
_spec_q      : queue.Queue = queue.Queue(maxsize=1000)
_clip_audio_q: queue.Queue = queue.Queue(maxsize=10)
_hetero_q    : queue.Queue = queue.Queue(maxsize=200)   # raw chunks for heterodyne
_out_q       : queue.Queue = queue.Queue(maxsize=600)   # large buffer for AirPlay latency


# ── Worker: frequency division ────────────────────────────────────────────────
def _freqdiv_worker(enabled: list, volume: list):
    """Daemon thread: high-pass filters the 384 kHz audio to strip audible-band
    noise, then decimates by 8 to 48 kHz.  Ultrasonic bat calls alias into the
    audible range — 45 kHz appears at 3 kHz, 110 kHz at 14 kHz, etc.  All
    species are heard simultaneously with no tuning required.
    """
    zi = sosfilt_zi(_HP_SOS)

    while True:
        chunk = _hetero_q.get()

        if not enabled[0]:
            zi = sosfilt_zi(_HP_SOS)   # reset on disable to avoid a pop on re-enable
            continue

        # High-pass to remove sub-15 kHz room/handling noise, then decimate
        filtered, zi = sosfilt(_HP_SOS, chunk.astype(np.float64), zi=zi)
        decimated    = filtered[::_DIV].astype(np.float32)

        try:
            _out_q.put_nowait(decimated * float(volume[0]))
        except queue.Full:
            pass


# ── Worker: save clip + run BatDetect2 ────────────────────────────────────────
def _save_and_analyse_worker():
    """Daemon thread: saves WAV clips and runs BatDetect2 on each one."""
    import batdetect2.api as api  # loaded once in background thread

    RECORDINGS_DIR.mkdir(exist_ok=True)

    while True:
        audio = _clip_audio_q.get()
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        sig_ts = datetime.now().strftime("%H:%M:%S")
        wav_path = str(RECORDINGS_DIR / f"bat_{ts}.wav")
        try:
            sf.write(wav_path, audio, SAMPLE_RATE)
            SIGNALS.status.emit(f"Saved clip → analysing {os.path.basename(wav_path)} …")

            results     = api.process_file(wav_path)
            annotations = results["pred_dict"].get("annotation", [])

            # Collect best confidence per species
            best: dict[str, float] = {}
            for ann in annotations:
                sp   = ann["class"]
                prob = float(ann["class_prob"])
                if prob > best.get(sp, 0.0):
                    best[sp] = prob

            # Emit high-resolution clip spectrogram for the detail view
            spec = _compute_clip_spec(audio)
            SIGNALS.clip_spec.emit(spec, sig_ts, annotations)

            # BSG-BAT second opinion (runs after BatDetect2, non-blocking on failure)
            if BSG.available:
                bsg = BSG.classify(wav_path)
                if bsg:
                    SIGNALS.bsg_result.emit(sig_ts, bsg)

            if best:
                for sp, conf in sorted(best.items(), key=lambda x: -x[1]):
                    SIGNALS.detection.emit(sig_ts, sp, conf)
                SIGNALS.status.emit(
                    f"Detection complete: {len(best)} species found in {os.path.basename(wav_path)}"
                )
            else:
                SIGNALS.detection.emit(sig_ts, "No bat detected", 0.0)
                SIGNALS.status.emit(f"No bat detected in {os.path.basename(wav_path)}")

        except Exception as e:
            SIGNALS.detection.emit(sig_ts, f"Error: {e}", 0.0)
            SIGNALS.status.emit(f"Analysis error: {e}")
        finally:
            _clip_audio_q.task_done()


# ── Main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Life of Bats")
        self.resize(1100, 860)

        # Audio stream
        self._stream = None

        # Waterfall buffer: shape (WATERFALL_COLS, N_FREQ_BINS)
        # axis-0 = time (x), axis-1 = frequency (y)
        self._waterfall = np.zeros((WATERFALL_COLS, N_FREQ_BINS), dtype=np.float32)
        self._level_max = 1.0   # rolling display ceiling — decays slowly
        self._hide_low_conf = False
        self._rms_thresh = RMS_THRESH   # adjustable at runtime via slider
        self._refs_visible = False
        self._prev_clip_active = False  # for recording-dot state tracking
        self._hide_common  = True        # hide common/soprano pip boxes by default
        self._brightness   = 1.0        # call detail brightness multiplier
        self._bright_step  = 0          # integer offset from default (0 = spec.max())

        # Heterodyne audio output — single-element lists so the worker thread
        # can read updated values without explicit locks
        self._audio_enabled = [False]
        self._volume        = [0.8]
        self._out_stream    = None

        # Notch filter — suppresses a fixed-frequency interference tone
        self._notch_enabled  = False
        self._notch_freq_hz  = 58_000
        self._notch_sos, self._notch_zi = self._build_notch(self._notch_freq_hz)

        self._is_playing    = False          # True while file playback is running
        self._playback_stop = threading.Event()  # set to abort playback
        self._last_detection_ts = ""        # timestamp of the most recent clip batch
        self._clip_boxes: list = []         # annotation overlays on call detail panel

        # Clip cache and pin state for the Recent Detections → Call Detail link
        # Stores the last 50 clips as {ts: (spec, annotations)} in insertion order
        self._clip_cache: dict = {}
        self._pinned_ts: str | None = None   # None = live (always shows latest)
        self._displayed_ts: str | None = None  # timestamp of whatever is on screen RIGHT NOW

        # Survey browser state
        self._survey_mode = False
        self._survey_row_map: dict[str, int] = {}   # wav_path_str → table row index

        # Clip accumulation (written only from audio callback thread)
        self._clip_active             = False
        self._clip_buf: list          = []
        self._clip_samples_collected  = 0
        self._last_trigger_time       = 0.0

        self._build_ui()
        self._start_workers()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(50)   # 20 fps

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # ── Row 1: device + transport + file ─────────────────────────────────
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Device:"))

        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(280)
        self._populate_devices()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        row1.addWidget(self._device_combo)

        refresh_btn = QPushButton("↺")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh device list")
        refresh_btn.clicked.connect(self._refresh_devices)
        row1.addWidget(refresh_btn)

        self._compat_label = QLabel("")
        self._compat_label.setFixedWidth(200)
        row1.addWidget(self._compat_label)
        self._on_device_changed()   # set label for default selection

        row1.addStretch()

        # Recording indicator dot
        self._rec_dot = QLabel("●")
        self._rec_dot.setStyleSheet("color: #444444; font-size: 18px;")
        self._rec_dot.setToolTip("Red = recording clip")
        row1.addWidget(self._rec_dot)

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedWidth(90)
        self._start_btn.clicked.connect(self._start)
        row1.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedWidth(90)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        row1.addWidget(self._stop_btn)

        self._file_btn = QPushButton("📂  Open Recording")
        self._file_btn.clicked.connect(self._open_file)
        row1.addWidget(self._file_btn)

        self._survey_btn = QPushButton("🦇  Browse Survey")
        self._survey_btn.setToolTip("Open a dated survey folder and browse interesting recordings")
        self._survey_btn.clicked.connect(self._browse_survey)
        row1.addWidget(self._survey_btn)

        self._live_btn = QPushButton("↩  Live")
        self._live_btn.setToolTip("Return to live monitoring mode")
        self._live_btn.setVisible(False)
        self._live_btn.clicked.connect(self._exit_survey)
        row1.addWidget(self._live_btn)

        vbox.addLayout(row1)

        # Survey mode banner — hidden until a survey folder is loaded
        self._survey_banner = QLabel("")
        self._survey_banner.setStyleSheet(
            "color: #4caf50; padding: 2px 4px; font-weight: bold;"
        )
        self._survey_banner.setVisible(False)
        vbox.addWidget(self._survey_banner)

        # ── Row 2: sensitivity + species refs + filter ────────────────────────
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Trigger sensitivity:"))
        row2.addWidget(QLabel("Low"))
        self._sens_slider = QSlider(Qt.Horizontal)
        self._sens_slider.setFixedWidth(130)
        self._sens_slider.setRange(1, 10)
        _default_level = self._slider_from_thresh(RMS_THRESH)
        self._sens_slider.setValue(_default_level)
        self._sens_slider.setToolTip(
            "How easily a clip recording is triggered.\n"
            "Increase if bats are being missed; decrease if wind or\n"
            "background noise is causing too many false recordings."
        )
        self._sens_slider.valueChanged.connect(self._on_sensitivity_changed)
        row2.addWidget(self._sens_slider)
        row2.addWidget(QLabel("High"))

        self._sens_label = QLabel(f"Level {_default_level}/10")
        self._sens_label.setFixedWidth(68)
        row2.addWidget(self._sens_label)
        # Sync the runtime threshold to the slider's starting value
        self._rms_thresh = round(0.050 - (_default_level - 1) / 9 * 0.049, 4)

        row2.addSpacing(16)

        self._refs_btn = QPushButton("Species refs")
        self._refs_btn.setCheckable(True)
        self._refs_btn.toggled.connect(self._toggle_refs)
        row2.addWidget(self._refs_btn)

        row2.addStretch()
        vbox.addLayout(row2)

        # ── Row 3: heterodyne audio output ────────────────────────────────────
        row3 = QHBoxLayout()

        self._audio_btn = QPushButton("🔊  Audio on")
        self._audio_btn.setCheckable(True)
        self._audio_btn.setFixedWidth(110)
        self._audio_btn.setToolTip(
            "Frequency division audio (÷8).\n"
            "All bat calls become audible simultaneously — no tuning needed.\n"
            "45 kHz → 5.6 kHz, 110 kHz → 13.75 kHz, etc."
        )
        self._audio_btn.toggled.connect(self._toggle_audio)
        self._audio_btn.setChecked(True)
        row3.addWidget(self._audio_btn)

        row3.addSpacing(24)

        self._notch_btn = QPushButton("Noise filter: on")
        self._notch_btn.setCheckable(True)
        self._notch_btn.setFixedWidth(120)
        self._notch_btn.setToolTip(
            "Narrow notch filter that removes a fixed-frequency interference tone\n"
            "(e.g. USB switching noise, ultrasonic sensors).\n"
            "Affects both the waterfall display and the audio output."
        )
        self._notch_btn.toggled.connect(self._toggle_notch)
        self._notch_btn.setChecked(True)
        row3.addWidget(self._notch_btn)

        self._notch_spin = QDoubleSpinBox()
        self._notch_spin.setRange(FREQ_MIN_HZ / 1000, FREQ_MAX_HZ / 1000)
        self._notch_spin.setValue(self._notch_freq_hz / 1000)
        self._notch_spin.setSuffix(" kHz")
        self._notch_spin.setSingleStep(0.1)
        self._notch_spin.setDecimals(1)
        self._notch_spin.setFixedWidth(90)
        self._notch_spin.setToolTip("Centre frequency of the noise notch filter")
        self._notch_spin.valueChanged.connect(self._on_notch_freq_changed)
        row3.addWidget(self._notch_spin)

        row3.addSpacing(24)
        row3.addWidget(QLabel("Out:"))

        self._out_combo = QComboBox()
        self._out_combo.setMinimumWidth(200)
        self._out_combo.setToolTip("Audio output device — change to use AirPlay, Sonos, etc.")
        self._populate_output_devices()
        self._out_combo.currentIndexChanged.connect(self._on_output_device_changed)
        row3.addWidget(self._out_combo)

        out_refresh_btn = QPushButton("↺")
        out_refresh_btn.setFixedWidth(28)
        out_refresh_btn.setToolTip(
            "Restart audio output on the selected device.\n"
            "For AirPlay/Sonos: first set it as your system output in\n"
            "macOS System Settings → Sound → Output, then click here."
        )
        out_refresh_btn.clicked.connect(self._on_output_device_changed)
        row3.addWidget(out_refresh_btn)

        row3.addStretch()
        vbox.addLayout(row3)

        # Splitter: waterfall on top, detections table on bottom
        splitter = QSplitter(Qt.Vertical)

        # ── Waterfall ──────────────────────────────────────────────────────────
        pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder='col-major')
        self._plot = pg.PlotWidget(background="k")
        self._plot.setLabel("left",   "Frequency (kHz)")
        self._plot.setLabel("bottom", "Time  ←  older — newer  →")

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        # Colour map
        try:
            cmap = pg.colormap.get("inferno")
        except Exception:
            # Manual inferno-like fallback
            pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            col = np.array([
                [0,   0,   0,   255],
                [64,  0,   128, 255],
                [200, 0,   50,  255],
                [255, 140, 0,   255],
                [255, 255, 0,   255],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, col)
        self._img.setColorMap(cmap)

        # Use raw pixel coordinates — no coordinate transform (avoids pyqtgraph
        # version quirks where setImage resets the transform set by setRect).
        # Y-axis tick labels are set manually to show kHz values.
        freq_lo_khz  = FREQ_AXIS_LO / 1000
        freq_hi_khz  = FREQ_AXIS_HI / 1000
        freq_span    = freq_hi_khz - freq_lo_khz
        y_ticks = [
            (int(round((f - freq_lo_khz) / freq_span * N_FREQ_BINS)), f"{f}")
            for f in range(int(freq_lo_khz) + 5, int(freq_hi_khz), 10)
        ]
        self._plot.getAxis("left").setTicks([y_ticks])
        self._plot.setYRange(0, N_FREQ_BINS, padding=0)
        self._plot.setXRange(0, WATERFALL_COLS, padding=0)
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        vb.setLimits(xMin=0, xMax=WATERFALL_COLS, yMin=0, yMax=N_FREQ_BINS)
        self._plot.showGrid(x=False, y=True, alpha=0.3)

        # Horizontal frequency crosshair
        self._hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(color=(255, 255, 255, 160), width=1, style=Qt.DashLine),
        )
        self._hline.setVisible(False)
        self._plot.addItem(self._hline)

        self._freq_label = pg.TextItem("", color=(255, 255, 255), anchor=(0, 1))
        self._freq_label.setVisible(False)
        self._plot.addItem(self._freq_label)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.viewport().installEventFilter(self)

        # ── Species frequency reference lines ──────────────────────────────────
        # (freq in kHz, label, colour)
        _UK_SPECIES = [
            (17,  "Noctule",              "#e74c3c"),
            (25,  "Leisler's",            "#e67e22"),
            (28,  "Serotine",             "#f1c40f"),
            (32,  "Barbastelle",          "#2ecc71"),
            (35,  "Brown long-eared",     "#1abc9c"),
            (38,  "Nathusius' pip",       "#3498db"),
            (45,  "Common pip",           "#9b59b6"),
            (50,  "Natterer's",           "#e91e63"),
            (55,  "Soprano pip",          "#00bcd4"),
            (83,  "Greater horseshoe",    "#ff5722"),
            (110, "Lesser horseshoe",     "#cddc39"),
        ]

        freq_lo_khz = FREQ_AXIS_LO / 1000
        freq_span   = (FREQ_AXIS_HI - FREQ_AXIS_LO) / 1000

        self._ref_lines = []
        for freq_khz, name, colour in _UK_SPECIES:
            if not (freq_lo_khz < freq_khz < freq_lo_khz + freq_span):
                continue   # outside display range
            y_px = (freq_khz - freq_lo_khz) / freq_span * N_FREQ_BINS
            line = pg.InfiniteLine(
                pos=y_px, angle=0, movable=False,
                pen=pg.mkPen(color=colour, width=1, style=Qt.DotLine),
                label=f"{name} {freq_khz}kHz",
                labelOpts={"position": 0.97, "color": colour,
                           "fill": (0, 0, 0, 120), "movable": False},
            )
            line.setVisible(False)
            self._plot.addItem(line)
            self._ref_lines.append(line)

        splitter.addWidget(self._plot)

        # ── Call detail view — high-resolution spectrogram of last saved clip ──
        clip_widget = QWidget()
        clip_vbox   = QVBoxLayout(clip_widget)
        clip_vbox.setContentsMargins(0, 4, 0, 0)
        clip_vbox.setSpacing(2)

        clip_header = QHBoxLayout()
        clip_header.setSpacing(4)

        self._clip_title = QLabel("<b>Call Detail</b> — no clip yet")
        clip_header.addWidget(self._clip_title)
        clip_header.addStretch()

        # ── Common pips filter ────────────────────────────────────────────────
        self._common_btn = QPushButton("Common pips: hidden")
        self._common_btn.setCheckable(True)
        self._common_btn.setChecked(True)
        self._common_btn.setToolTip(
            "Hide bounding boxes and labels for Common and Soprano Pipistrelle."
        )
        self._common_btn.toggled.connect(self._toggle_hide_common)
        clip_header.addWidget(self._common_btn)

        # ── Call boxes toggle ─────────────────────────────────────────────────
        self._boxes_btn = QPushButton("Call boxes: on")
        self._boxes_btn.setCheckable(True)
        self._boxes_btn.setChecked(True)
        self._boxes_btn.setToolTip("Show/hide BatDetect2 bounding boxes on the call detail panel")
        self._boxes_btn.toggled.connect(self._toggle_boxes)
        clip_header.addWidget(self._boxes_btn)

        clip_header.addSpacing(8)

        # ── Brightness +/− controls ───────────────────────────────────────────
        clip_header.addWidget(QLabel("Brightness:"))

        bright_minus = QPushButton("−")
        bright_minus.setFixedWidth(26)
        bright_minus.setToolTip("Decrease brightness (dimmer)")
        bright_minus.clicked.connect(self._brightness_down)
        clip_header.addWidget(bright_minus)

        self._bright_label = QLabel("0")
        self._bright_label.setFixedWidth(28)
        self._bright_label.setAlignment(Qt.AlignCenter)
        self._bright_label.setToolTip("0 = natural (spec.max()), + = brighter, − = dimmer")
        clip_header.addWidget(self._bright_label)

        bright_plus = QPushButton("+")
        bright_plus.setFixedWidth(26)
        bright_plus.setToolTip("Increase brightness (brighter)")
        bright_plus.clicked.connect(self._brightness_up)
        clip_header.addWidget(bright_plus)

        bright_reset = QPushButton("↺")
        bright_reset.setFixedWidth(26)
        bright_reset.setToolTip("Reset brightness to default")
        bright_reset.clicked.connect(self._brightness_reset)
        clip_header.addWidget(bright_reset)

        clip_vbox.addLayout(clip_header)

        self._bsg_label = QLabel("BSG-BAT: loading models…")
        self._bsg_label.setStyleSheet("color: #666666; font-size: 11px; padding: 0 2px;")
        clip_vbox.addWidget(self._bsg_label)

        self._clip_plot = pg.PlotWidget(background="k")
        self._clip_plot.setLabel("left",   "Frequency (kHz)")
        self._clip_plot.setLabel("bottom", "Time (ms)")

        self._clip_img = pg.ImageItem()
        self._clip_plot.addItem(self._clip_img)

        # Same inferno colourmap as the main waterfall
        try:
            clip_cmap = pg.colormap.get("inferno")
        except Exception:
            clip_cmap = cmap   # reuse fallback from above
        self._clip_img.setColorMap(clip_cmap)

        # Y-axis frequency ticks — same range as main display
        clip_freq_lo = FREQ_MIN_HZ / 1000
        clip_freq_hi = FREQ_MAX_HZ / 1000
        clip_y_ticks = [
            (int(round((f - clip_freq_lo) / (clip_freq_hi - clip_freq_lo) * N_CLIP_BINS)), f"{f}")
            for f in range(int(clip_freq_lo) + 5, int(clip_freq_hi), 10)
        ]
        self._clip_plot.getAxis("left").setTicks([clip_y_ticks])

        clip_vb = self._clip_plot.getViewBox()
        clip_vb.setMouseEnabled(x=True, y=False)   # horizontal pan/zoom, y locked
        clip_vb.setMouseMode(pg.ViewBox.PanMode)    # left-drag pans
        clip_vb.setMenuEnabled(False)
        clip_vb.rbScaleBox = None
        # Default limits before any clip loads — prevents runaway zoom on empty view
        clip_vb.setLimits(xMin=0, xMax=1, yMin=0, yMax=N_CLIP_BINS)
        self._clip_plot.showGrid(x=True, y=True, alpha=0.2)

        # Frequency crosshair for clip detail
        self._clip_hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(color=(255, 255, 255, 160), width=1, style=Qt.DashLine),
        )
        self._clip_hline.setVisible(False)
        self._clip_plot.addItem(self._clip_hline)

        self._clip_freq_label = pg.TextItem("", color=(255, 255, 255), anchor=(0, 1))
        self._clip_freq_label.setVisible(False)
        self._clip_plot.addItem(self._clip_freq_label)

        self._clip_plot.scene().sigMouseMoved.connect(self._on_clip_mouse_moved)
        self._clip_plot.viewport().installEventFilter(self)

        clip_vbox.addWidget(self._clip_plot)
        splitter.addWidget(clip_widget)

        # ── Detections table ───────────────────────────────────────────────────
        det_widget = QWidget()
        det_vbox   = QVBoxLayout(det_widget)
        det_vbox.setContentsMargins(0, 4, 0, 0)

        det_header = QHBoxLayout()
        det_header.addWidget(QLabel("<b>Recent Detections</b>"))
        det_header.addStretch()
        self._filter_btn = QPushButton("Filter < 40%")
        self._filter_btn.setCheckable(True)
        self._filter_btn.setChecked(True)
        self._filter_btn.setToolTip("Hide detections below 40% confidence")
        self._filter_btn.toggled.connect(self._toggle_low_conf_filter)
        det_header.addWidget(self._filter_btn)
        det_vbox.addLayout(det_header)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Time", "Species", "Confidence"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setDefaultSectionSize(120)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.itemClicked.connect(self._on_table_click)
        det_vbox.addWidget(self._table)

        splitter.addWidget(det_widget)
        splitter.setSizes([320, 260, 220])

        vbox.addWidget(splitter)

        # Status bar
        self._sb = self.statusBar()
        self._sb.showMessage("Life of Bats — select your pippyg device and click Start")
        SIGNALS.status.connect(self._sb.showMessage)

    def _populate_devices(self):
        self._device_combo.clear()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                ok = self._check_384k(i)
                badge = "✓" if ok else "✗"
                self._device_combo.addItem(f"[{i}] {badge}  {dev['name']}", i)

    def _populate_output_devices(self):
        """Fill the output combo. 'System default' is always first so AirPlay
        and other virtual macOS devices (which PortAudio cannot enumerate) are
        reachable: set them as the macOS system output, then pick this entry."""
        self._out_combo.blockSignals(True)
        self._out_combo.clear()
        # None = let PortAudio follow whatever macOS has set as system default
        self._out_combo.addItem("System default  (use macOS Sound settings)", None)
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                self._out_combo.addItem(dev["name"], i)
        self._out_combo.setCurrentIndex(0)   # default to system default
        self._out_combo.blockSignals(False)

    def _refresh_output_devices(self):
        prev = self._out_combo.currentData()
        self._populate_output_devices()
        for i in range(self._out_combo.count()):
            if self._out_combo.itemData(i) == prev:
                self._out_combo.setCurrentIndex(i)
                return
        # If previous device no longer found, fall back to system default
        self._out_combo.setCurrentIndex(0)

    def _on_output_device_changed(self, _index: int = 0):
        self._restart_output_stream(self._out_combo.currentData())

    def _restart_output_stream(self, device_idx):
        """Close the current output stream and reopen on device_idx.
        Pass device_idx=None to target the macOS system default output,
        which includes AirPlay and other virtual Core Audio devices."""
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None
        # Drain stale audio so it doesn't burst through on the new device
        while not _out_q.empty():
            try:
                _out_q.get_nowait()
            except queue.Empty:
                break
        try:
            self._out_stream = sd.OutputStream(
                device=device_idx,   # None → system default (picks up AirPlay)
                samplerate=OUTPUT_RATE,
                channels=1,
                dtype="float32",
                blocksize=_OUT_BLOCK,
                callback=self._output_callback,
            )
            self._out_stream.start()
            if device_idx is None:
                label = "system default"
            else:
                label = sd.query_devices(device_idx)["name"]
            self._sb.showMessage(f"Audio output → {label}")
        except Exception as e:
            self._sb.showMessage(f"Output device error: {e}")

    def _refresh_devices(self):
        # Force PortAudio to rescan — without this it returns the cached list
        # from startup and newly plugged USB devices never appear.
        # Stop the output stream first so sd._terminate() doesn't silently
        # kill it, then restart it afterwards.
        if self._stream is None:
            try:
                out_idx = self._out_combo.currentData()
                if self._out_stream is not None:
                    self._out_stream.stop()
                    self._out_stream.close()
                    self._out_stream = None
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
            self._restart_output_stream(out_idx)

        prev = self._device_combo.currentData()
        self._populate_devices()
        # Restore previous selection if it still exists
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == prev:
                self._device_combo.setCurrentIndex(i)
                break
        self._on_device_changed()

    @staticmethod
    def _check_384k(device_idx: int) -> bool:
        """Return True if the device natively operates at 384 kHz.

        sounddevice.check_input_settings() is unreliable on macOS because
        Core Audio silently resamples, making every device appear compatible.
        Instead we check the sample rate the device itself advertises — a
        genuine 384 kHz USB bat detector will report exactly that.
        """
        try:
            dev = sd.query_devices(device_idx)
            return int(dev["default_samplerate"]) == SAMPLE_RATE
        except Exception:
            return False

    def _on_device_changed(self, _index: int = 0):
        device_idx = self._device_combo.currentData()
        if device_idx is None:
            return
        if self._check_384k(device_idx):
            self._compat_label.setText("✓ Supports 384 kHz")
            self._compat_label.setStyleSheet("color: #4caf50;")
        else:
            self._compat_label.setText("✗ Won't work — not a 384 kHz device")
            self._compat_label.setStyleSheet("color: #e74c3c;")

    # ── Audio callback (real-time thread) ─────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()

        # Apply notch filter before anything else so both the waterfall and
        # the audio output see the suppressed signal
        if self._notch_enabled:
            chunk, self._notch_zi = sosfilt(
                self._notch_sos, chunk.astype(np.float64), zi=self._notch_zi
            )
            chunk = chunk.astype(np.float32)

        # Push to spectrogram queue (non-blocking)
        try:
            _spec_q.put_nowait(chunk)
        except queue.Full:
            pass

        # Push to heterodyne queue for audio output
        if self._audio_enabled[0]:
            try:
                _hetero_q.put_nowait(chunk)
            except queue.Full:
                pass

        # ── Clip accumulation ──
        if self._clip_active:
            self._clip_buf.append(chunk)
            self._clip_samples_collected += len(chunk)
            if self._clip_samples_collected >= CLIP_SAMPLES:
                audio = np.concatenate(self._clip_buf)
                self._clip_buf = []
                self._clip_samples_collected = 0
                self._clip_active = False
                self._last_trigger_time = time.monotonic()
                try:
                    _clip_audio_q.put_nowait(audio)
                except queue.Full:
                    pass   # analysis backlogged; drop clip
        else:
            # Lightweight trigger check
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > self._rms_thresh:
                now = time.monotonic()
                if now - self._last_trigger_time > DEBOUNCE_SECS:
                    # Ultrasonic energy check
                    mag   = np.abs(np.fft.rfft(chunk))
                    freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)
                    ultra = float(np.sum(mag[freqs > ULTRA_MIN_HZ]))
                    total = float(np.sum(mag))
                    if total > 0 and (ultra / total) > ULTRA_RATIO:
                        self._clip_active = True
                        self._clip_buf = [chunk]
                        self._clip_samples_collected = len(chunk)

    # ── Waterfall refresh (GUI timer) ─────────────────────────────────────────
    def _tick(self):
        new_cols = []
        try:
            while True:
                chunk = _spec_q.get_nowait()
                # Compute FFT with Hanning window
                padded = chunk[:FFT_SIZE] if len(chunk) >= FFT_SIZE \
                         else np.pad(chunk, (0, FFT_SIZE - len(chunk)))
                mag = np.abs(np.fft.rfft(padded * _WINDOW))[_BIN_LO:_BIN_HI]
                new_cols.append(np.log1p(mag).astype(np.float32))
        except queue.Empty:
            pass

        if not new_cols:
            return

        n = len(new_cols)
        if n >= WATERFALL_COLS:
            self._waterfall[:] = np.array(new_cols[-WATERFALL_COLS:])
        else:
            self._waterfall = np.roll(self._waterfall, -n, axis=0)
            self._waterfall[-n:, :] = np.array(new_cols)

        # Rolling max with slow decay — stable contrast without per-frame flashing
        frame_max = float(self._waterfall.max())
        self._level_max = max(self._level_max * 0.99, frame_max, 0.1)
        self._img.setImage(self._waterfall, autoLevels=False, levels=(0.0, self._level_max))

        # Recording indicator dot
        recording = self._clip_active
        if recording != self._prev_clip_active:
            self._prev_clip_active = recording
            if recording:
                self._rec_dot.setStyleSheet("color: #e74c3c; font-size: 18px;")  # red
            else:
                self._rec_dot.setStyleSheet("color: #444444; font-size: 18px;")  # grey

    # ── Detection result (Qt slot, GUI thread) ────────────────────────────────
    def _on_detection(self, ts: str, species: str, confidence: float):
        # Insert a recording-separator row the first time we see a new timestamp,
        # so each clip's results are visually grouped. Newest is always at top.
        if ts != self._last_detection_ts:
            self._last_detection_ts = ts
            self._table.insertRow(0)
            sep_item = QTableWidgetItem(f"── {ts} ──")
            sep_item.setForeground(QColor("#888888"))
            sep_item.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(0, 0, sep_item)
            self._table.setSpan(0, 0, 1, 3)

        # Insert species row just below the separator (row 1) so each new
        # species within the same clip appears under its timestamp header.
        insert_at = 1 if self._table.rowCount() > 0 else 0
        self._table.insertRow(insert_at)

        self._table.setItem(insert_at, 0, QTableWidgetItem(ts))

        common = _COMMON_NAMES.get(species, species)   # fall back to Latin if unknown
        sp_item = QTableWidgetItem(common)
        if species != common:
            sp_item.setToolTip(species)                # hover to see Latin name
        if confidence >= 0.7:
            sp_item.setForeground(QColor("#4caf50"))   # green
        elif confidence >= 0.4:
            sp_item.setForeground(QColor("#ff9800"))   # amber

        self._table.setItem(insert_at, 1, sp_item)

        conf_text = f"{round(confidence * 100)}%" if confidence > 0 else "—"
        self._table.setItem(insert_at, 2, QTableWidgetItem(conf_text))

        if self._hide_low_conf and confidence < 0.40:
            self._table.setRowHidden(insert_at, True)

        # Keep at most 200 rows (prune oldest from the bottom)
        while self._table.rowCount() > 200:
            self._table.removeRow(self._table.rowCount() - 1)

    # ── Call detail spectrogram (Qt slot, GUI thread) ────────────────────────
    def _on_clip_spec(self, spec: np.ndarray, ts: str, annotations: list):
        """Cache the clip and render it — unless the view is pinned to a
        different recording, in which case just cache and return.
        """
        # Cache for later replay (keyed by timestamp, most-recent-last)
        self._clip_cache[ts] = (spec, annotations)
        # Trim to last 50 clips to bound memory use
        while len(self._clip_cache) > 50:
            self._clip_cache.pop(next(iter(self._clip_cache)))

        # Don't overwrite the pinned view with a new live clip
        if self._pinned_ts is not None and self._pinned_ts != ts:
            return

        self._render_clip(spec, ts, annotations)

    def _render_clip(self, spec: np.ndarray, ts: str, annotations: list):
        """Render the high-resolution spectrogram of a clip,
        with BatDetect2 bounding boxes and species labels overlaid.
        """
        self._displayed_ts = ts   # track exactly what's on screen for brightness controls

        # Remove previous annotation overlays
        for item in self._clip_boxes:
            self._clip_plot.removeItem(item)
        self._clip_boxes.clear()

        if spec.shape[0] < 2:
            return

        # spec.max() as natural baseline; brightness multiplier shifts it
        # (< 1.0 = brighter, > 1.0 = dimmer — set via slider)
        clip_ceil = float(spec.max()) * self._brightness
        self._clip_img.setImage(spec, autoLevels=False,
                                levels=(0.0, max(clip_ceil, 0.01)))

        # Fit the view to the image extent — show full clip on load so nothing
        # is hidden; user can scroll/pinch-zoom in to inspect individual calls
        n_frames, n_bins = spec.shape
        ms_per_frame = (CLIP_HOP / SAMPLE_RATE) * 1000.0   # ≈ 0.333 ms per frame
        total_ms     = n_frames * ms_per_frame

        self._clip_plot.setXRange(0, n_frames, padding=0)
        self._clip_plot.setYRange(0, n_bins, padding=0)

        # Constrain scrolling to the actual clip extent
        self._clip_plot.getViewBox().setLimits(
            xMin=0, xMax=n_frames, yMin=0, yMax=n_bins
        )

        # X-axis ticks every 50 ms across the full clip length
        x_ticks = []
        t_ms = 0.0
        while t_ms <= total_ms + 1:
            x_ticks.append((t_ms / ms_per_frame, f"{t_ms:.0f}"))
            t_ms += 50.0
        self._clip_plot.getAxis("bottom").setTicks([x_ticks])

        # ── BatDetect2 bounding boxes ──────────────────────────────────────────
        freq_range = FREQ_MAX_HZ - FREQ_MIN_HZ
        for ann in annotations:
            prob = float(ann.get("class_prob", 0.0))
            if prob < 0.2:          # skip very low confidence detections
                continue
            if self._hide_common and ann.get("class") in _SURVEY_COMMON:
                continue            # suppress common/soprano pip boxes

            # Convert time (seconds) and frequency (Hz) to plot coordinates
            x0 = ann.get("start_time", 0.0) * SAMPLE_RATE / CLIP_HOP
            x1 = ann.get("end_time",   0.0) * SAMPLE_RATE / CLIP_HOP
            y0 = (float(ann.get("low_freq",  FREQ_MIN_HZ)) - FREQ_MIN_HZ) / freq_range * N_CLIP_BINS
            y1 = (float(ann.get("high_freq", FREQ_MAX_HZ)) - FREQ_MIN_HZ) / freq_range * N_CLIP_BINS

            if x1 <= x0 or y1 <= y0:
                continue

            # Colour coding for text labels
            if prob >= 0.7:
                text_colour = "#4caf50"   # green
            elif prob >= 0.4:
                text_colour = "#ff9800"   # amber
            else:
                text_colour = "#f44336"   # red

            # Thin dark grey box — subtle outline, no fill
            rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
            rect.setPen(pg.mkPen((110, 110, 110), width=1))
            rect.setBrush(pg.mkBrush(0, 0, 0, 0))   # fully transparent fill
            self._clip_plot.addItem(rect)
            self._clip_boxes.append(rect)

            # Species label just above the box, colour-coded by confidence
            species = ann.get("class", "")
            common  = _COMMON_NAMES.get(species, species)
            label   = pg.TextItem(
                f"{common}  {round(prob * 100)}%",
                color=text_colour, anchor=(0, 1),
            )
            label.setPos(x0, y1)
            self._clip_plot.addItem(label)
            self._clip_boxes.append(label)

        # Respect the current boxes toggle state
        show = self._boxes_btn.isChecked()
        for item in self._clip_boxes:
            item.setVisible(show)

        # Title shows pin state
        if self._pinned_ts == ts:
            self._clip_title.setText(
                f"<b>Call Detail</b> — {ts} &nbsp;📌&nbsp;"
                f"<small style='color:#888'>pinned — click row again to unpin</small>"
            )
        else:
            self._clip_title.setText(f"<b>Call Detail</b> — {ts}")

    # ── Table row click — pin / unpin a recording ─────────────────────────────
    def _get_row_ts(self, row: int) -> str | None:
        """Extract the HH:MM:SS timestamp from any row (separator or species)."""
        item = self._table.item(row, 0)
        if item is None:
            return None
        text = item.text().strip()
        # Separator rows look like "── 14:23:45 ──"
        if text.startswith("─"):
            text = text.strip("─ ")
        return text if text else None

    def _on_table_click(self, item: QTableWidgetItem):
        # ── Survey mode: load the WAV the row refers to ───────────────────────
        if self._survey_mode:
            # Always read wav path from column 0 of the clicked row so any
            # column click (including the dynamically-filled BSG-BAT cell) works
            col0 = self._table.item(item.row(), 0)
            wav_str = col0.data(Qt.UserRole) if col0 else None
            if not wav_str:
                self._table.clearSelection()
                return
            self._load_survey_wav(Path(wav_str))
            return

        # ── Live mode: pin / unpin ────────────────────────────────────────────
        ts = self._get_row_ts(item.row())
        if ts is None:
            self._table.clearSelection()
            return

        if self._pinned_ts == ts:
            # Unpin: return to live view
            self._pinned_ts = None
            self._table.clearSelection()
            if self._clip_cache:
                latest_ts = list(self._clip_cache.keys())[-1]
                spec, anns = self._clip_cache[latest_ts]
                self._render_clip(spec, latest_ts, anns)
            else:
                self._clip_title.setText("<b>Call Detail</b> — live (no clip yet)")
        else:
            # Pin to this recording
            if ts not in self._clip_cache:
                self._table.clearSelection()
                return
            self._pinned_ts = ts
            spec, anns = self._clip_cache[ts]
            self._render_clip(spec, ts, anns)

    # ── Survey browser ────────────────────────────────────────────────────────
    def _browse_survey(self):
        """Open a folder picker and load interesting recordings from that survey night."""
        start = str(BAT_DATA_DIR) if BAT_DATA_DIR.exists() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select survey folder", start)
        if folder:
            self._load_survey_folder(Path(folder))

    def _load_survey_folder(self, folder: Path):
        """Scan a survey folder, populate the table with recordings that have
        interesting (non-common-pip) detections and a WAV file still present.
        """
        det_dir = folder / "detections"
        if not det_dir.exists():
            self._sb.showMessage(f"No detections folder in {folder.name} — run BatDetect2 first.")
            return

        entries = []   # (ts_display, wav_path, all_annotations, interesting_annotations)
        for wav in sorted(
            w for w in folder.iterdir()
            if w.suffix.lower() == ".wav" and not w.name.startswith("._")
        ):
            json_path = det_dir / (wav.name + ".json")
            if not json_path.exists():
                continue   # not yet processed
            try:
                data = _json.loads(json_path.read_text())
            except Exception:
                continue
            all_anns = data.get("annotation", [])
            interesting = [
                a for a in all_anns
                if float(a.get("class_prob", 0)) >= _SURVEY_THRESH
                and a.get("class") not in _SURVEY_COMMON
            ]
            if not interesting:
                continue   # only common/soprano pips — skip
            m = re.search(r"(\d{8})_(\d{6})", wav.name)
            t = m.group(2) if m else "000000"
            ts = f"{t[:2]}:{t[2:4]}:{t[4:]}"
            entries.append((ts, wav, all_anns, interesting))

        if not entries:
            self._sb.showMessage(
                f"No interesting calls found in {folder.name} "
                f"(all WAVs are common/soprano pip only, or none processed yet)."
            )
            return

        # Enter survey mode
        self._survey_mode = True
        self._pinned_ts   = None
        self._clip_cache.clear()

        # Switch to 5-column layout for survey mode
        self._table.setRowCount(0)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Time", "File", "BatDetect2", "Conf.", "BSG-BAT"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 65)
        self._table.setColumnWidth(1, 175)
        self._table.setColumnWidth(3, 50)

        for ts, wav, all_anns, interesting_anns in entries:
            # Deduplicate: keep best confidence per species across all detections
            best: dict[str, float] = {}
            for ann in interesting_anns:
                sp   = ann.get("class", "")
                prob = float(ann.get("class_prob", 0))
                if prob > best.get(sp, 0.0):
                    best[sp] = prob

            # Build display strings — sorted by confidence descending
            ranked   = sorted(best.items(), key=lambda x: -x[1])
            sp_names = "  ·  ".join(_COMMON_NAMES.get(sp, sp) for sp, _ in ranked)
            sp_tips  = "\n".join(f"{_COMMON_NAMES.get(sp,sp)}: {round(p*100)}%" for sp, p in ranked)
            top_conf = ranked[0][1] if ranked else 0.0

            r = self._table.rowCount()
            self._table.insertRow(r)

            ts_item = QTableWidgetItem(ts)
            ts_item.setData(Qt.UserRole, str(wav))
            self._table.setItem(r, 0, ts_item)

            file_item = QTableWidgetItem(wav.stem)   # filename without .wav
            file_item.setToolTip(wav.name)
            file_item.setData(Qt.UserRole, str(wav))
            file_item.setForeground(QColor("#888888"))
            self._table.setItem(r, 1, file_item)

            sp_item = QTableWidgetItem(sp_names)
            sp_item.setToolTip(sp_tips)
            sp_item.setData(Qt.UserRole, str(wav))
            if top_conf >= 0.7:
                sp_item.setForeground(QColor("#4caf50"))
            elif top_conf >= 0.4:
                sp_item.setForeground(QColor("#ff9800"))
            self._table.setItem(r, 2, sp_item)

            conf_item = QTableWidgetItem(f"{round(top_conf * 100)}%")
            conf_item.setData(Qt.UserRole, str(wav))
            self._table.setItem(r, 3, conf_item)

            # BSG-BAT column — placeholder until background thread fills it in
            bsg_placeholder = QTableWidgetItem("…")
            bsg_placeholder.setForeground(QColor("#555555"))
            bsg_placeholder.setData(Qt.UserRole, str(wav))
            self._table.setItem(r, 4, bsg_placeholder)

        # Build wav → row map so BSG results can update the right cell
        self._survey_row_map = {str(wav): idx for idx, (_, wav, _, _) in enumerate(entries)}

        # Run BSG-BAT on every interesting file in a background thread.
        # If models are still loading, the thread waits up to 2 minutes.
        def _run_bsg_survey(entry_list):
            wait = 0.0
            while not BSG.available and not BSG.error and wait < 120:
                time.sleep(0.5)
                wait += 0.5
            if not BSG.available:
                return
            for _, wav, _, _ in entry_list:
                try:
                    res = BSG.classify(str(wav))
                    if res:
                        SIGNALS.bsg_file_result.emit(str(wav), res)
                except Exception:
                    pass

        threading.Thread(
            target=_run_bsg_survey, args=(entries,), daemon=True
        ).start()

        n = len(entries)
        self._survey_banner.setText(
            f"🦇  Survey: {folder.name}  ·  "
            f"{n} interesting recording{'s' if n != 1 else ''}  —  click any row to view"
        )
        self._survey_banner.setVisible(True)
        self._live_btn.setVisible(True)
        self._survey_btn.setVisible(False)
        self._clip_title.setText("<b>Call Detail</b> — click a row to view")
        self._sb.showMessage(f"Survey loaded: {n} interesting recordings in {folder.name}")

    def _load_survey_wav(self, wav_path: Path):
        """Load a WAV file and its existing JSON annotations into the Call Detail
        panel without re-running BatDetect2 — the JSON already has the results.
        """
        self._sb.showMessage(f"Loading {wav_path.name} …")

        def _work():
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                if sr != SAMPLE_RATE:
                    g     = gcd(SAMPLE_RATE, sr)
                    audio = resample_poly(
                        audio, SAMPLE_RATE // g, sr // g
                    ).astype(np.float32)

                spec = _compute_clip_spec(audio)

                json_path = wav_path.parent / "detections" / (wav_path.name + ".json")
                annotations = []
                if json_path.exists():
                    data        = _json.loads(json_path.read_text())
                    annotations = data.get("annotation", [])

                m  = re.search(r"(\d{8})_(\d{6})", wav_path.name)
                t  = m.group(2) if m else "000000"
                ts = f"{t[:2]}:{t[2:4]}:{t[4:]}"

                SIGNALS.clip_spec.emit(spec, ts, annotations)
                SIGNALS.status.emit(f"Loaded {wav_path.name}")

                # BSG-BAT second opinion (non-blocking; skipped if models not ready)
                if BSG.available:
                    bsg = BSG.classify(str(wav_path))
                    if bsg:
                        SIGNALS.bsg_result.emit(ts, bsg)

            except Exception as e:
                SIGNALS.status.emit(f"Error loading {wav_path.name}: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _exit_survey(self):
        """Leave survey mode and return to live monitoring."""
        self._survey_mode = False
        self._pinned_ts   = None
        self._clip_cache.clear()
        self._survey_row_map.clear()
        self._table.setRowCount(0)
        self._last_detection_ts = ""
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Time", "Species", "Confidence"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        self._table.setColumnWidth(0, 120)
        self._table.setColumnWidth(2, 120)
        self._survey_banner.setVisible(False)
        self._live_btn.setVisible(False)
        self._survey_btn.setVisible(True)
        self._clip_title.setText("<b>Call Detail</b> — no clip yet")
        self._sb.showMessage("Returned to live mode.")

    # ── Sensitivity slider ────────────────────────────────────────────────────
    @staticmethod
    def _slider_from_thresh(thresh: float) -> int:
        """Map RMS threshold (0.001–0.050) → level (10–1), inverted."""
        return max(1, min(10, round((0.051 - thresh) / 0.049 * 9) + 1))

    def _on_sensitivity_changed(self, value: int):
        # Level 10 = most sensitive (thresh 0.001), level 1 = least (0.050)
        self._rms_thresh = round(0.050 - (value - 1) / 9 * 0.049, 4)
        self._sens_label.setText(f"Level {value}/10")

    # ── Species reference lines toggle ────────────────────────────────────────
    def _toggle_refs(self, checked: bool):
        self._refs_visible = checked
        for line in self._ref_lines:
            line.setVisible(checked)

    def _apply_brightness(self):
        """Recompute the brightness multiplier from _bright_step and update the
        image levels without touching the pan/zoom state."""
        # Log scale: each step ≈ ×1.33 brighter or dimmer
        # step  0 → 1.0  (spec.max(), original look)
        # step +9 → 0.13 (very bright)
        # step −9 → 7.9  (very dim)
        self._brightness = 10.0 ** (-self._bright_step / 9.0)
        sign = "+" if self._bright_step > 0 else ""
        self._bright_label.setText(f"{sign}{self._bright_step}")
        # Use _displayed_ts — the timestamp of whatever is actually on screen.
        # Avoids the stale-cache-key bug where async BSG/clip_spec signals
        # insert a different entry as the "last" key after the user has moved on.
        ts = self._displayed_ts
        if ts and ts in self._clip_cache:
            spec, _ = self._clip_cache[ts]
            clip_ceil = float(spec.max()) * self._brightness
            self._clip_img.setImage(spec, autoLevels=False,
                                    levels=(0.0, max(clip_ceil, 0.01)))

    def _brightness_up(self):
        if self._bright_step < 9:
            self._bright_step += 1
            self._apply_brightness()

    def _brightness_down(self):
        if self._bright_step > -9:
            self._bright_step -= 1
            self._apply_brightness()

    def _brightness_reset(self):
        self._bright_step = 0
        self._apply_brightness()

    def _toggle_hide_common(self, checked: bool):
        self._hide_common = checked
        self._common_btn.setText(
            "Common pips: hidden" if checked else "Common pips: visible"
        )
        # Re-render the current clip so the change takes effect immediately
        if self._clip_cache:
            ts = self._pinned_ts or list(self._clip_cache.keys())[-1]
            if ts in self._clip_cache:
                spec, anns = self._clip_cache[ts]
                self._render_clip(spec, ts, anns)

    def _toggle_boxes(self, checked: bool):
        self._boxes_btn.setText("Call boxes: on" if checked else "Call boxes: off")
        for item in self._clip_boxes:
            item.setVisible(checked)

    # ── Event filter — hide crosshairs when mouse leaves either plot ─────────
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Leave:
            if obj is self._plot.viewport():
                self._hline.setVisible(False)
                self._freq_label.setVisible(False)
            elif obj is self._clip_plot.viewport():
                self._clip_hline.setVisible(False)
                self._clip_freq_label.setVisible(False)
        return super().eventFilter(obj, event)

    # ── Low-confidence filter toggle ──────────────────────────────────────────
    def _toggle_low_conf_filter(self, checked: bool):
        self._hide_low_conf = checked
        # Table may not exist yet if called during UI construction
        if not hasattr(self, '_table'):
            return
        # Show/hide existing rows whose confidence is below 0.40
        for row in range(self._table.rowCount()):
            conf_item = self._table.item(row, 2)
            if conf_item is None:
                continue
            try:
                conf = float(conf_item.text().rstrip("%")) / 100
            except ValueError:
                conf = 0.0
            self._table.setRowHidden(row, checked and conf < 0.40)

    # ── Mouse crosshair ───────────────────────────────────────────────────────
    def _on_mouse_moved(self, scene_pos):
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            self._hline.setVisible(False)
            self._freq_label.setVisible(False)
            return

        pt = self._plot.getViewBox().mapSceneToView(scene_pos)
        y = pt.y()
        if 0 <= y <= N_FREQ_BINS:
            freq_lo  = FREQ_AXIS_LO / 1000
            freq_khz = freq_lo + (y / N_FREQ_BINS) * ((FREQ_AXIS_HI - FREQ_AXIS_LO) / 1000)
            self._hline.setValue(y)
            self._freq_label.setText(f" {freq_khz:.1f} kHz")
            self._freq_label.setPos(2, y)   # pin label to left edge
            self._hline.setVisible(True)
            self._freq_label.setVisible(True)
        else:
            self._hline.setVisible(False)
            self._freq_label.setVisible(False)

    # ── Clip detail crosshair ─────────────────────────────────────────────────
    def _on_clip_mouse_moved(self, scene_pos):
        if not self._clip_plot.sceneBoundingRect().contains(scene_pos):
            self._clip_hline.setVisible(False)
            self._clip_freq_label.setVisible(False)
            return

        pt = self._clip_plot.getViewBox().mapSceneToView(scene_pos)
        y = pt.y()
        if 0 <= y <= N_CLIP_BINS:
            freq_khz = (FREQ_MIN_HZ + y / N_CLIP_BINS * (FREQ_MAX_HZ - FREQ_MIN_HZ)) / 1000
            self._clip_hline.setValue(y)
            self._clip_freq_label.setText(f" {freq_khz:.1f} kHz")
            # Pin label to the left edge of the current view
            x_left = self._clip_plot.getViewBox().viewRange()[0][0]
            self._clip_freq_label.setPos(x_left, y)
            self._clip_hline.setVisible(True)
            self._clip_freq_label.setVisible(True)
        else:
            self._clip_hline.setVisible(False)
            self._clip_freq_label.setVisible(False)

    # ── File playback ─────────────────────────────────────────────────────────
    def _open_file(self):
        # If already playing, act as a stop button
        if self._is_playing:
            self._playback_stop.set()
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open bat recording",
            str(Path(__file__).resolve().parent.parent / "bat_detector"),
            "WAV files (*.wav);;All files (*)"
        )
        if not path:
            return

        self._is_playing = True
        self._playback_stop.clear()
        self._file_btn.setText("⏹  Stop playback")
        self._sb.showMessage(f"Loading {os.path.basename(path)} …")
        if not self._timer.isActive():
            self._timer.start()

        def _play():
            name = os.path.basename(path)
            try:
                audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio[:, 0]

                if file_sr != SAMPLE_RATE:
                    g = gcd(SAMPLE_RATE, file_sr)
                    audio = resample_poly(
                        audio, SAMPLE_RATE // g, file_sr // g
                    ).astype(np.float32)

                total_secs    = len(audio) / SAMPLE_RATE
                chunk_secs    = CHUNK_SIZE / SAMPLE_RATE      # ~21.3 ms
                n_chunks      = -(-len(audio) // CHUNK_SIZE)  # ceiling div
                playback_start = time.monotonic()

                # Queue whole file for BatDetect2 upfront so results come in
                # progressively while playback runs
                _clip_audio_q.put(audio)

                for idx in range(n_chunks):
                    if self._playback_stop.is_set():
                        break

                    start = idx * CHUNK_SIZE
                    chunk = audio[start : start + CHUNK_SIZE]
                    if len(chunk) < CHUNK_SIZE:
                        chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

                    # Spectrogram display
                    try:
                        _spec_q.put_nowait(chunk)
                    except queue.Full:
                        pass

                    # Heterodyne audio
                    if self._audio_enabled[0]:
                        try:
                            _hetero_q.put_nowait(chunk)
                        except queue.Full:
                            pass

                    # Progress update every ~0.5 s
                    elapsed_file = idx * chunk_secs
                    if idx % 24 == 0:
                        SIGNALS.status.emit(
                            f"▶  {name}  —  "
                            f"{elapsed_file:.1f}s / {total_secs:.1f}s"
                        )

                    # Sleep until the next chunk's real-time deadline
                    deadline = playback_start + (idx + 1) * chunk_secs
                    gap = deadline - time.monotonic()
                    if gap > 0:
                        time.sleep(gap)

                if not self._playback_stop.is_set():
                    SIGNALS.status.emit(f"Playback complete — {name}")
                else:
                    SIGNALS.status.emit(f"Playback stopped — {name}")

            except Exception as e:
                SIGNALS.status.emit(f"Playback error: {e}")
            finally:
                SIGNALS.file_done.emit()

        threading.Thread(target=_play, daemon=True).start()

    def _on_file_done(self):
        self._is_playing = False
        self._playback_stop.clear()
        self._file_btn.setText("📂  Open Recording")
        self._file_btn.setEnabled(True)

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start(self):
        device_idx = self._device_combo.currentData()
        if device_idx is None:
            self._sb.showMessage("No input device selected.")
            return
        if not self._check_384k(device_idx):
            self._sb.showMessage(
                "This device doesn't support 384 kHz — please select your pippyg bat detector."
            )
            return

        try:
            self._stream = sd.InputStream(
                device=device_idx,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._sb.showMessage(f"Could not open device: {e}")
            return

        self._timer.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._device_combo.setEnabled(False)
        self._sb.showMessage(f"Listening — {self._device_combo.currentText()}")

    def _stop(self):
        self._timer.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._device_combo.setEnabled(True)
        self._sb.showMessage("Stopped.")

    def closeEvent(self, event):
        self._playback_stop.set()
        self._stop()
        if self._out_stream is not None:
            self._out_stream.stop()
            self._out_stream.close()
        event.accept()

    # ── Heterodyne audio controls ─────────────────────────────────────────────
    def _toggle_audio(self, checked: bool):
        self._audio_enabled[0] = checked
        self._audio_btn.setText("🔊  Audio on" if checked else "🔇  Audio off")

    # ── Notch filter ──────────────────────────────────────────────────────────
    @staticmethod
    def _build_notch(freq_hz: float):
        """Return (sos, zi) for a narrow IIR notch at freq_hz."""
        w0  = freq_hz / (SAMPLE_RATE / 2)
        b, a = iirnotch(w0, Q=30)
        # Pack into a single-section SOS row so we can use sosfilt with state
        sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]], dtype=np.float64)
        zi  = sosfilt_zi(sos)
        return sos, zi

    def _toggle_notch(self, checked: bool):
        self._notch_enabled = checked
        if checked:
            # Reset state so there's no click when the filter kicks in
            _, self._notch_zi = self._build_notch(self._notch_freq_hz)
            self._notch_btn.setText("Noise filter: on")
        else:
            self._notch_btn.setText("Noise filter: off")

    def _on_notch_freq_changed(self, value: float):
        self._notch_freq_hz = value * 1000
        self._notch_sos, self._notch_zi = self._build_notch(self._notch_freq_hz)

    def _output_callback(self, outdata, frames, time_info, status):
        """Called by sounddevice in its output thread — must be fast."""
        try:
            chunk = _out_q.get_nowait()
            n = min(len(chunk), frames)
            outdata[:n, 0] = chunk[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        except queue.Empty:
            outdata[:, 0] = 0.0

    # ── BSG-BAT helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _parse_bsg(results: dict):
        """Return (hits, bg_prob).  hits = [(species_key, prob)] sorted desc, ≥20%.

        The threshold is deliberately low so that secondary species (e.g. Serotine
        at 25% alongside Common Pip at 99%) still appear — colour-coding conveys
        confidence, hiding them entirely causes more confusion than showing them dimly.
        """
        bg_prob = results.get("Background", 0.0)
        hits = [
            (sp, p) for sp, p in results.items()
            if sp != "Background" and p >= 0.20
        ]
        hits.sort(key=lambda x: -x[1])
        return hits, bg_prob

    # ── BSG-BAT result handler — live clips (Qt slot, GUI thread) ─────────────
    def _on_bsg_result(self, ts: str, results: dict):
        """Handle a BSG-BAT result for a live clip.

        Always annotates the matching separator row in the live detections table
        so the result is visible even when another clip is pinned in Call Detail.
        Only updates the clip-panel label when the result is for the shown clip.
        """
        # ── Sentinels ─────────────────────────────────────────────────────────
        if ts == "__ready__":
            if not self._clip_cache:
                self._bsg_label.setText("BSG-BAT: ready — waiting for first clip")
                self._bsg_label.setStyleSheet(
                    "color: #4caf50; font-size: 11px; padding: 0 2px;"
                )
            return

        if ts == "__error__":
            err = results.get("msg", "load failed")
            self._bsg_label.setText(f"BSG-BAT unavailable: {err}")
            self._bsg_label.setStyleSheet(
                "color: #e74c3c; font-size: 10px; padding: 0 2px;"
            )
            self._bsg_label.setToolTip(
                "BSG-BAT models could not be loaded.\n"
                "Check that ~/Desktop/bsgbat/ exists with models/ and code/ folders,\n"
                "and that torch is installed in this venv."
            )
            return

        hits, bg_prob = self._parse_bsg(results)

        # ── Annotate separator row in live table ──────────────────────────────
        if not self._survey_mode:
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item and f"── {ts} ──" in item.text() and "BSG:" not in item.text():
                    if hits:
                        top_name = _BSG_NAMES.get(hits[0][0], hits[0][0])
                        bsg_str  = f"{top_name} {round(hits[0][1] * 100)}%"
                        colour   = "#4caf50" if hits[0][1] >= 0.70 else "#ff9800"
                    elif bg_prob >= 0.70:
                        bsg_str = f"Background {round(bg_prob * 100)}%"
                        colour  = "#e74c3c"
                    else:
                        bsg_str, colour = None, None
                    if bsg_str:
                        item.setText(f"── {ts} ──  ·  BSG: {bsg_str}")
                        item.setForeground(QColor(colour))
                    break

        # ── Update clip-panel label only for the currently displayed clip ─────
        current_ts = self._pinned_ts or self._displayed_ts
        if ts != current_ts:
            return

        if not hits:
            if bg_prob >= 0.70:
                text   = f"BSG-BAT: Background {round(bg_prob * 100)}% — likely not a bat"
                colour = "#e74c3c"
            else:
                text   = "BSG-BAT: no confident ID"
                colour = "#888888"
        else:
            parts = "  ·  ".join(
                f"{_BSG_NAMES.get(sp, sp)} {round(p * 100)}%"
                for sp, p in hits[:3]
            )
            if bg_prob >= 0.50:
                parts += f"  (Background {round(bg_prob * 100)}%)"
            text   = f"BSG-BAT: {parts}"
            colour = "#4caf50" if hits[0][1] >= 0.70 else "#ff9800"

        self._bsg_label.setText(text)
        self._bsg_label.setStyleSheet(
            f"color: {colour}; font-size: 11px; padding: 0 2px;"
        )

    # ── BSG-BAT result handler — survey files (Qt slot, GUI thread) ───────────
    def _on_bsg_file_result(self, wav_path_str: str, results: dict):
        """Fill in the BSG-BAT column for a survey table row as results arrive."""
        if not self._survey_mode:
            return
        row = self._survey_row_map.get(wav_path_str)
        if row is None or row >= self._table.rowCount():
            return

        hits, bg_prob = self._parse_bsg(results)

        if not hits:
            if bg_prob >= 0.70:
                text   = f"Background {round(bg_prob * 100)}%"
                colour = "#e74c3c"
            else:
                text   = "—"
                colour = "#555555"
        else:
            text   = "  ·  ".join(
                f"{_BSG_NAMES.get(sp, sp)} {round(p * 100)}%"
                for sp, p in hits[:2]
            )
            top_p  = hits[0][1]
            colour = "#4caf50" if top_p >= 0.70 else ("#ff9800" if top_p >= 0.40 else "#888888")

        col0    = self._table.item(row, 0)
        wav_key = col0.data(Qt.UserRole) if col0 else wav_path_str

        bsg_item = QTableWidgetItem(text)
        bsg_item.setForeground(QColor(colour))
        bsg_item.setData(Qt.UserRole, wav_key)
        self._table.setItem(row, 4, bsg_item)

    # ── Workers ───────────────────────────────────────────────────────────────
    def _start_workers(self):
        threading.Thread(target=_save_and_analyse_worker, daemon=True).start()
        threading.Thread(
            target=_freqdiv_worker,
            args=(self._audio_enabled, self._volume),
            daemon=True,
        ).start()

        # Load BSG-BAT models in the background so the app is immediately usable.
        # Status messages go to the status bar; once loaded (or failed), the label updates.
        def _load_bsg():
            def _cb(msg):
                SIGNALS.status.emit(msg)
                if "ready ✓" in msg:
                    SIGNALS.bsg_result.emit("__ready__", {})
                elif "load failed" in msg or "failed" in msg.lower():
                    SIGNALS.bsg_result.emit("__error__", {"msg": msg})

            BSG.load(status_cb=_cb)
            # Fallback: if load() returned without firing either sentinel
            if not BSG.available:
                err = BSG.error or "unknown error — check BSG-BAT installation"
                SIGNALS.bsg_result.emit("__error__", {"msg": err})

        threading.Thread(target=_load_bsg, daemon=True).start()

        SIGNALS.detection.connect(self._on_detection)
        SIGNALS.clip_spec.connect(self._on_clip_spec)
        SIGNALS.file_done.connect(self._on_file_done)
        SIGNALS.file_done.connect(lambda: self._file_btn.setEnabled(True))
        SIGNALS.bsg_result.connect(self._on_bsg_result)
        SIGNALS.bsg_file_result.connect(self._on_bsg_file_result)

        # Output stream — open on whichever device is selected in the combo
        self._restart_output_stream(self._out_combo.currentData())


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
