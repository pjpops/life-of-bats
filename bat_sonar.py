#!/usr/bin/env python3
"""
bat_sonar.py — Live bat sonogram viewer with BatDetect2 species detection.

Connect your pippyg USB bat detector, select the device from the dropdown,
and click Start. Clips are saved to recordings/ when ultrasonic energy is
detected and automatically identified by BatDetect2.
"""

import sys
import os
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

# ── BatDetect2 path ────────────────────────────────────────────────────────────
_BD2_PATH = str(Path(__file__).resolve().parent.parent / "bat_detector")
if _BD2_PATH not in sys.path:
    sys.path.insert(0, _BD2_PATH)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFileDialog, QSlider, QDoubleSpinBox,
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
RMS_THRESH      = 0.005          # minimum RMS to check for ultrasonics
ULTRA_RATIO     = 0.05           # fraction of energy that must be above ULTRA_MIN_HZ
ULTRA_MIN_HZ    = 20_000         # ultrasonic threshold

RECORDINGS_DIR  = Path(__file__).resolve().parent / "recordings"

# Latin → English common names for UK bat species returned by BatDetect2
_COMMON_NAMES: dict[str, str] = {
    "Barbastella barbastellus":    "Barbastelle",
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
    detection = pyqtSignal(str, str, float)   # timestamp, species, confidence
    status    = pyqtSignal(str)               # status bar message
    file_done = pyqtSignal()                  # file load finished → re-enable button
    clip_spec = pyqtSignal(object, str, object)  # spec array, timestamp, annotations

SIGNALS = _Signals()


# ── Queues ─────────────────────────────────────────────────────────────────────
_spec_q      : queue.Queue = queue.Queue(maxsize=1000)
_clip_audio_q: queue.Queue = queue.Queue(maxsize=10)
_hetero_q    : queue.Queue = queue.Queue(maxsize=200)   # raw chunks for heterodyne
_out_q       : queue.Queue = queue.Queue(maxsize=200)   # decimated output chunks


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
        self.setWindowTitle("Bat Sonar — Live Sonogram")
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

        vbox.addLayout(row1)

        # ── Row 2: sensitivity + species refs + filter ────────────────────────
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Trigger sensitivity:"))
        row2.addWidget(QLabel("Low"))
        self._sens_slider = QSlider(Qt.Horizontal)
        self._sens_slider.setFixedWidth(130)
        self._sens_slider.setRange(1, 10)
        self._sens_slider.setValue(self._slider_from_thresh(RMS_THRESH))
        self._sens_slider.setToolTip(
            "How easily a clip recording is triggered.\n"
            "Increase if bats are being missed; decrease if wind or\n"
            "background noise is causing too many false recordings."
        )
        self._sens_slider.valueChanged.connect(self._on_sensitivity_changed)
        row2.addWidget(self._sens_slider)
        row2.addWidget(QLabel("High"))

        self._sens_label = QLabel(f"Level {self._sens_slider.value()}/10")
        self._sens_label.setFixedWidth(68)
        row2.addWidget(self._sens_label)

        row2.addSpacing(16)

        self._refs_btn = QPushButton("Species refs")
        self._refs_btn.setCheckable(True)
        self._refs_btn.toggled.connect(self._toggle_refs)
        row2.addWidget(self._refs_btn)

        self._filter_btn = QPushButton("Filter < 40% confidence")
        self._filter_btn.setCheckable(True)
        self._filter_btn.toggled.connect(self._toggle_low_conf_filter)
        row2.addWidget(self._filter_btn)

        row2.addStretch()
        vbox.addLayout(row2)

        # ── Row 3: heterodyne audio output ────────────────────────────────────
        row3 = QHBoxLayout()

        self._audio_btn = QPushButton("🔇  Audio off")
        self._audio_btn.setCheckable(True)
        self._audio_btn.setFixedWidth(110)
        self._audio_btn.setToolTip(
            "Frequency division audio (÷8).\n"
            "All bat calls become audible simultaneously — no tuning needed.\n"
            "45 kHz → 5.6 kHz, 110 kHz → 13.75 kHz, etc."
        )
        self._audio_btn.toggled.connect(self._toggle_audio)
        row3.addWidget(self._audio_btn)

        row3.addSpacing(16)
        row3.addWidget(QLabel("Volume:"))

        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setFixedWidth(100)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.valueChanged.connect(self._on_volume_changed)
        row3.addWidget(self._vol_slider)

        self._vol_label = QLabel("80%")
        self._vol_label.setFixedWidth(36)
        row3.addWidget(self._vol_label)

        row3.addSpacing(24)

        self._notch_btn = QPushButton("Noise filter: off")
        self._notch_btn.setCheckable(True)
        self._notch_btn.setFixedWidth(120)
        self._notch_btn.setToolTip(
            "Narrow notch filter that removes a fixed-frequency interference tone\n"
            "(e.g. USB switching noise, ultrasonic sensors).\n"
            "Affects both the waterfall display and the audio output."
        )
        self._notch_btn.toggled.connect(self._toggle_notch)
        row3.addWidget(self._notch_btn)

        self._notch_spin = QDoubleSpinBox()
        self._notch_spin.setRange(FREQ_MIN_HZ / 1000, FREQ_MAX_HZ / 1000)
        self._notch_spin.setValue(self._notch_freq_hz / 1000)
        self._notch_spin.setSuffix(" kHz")
        self._notch_spin.setSingleStep(0.5)
        self._notch_spin.setDecimals(1)
        self._notch_spin.setFixedWidth(90)
        self._notch_spin.setToolTip("Centre frequency of the noise notch filter")
        self._notch_spin.valueChanged.connect(self._on_notch_freq_changed)
        row3.addWidget(self._notch_spin)

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

        self._clip_title = QLabel("<b>Call Detail</b> — no clip yet")
        clip_vbox.addWidget(self._clip_title)

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
        det_vbox.addWidget(QLabel("<b>Recent Detections</b>"))

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Time", "Species", "Confidence"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setDefaultSectionSize(120)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(False)
        det_vbox.addWidget(self._table)

        splitter.addWidget(det_widget)
        splitter.setSizes([320, 260, 220])

        vbox.addWidget(splitter)

        # Status bar
        self._sb = self.statusBar()
        self._sb.showMessage("Ready — select your pippyg device and click Start")
        SIGNALS.status.connect(self._sb.showMessage)

    def _populate_devices(self):
        self._device_combo.clear()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                ok = self._check_384k(i)
                badge = "✓" if ok else "✗"
                self._device_combo.addItem(f"[{i}] {badge}  {dev['name']}", i)

    def _refresh_devices(self):
        # Force PortAudio to rescan — without this it returns the cached list
        # from startup and newly plugged USB devices never appear.
        # Stop the output stream first so sd._terminate() doesn't silently
        # kill it, then restart it afterwards.
        if self._stream is None:
            try:
                if self._out_stream is not None:
                    self._out_stream.stop()
                    self._out_stream.close()
                    self._out_stream = None
                sd._terminate()
                sd._initialize()
                self._out_stream = sd.OutputStream(
                    samplerate=OUTPUT_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=_OUT_BLOCK,
                    callback=self._output_callback,
                )
                self._out_stream.start()
            except Exception:
                pass

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
        """Render the high-resolution spectrogram of the last saved clip.

        spec : (n_frames, N_CLIP_BINS) float32 — log-magnitude
        """
        if spec.shape[0] < 2:
            return

        # Display image — spec is (time, freq), col-major so x=time, y=freq
        clip_max = float(spec.max())
        self._clip_img.setImage(spec, autoLevels=False,
                                levels=(0.0, max(clip_max, 0.1)))

        # Fit the view to the image extent
        n_frames, n_bins = spec.shape
        ms_per_frame = (CLIP_HOP / SAMPLE_RATE) * 1000.0   # ≈ 0.333 ms per frame
        total_ms     = n_frames * ms_per_frame

        # Start zoomed to ~500 ms so individual call shapes are clearly visible;
        # user can click-drag left/right to scroll through the rest of the clip.
        view_frames = min(n_frames, int(500.0 / ms_per_frame))
        self._clip_plot.setXRange(0, view_frames, padding=0)
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

        self._clip_title.setText(f"<b>Call Detail</b> — {ts}")

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

    def _on_volume_changed(self, value: int):
        self._volume[0] = value / 100.0
        self._vol_label.setText(f"{value}%")

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

    # ── Workers ───────────────────────────────────────────────────────────────
    def _start_workers(self):
        threading.Thread(target=_save_and_analyse_worker, daemon=True).start()
        threading.Thread(
            target=_freqdiv_worker,
            args=(self._audio_enabled, self._volume),
            daemon=True,
        ).start()
        SIGNALS.detection.connect(self._on_detection)
        SIGNALS.clip_spec.connect(self._on_clip_spec)
        SIGNALS.file_done.connect(self._on_file_done)
        SIGNALS.file_done.connect(lambda: self._file_btn.setEnabled(True))

        # Output stream opens once at startup — outputs silence when queue empty
        try:
            self._out_stream = sd.OutputStream(
                samplerate=OUTPUT_RATE,
                channels=1,
                dtype="float32",
                blocksize=_OUT_BLOCK,
                callback=self._output_callback,
            )
            self._out_stream.start()
        except Exception as e:
            self._sb.showMessage(f"Warning: no audio output — {e}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
