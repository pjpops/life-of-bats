"""
bsg_bat.py — Thin wrapper around the BSG-BAT v0.21 ensemble classifier.

Loads all 6 models lazily on first use (once, in a background thread).
Call  BSG.classify(wav_path)  →  dict[species_name, max_prob_across_segments]

The 'Background' key is included so callers can detect likely non-bat clips.
"""

import sys
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_BSG_ROOT   = Path.home() / "Desktop" / "bsgbat"
_BSG_CODE   = _BSG_ROOT   / "code"
_BSG_MODELS = _BSG_ROOT   / "models"
_SPECIES_FILE = _BSG_MODELS / "species21bg"
_N_MODELS   = 6
_N_CLASSES  = 22

# ── Common-name map (BSG-BAT internal names → display names) ──────────────────
COMMON_NAMES: dict[str, str] = {
    "Barbastella_barbastellus":  "Barbastelle",
    "Eptesicus_nilssonii":       "Northern Bat",
    "Eptesicus_serotinus":       "Serotine",
    "Hypsugo_savii":             "Savi's Pip",
    "Miniopterus_schreibersii":  "Schreibers'",
    "Myotis_alcathoe":           "Alcathoe",
    "Myotis_crypticus":          "Cryptic Myotis",
    "Myotis_daubentonii":        "Daubenton's",
    "Nyctalus_leisleri":         "Leisler's",
    "Nyctalus_noctula":          "Noctule",
    "Pipistrellus_kuhlii":       "Kuhl's Pip",
    "Pipistrellus_nathusii":     "Nathusius' Pip",
    "Pipistrellus_pipistrellus": "Common Pip",
    "Pipistrellus_pygmaeus":     "Soprano Pip",
    "Plecotus_auritus":          "Brown Long-eared",
    "Plecotus_austriacus":       "Grey Long-eared",
    "Rhinolophus_euryale":       "Med. Horseshoe",
    "Rhinolophus_ferrumequinum": "Greater Horseshoe",
    "Rhinolophus_hipposideros":  "Lesser Horseshoe",
    "Tadarida_teniotis":         "Free-tailed",
    "Vespertilio_murinus":       "Parti-coloured",
    "Background":                "Background",
}


class BSGBat:
    """Lazy-loading BSG-BAT ensemble classifier.

    Usage:
        result = BSG.classify("/path/to/file.wav")
        # → {"Barbastella_barbastellus": 0.93, "Background": 0.12, ...}
    """

    def __init__(self):
        self._models   = None   # list of loaded Net instances
        self._species  = None   # list[str] matching column order
        self._classify = None   # bound function reference
        self._wav2spec = None   # bound function reference
        self._error    = None   # string if loading failed

    # ── Public ─────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True once models are loaded and ready."""
        return self._models is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self, status_cb=None):
        """Load all 6 models. Call from a background thread.
        Optional status_cb(msg: str) receives progress messages.
        """
        if self._models is not None:
            return   # already loaded

        def _msg(m):
            if status_cb:
                status_cb(m)

        try:
            import torch

            if str(_BSG_CODE) not in sys.path:
                sys.path.insert(0, str(_BSG_CODE))

            import supervised as _sup
            import data384   as _d384

            _msg("BSG-BAT: reading species list…")
            self._species = _d384.read_filelist(str(_SPECIES_FILE))

            device = torch.device("cpu")
            models = []
            for r in range(1, _N_MODELS + 1):
                _msg(f"BSG-BAT: loading model {r}/{_N_MODELS}…")
                m = _sup.Net(nclasses=_N_CLASSES)
                m.load_state_dict(
                    torch.load(
                        str(_BSG_MODELS / f"model_v0.21_r{r}.pt"),
                        map_location=device,
                        weights_only=True,
                    )
                )
                m.eval()
                models.append(m)

            self._models   = models
            self._classify = _sup.classify1_cpu
            self._wav2spec = _d384.wav2spectrograms
            _msg("BSG-BAT: ready ✓")

        except Exception as e:
            self._error = str(e)
            _msg(f"BSG-BAT load failed: {e}")

    def classify(self, wav_path: str) -> dict[str, float]:
        """Run all 6 models on wav_path and return {species: max_prob}.

        Probabilities are the maximum sigmoid value across all 0.5-second
        segments — i.e. 'was this species present at any point in the clip?'
        Returns an empty dict if models are not loaded or an error occurs.
        """
        if self._models is None:
            return {}
        try:
            dat = self._wav2spec(wav_path)                       # (n_segs, 512, 128)
            all_logits = np.array([
                self._classify(dat, m, _N_CLASSES)
                for m in self._models
            ])                                                   # (6, n_segs, 22)
            mean_logits = all_logits.mean(axis=0)                # (n_segs, 22)
            probs       = 1.0 / (1.0 + np.exp(-mean_logits))    # sigmoid → (n_segs, 22)
            max_probs   = probs.max(axis=0)                      # (22,) — peak per species
            return {sp: float(max_probs[i]) for i, sp in enumerate(self._species)}
        except Exception:
            return {}


# ── Module-level singleton ─────────────────────────────────────────────────────
BSG = BSGBat()
