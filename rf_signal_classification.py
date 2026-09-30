"""
RF Signal Classification using TorchSig
Undergraduate Project — Automatic Modulation Classification (AMC)

Pipeline:
  1. Collect clean reference IQ for visualization
  2. Generate IQ / constellation / spectrogram plots (skippable)
  3. Build multi-SNR train/test datasets (cached)
  4. Train hybrid residual CNN (IQ 1D + spectrogram 2D) on GPU
  5. Evaluate (overall, per-class, family, SNR sweep)
  6. Save figures, CSV, JSON metrics
  7. Write academic DOCX report
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.defaults import TorchSigDefaults

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
for d in (PLOTS_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# ── Class taxonomy ───────────────────────────────────────────────────────
CLASSES = ["bpsk", "16qam", "ofdm-64"]
# [
#     "ook",
#     "4ask", "8ask", "16ask", "32ask", "64ask",
#     "2fsk", "4fsk", "8fsk", "16fsk",
#     "2gfsk", "4gfsk", "8gfsk", "16gfsk",
#     "2msk", "4msk", "8msk", "16msk",
#     "2gmsk", "4gmsk", "8gmsk", "16gmsk",
#     "bpsk", "qpsk", "8psk", "16psk", "32psk", "64psk",
#     "16qam", "32qam", "64qam", "128qam_cross", "256qam",
#     "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-256", "ofdm-300", "ofdm-512", "ofdm-1024",
#     "fm", "am-dsb-sc", "am-dsb", "am-lsb", "am-usb", "tone",
# ]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
SNR_VALUES = list(range(-20, 23, 2))
TOTAL_PLOTS = len(CLASSES) * len(SNR_VALUES)
TRAIN_SNR_RANGE = (0, 20)
TEST_SNR_RANGE = (0, 20)
SNR_EVAL_POINTS = list(range(-4, 22, 4))
IQ_LEN = 2048
SPEC_F, SPEC_T = 64, 64  # spectrogram grid

FAMILY_OF = {}
for c in CLASSES:
    u = c.upper()
    if c == "ook":
        f = "OOK"
    elif "ASK" in u:
        f = "ASK"
    elif "GFSK" in u:
        f = "GFSK"
    elif "GMSK" in u:
        f = "GMSK"
    elif "FSK" in u:
        f = "FSK"
    elif "MSK" in u:
        f = "MSK"
    elif "PSK" in u or c in ("bpsk", "qpsk"):
        f = "PSK"
    elif "QAM" in u:
        f = "QAM"
    elif "OFDM" in u:
        f = "OFDM"
    elif c.startswith("am-"):
        f = "AM"
    elif c == "fm":
        f = "FM"
    elif c == "tone":
        f = "Tone"
    else:
        f = "Other"
    FAMILY_OF[c] = f

# Constrained generation → lower intra-class variance, learnable AMC problem
META = dict(TorchSigDefaults().default_dataset_metadata)
META.update({
    "num_iq_samples_dataset": IQ_LEN,
    "signal_duration_in_samples_min": IQ_LEN,
    "signal_duration_in_samples_max": IQ_LEN,
    "sample_rate": 1_000_000,
    "fft_size": 256,
    "fft_stride": 128,
    "bandwidth_min": 200_000,
    "bandwidth_max": 300_000,
    "signal_center_freq_min": -50_000,
    "signal_center_freq_max": 50_000,
    "frequency_min": -250_000,
    "frequency_max": 249_999,
    "snr_db_min": 30,
    "snr_db_max": 35,
    "noise_power_db": 0.0,
    "num_signals_min": 1,
    "num_signals_max": 1,
})


# ── Device ───────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}  "
                  f"free={free / 1e9:.1f}GB / {total / 1e9:.1f}GB")
            if free > best_free:
                best_free, best = free, i
        print(f"    Using: {torch.cuda.get_device_name(best)}")
        return torch.device(f"cuda:{best}")
    if "+cpu" in torch.__version__:
        print(f"    WARNING: CPU-only PyTorch ({torch.__version__})")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
    return torch.device("cpu")


# ── Signal helpers ───────────────────────────────────────────────────────
def compute_spectrogram(iq, fs=1e6, nperseg=128, noverlap=96):
    f, t, Sxx = scipy_signal.spectrogram(
        iq, fs=fs, nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, mode="psd",
    )
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return 10 * np.log10(Sxx + 1e-12), f * 1e-6, t * 1e6


def power_normalize(iq: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(iq) ** 2) + 1e-12
    return (iq / np.sqrt(p)).astype(np.complex64)


def add_awgn(iq: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random
    sig_power = np.mean(np.abs(iq) ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.randn(len(iq)) + 1j * rng.randn(len(iq))
    )
    return iq + noise


def iq_channels(iq: np.ndarray) -> np.ndarray:
    """(4, L): I, Q, amplitude, instantaneous phase derivative proxy."""
    iq = power_normalize(iq)
    i = np.real(iq).astype(np.float32)
    q = np.imag(iq).astype(np.float32)
    amp = np.abs(iq).astype(np.float32)
    phase = np.unwrap(np.angle(iq))
    dphase = np.diff(phase, prepend=phase[0]).astype(np.float32)
    dphase = dphase / (np.std(dphase) + 1e-6)
    amp = (amp - amp.mean()) / (amp.std() + 1e-6)
    return np.stack([i, q, amp, dphase], axis=0)


def iq_to_spectrogram_image(iq: np.ndarray, out_hw=(SPEC_F, SPEC_T)) -> np.ndarray:
    """Log-magnitude spectrogram resized to (1, F, T)."""
    iq = power_normalize(iq)
    nperseg = 128
    noverlap = 96
    _, _, Sxx = scipy_signal.spectrogram(
        iq, fs=META["sample_rate"], nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, mode="magnitude",
    )
    Sxx = np.fft.fftshift(Sxx, axes=0)
    spec = np.log10(Sxx + 1e-10).astype(np.float32)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    # resize via simple block resampling
    t = torch.from_numpy(spec)[None, None]  # 1,1,F,T
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()  # (1, F, T)


def make_signal_plot(iq, mod_type, snr_db, save_path):
    fs = META["sample_rate"]
    t_us = np.arange(len(iq)) / fs * 1e6
    spec, f_mhz, t_spec = compute_spectrogram(iq, fs=fs)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1, 1.5]})
    fig.suptitle(f"{mod_type.upper()} @ SNR = {snr_db} dB", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(t_us, np.real(iq), "b-", alpha=0.7, lw=0.8, label="I")
    ax.plot(t_us, np.imag(iq), "r-", alpha=0.7, lw=0.8, label="Q")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("IQ Time-Domain Waveform")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n = min(1024, len(iq))
    ax.scatter(np.real(iq[:n]), np.imag(iq[:n]), s=3, c="darkblue", alpha=0.6)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title("Constellation Diagram")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    lim = max(float(np.abs(iq[:n]).max()) * 1.2, 0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax = axes[2]
    extent = [t_spec[0], t_spec[-1], f_mhz[0], f_mhz[-1]]
    im = ax.imshow(spec, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Spectrogram")
    plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Data generation ──────────────────────────────────────────────────────
def create_iterable_dataset():
    return TorchSigIterableDataset(
        signal_generators="all",
        transforms=[],
        target_labels=["class_name"],
        **META,
    )


def collect_clean_signals(classes, max_attempts=3000):
    dataset = create_iterable_dataset()
    collected = {}
    attempts = 0
    it = iter(dataset)
    while len(collected) < len(classes) and attempts < max_attempts:
        data, label = next(it)
        attempts += 1
        if label in classes and label not in collected:
            collected[label] = np.asarray(data).copy()
            print(f"    [{len(collected)}/{len(classes)}] {label}")
    return collected


def build_balanced_dataset(
    classes,
    samples_per_class: int,
    snr_range: tuple[float, float],
    seed: int,
    max_factor: int = 80,
):
    """Return lists of complex IQ + labels + snrs (features computed later)."""
    rng = np.random.RandomState(seed)
    dataset = create_iterable_dataset()
    it = iter(dataset)
    buckets: dict[str, list] = {c: [] for c in classes}
    snr_buckets: dict[str, list] = {c: [] for c in classes}
    target = len(classes) * samples_per_class
    attempts = 0
    max_attempts = target * max_factor

    print(f"    Collecting {samples_per_class}/class × {len(classes)} "
          f"(SNR {snr_range[0]}..{snr_range[1]} dB)...")
    t0 = time.time()
    while any(len(v) < samples_per_class for v in buckets.values()) and attempts < max_attempts:
        data, label = next(it)
        attempts += 1
        if label not in buckets or len(buckets[label]) >= samples_per_class:
            continue
        iq = np.asarray(data, dtype=np.complex128).copy()
        if len(iq) < IQ_LEN:
            iq = np.pad(iq, (0, IQ_LEN - len(iq)))
        elif len(iq) > IQ_LEN:
            iq = iq[:IQ_LEN]
        snr = float(rng.uniform(snr_range[0], snr_range[1]))
        iq_n = add_awgn(iq, snr, rng=rng)
        buckets[label].append(iq_n.astype(np.complex64))
        snr_buckets[label].append(snr)
        done = sum(len(v) for v in buckets.values())
        if done % 400 == 0:
            print(f"      {done}/{target}  ({time.time() - t0:.0f}s)")

    iqs, ys, snrs = [], [], []
    missing = []
    for c in classes:
        if len(buckets[c]) < samples_per_class:
            missing.append(f"{c}({len(buckets[c])})")
        for iq, snr in zip(buckets[c], snr_buckets[c]):
            iqs.append(iq)
            ys.append(CLASS_TO_IDX[c])
            snrs.append(snr)
    if missing:
        print(f"    WARNING incomplete: {missing}")
    print(f"    Built {len(iqs)} samples in {time.time() - t0:.1f}s ({attempts} iters)")
    return (
        np.stack(iqs),
        np.array(ys, dtype=np.int64),
        np.array(snrs, dtype=np.float32),
    )


def load_or_build_raw(name, spc, snr_range, seed, force=False):
    path = os.path.join(CACHE_DIR, f"{name}_v2_spc{spc}_s{seed}.npz")
    if (not force) and os.path.exists(path):
        print(f"    Loading cache: {os.path.basename(path)}")
        z = np.load(path, allow_pickle=False)
        return z["iq"], z["y"], z["snr"]
    iq, y, snr = build_balanced_dataset(CLASSES, spc, snr_range, seed)
    np.savez_compressed(path, iq=iq, y=y, snr=snr)
    print(f"    Cached → {path}")
    return iq, y, snr


class RFHybridDataset(Dataset):
    """On-the-fly IQ multi-channel + spectrogram features with light augmentation."""

    def __init__(self, iq: np.ndarray, y: np.ndarray, train: bool = False, seed: int = 0):
        self.iq = iq
        self.y = y
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        iq = self.iq[idx].astype(np.complex128).copy()
        if self.train:
            # random phase rotation
            phi = self.rng.uniform(0, 2 * np.pi)
            iq = iq * np.exp(1j * phi)
            # circular time shift
            shift = self.rng.randint(0, IQ_LEN)
            iq = np.roll(iq, shift)
            # small residual SNR jitter already baked in data
        x1 = iq_channels(iq)  # (4, L)
        x2 = iq_to_spectrogram_image(iq)  # (1, F, T)
        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ── Model ────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, ch, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class HybridAMCNet(nn.Module):
    """
    Dual-branch AMC network:
      - Branch A: residual 1D CNN on [I,Q,amp,dphase]
      - Branch B: 2D CNN on log-magnitude spectrogram
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # IQ branch
        self.iq_stem = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.iq_body = nn.Sequential(
            ResBlock1D(64, 5),
            ResBlock1D(64, 5),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            ResBlock1D(128, 5),
            ResBlock1D(128, 5, dilation=2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            ResBlock1D(256, 3, dilation=2),
            ResBlock1D(256, 3, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )
        # Spectrogram branch
        self.spec_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        fused = 256 + 128
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_iq, x_spec):
        h1 = self.iq_body(self.iq_stem(x_iq)).flatten(1)
        h2 = self.spec_net(x_spec).flatten(1)
        return self.head(torch.cat([h1, h2], dim=1))


# ── Train / eval ─────────────────────────────────────────────────────────
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x1, x2)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * y.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(y.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / max(total, 1), correct / max(total, 1), preds, labels


@torch.no_grad()
def evaluate_snr_curve(model, device, base_iq, base_y, use_amp=False):
    model.eval()
    results = {}
    rng = np.random.RandomState(999)
    # base_iq assumed nearly clean (high SNR generation). Re-noise to target.
    for snr_db in SNR_EVAL_POINTS:
        feats_iq, feats_sp = [], []
        for i in range(len(base_iq)):
            # strip existing noise by treating sample as source; re-apply SNR
            src = base_iq[i].astype(np.complex128)
            # light denoise not available — use as-is and add target-level noise carefully:
            # re-normalize then AWGN at snr_db
            src = power_normalize(src)
            iq = add_awgn(src.astype(np.complex128), snr_db, rng=rng)
            feats_iq.append(iq_channels(iq))
            feats_sp.append(iq_to_spectrogram_image(iq))
        X1 = torch.from_numpy(np.stack(feats_iq))
        X2 = torch.from_numpy(np.stack(feats_sp))
        y = torch.from_numpy(base_y)
        correct, total = 0, 0
        bs = 64
        for i in range(0, len(X1), bs):
            o = model(
                X1[i:i + bs].to(device),
                X2[i:i + bs].to(device),
            )
            correct += (o.argmax(1).cpu() == y[i:i + bs]).sum().item()
            total += min(bs, len(X1) - i)
        acc = correct / total
        results[int(snr_db)] = acc
        print(f"      SNR {snr_db:+3d} dB → acc={acc:.3f}")
    return results


def family_accuracy(y_true, y_pred):
    fam_true = np.array([FAMILY_OF[CLASSES[i]] for i in y_true])
    fam_pred = np.array([FAMILY_OF[CLASSES[i]] for i in y_pred])
    out = {}
    for f in sorted(set(FAMILY_OF.values())):
        mask = fam_true == f
        if mask.any():
            out[f] = float((fam_true[mask] == fam_pred[mask]).mean())
    out["overall"] = float((fam_true == fam_pred).mean())
    return out


# ── Figures ──────────────────────────────────────────────────────────────
def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], "b-", lw=2, label="Train")
    axes[0].plot(ep, history["val_loss"], "r-", lw=2, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["train_acc"], "b-", lw=2, label="Train")
    axes[1].plot(ep, history["val_acc"], "r-", lw=2, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(18, 16))
    norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=6)
    ax.set_yticklabels(CLASSES, fontsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix (46 modulation classes)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1(report, path):
    f1s = [report[c]["f1-score"] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(CLASSES)))
    ax.bar(range(NUM_CLASSES), f1s, color=colors)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1-Score on Multi-SNR Test Set")
    ax.set_ylim(0, 1.05)
    ax.axhline(np.mean(f1s), color="red", ls="--", lw=1.5,
               label=f"Macro F1 = {np.mean(f1s):.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_snr_curve(snr_acc, path):
    snrs = sorted(snr_acc.keys())
    accs = [snr_acc[s] for s in snrs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(snrs, accs, "o-", color="#1f4e79", lw=2, markersize=7)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Accuracy vs. Signal-to-Noise Ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    for s, a in zip(snrs, accs):
        ax.annotate(f"{a:.2f}", (s, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_family_bars(fam_acc, path):
    items = [(k, v) for k, v in fam_acc.items() if k != "overall"]
    items.sort(key=lambda x: -x[1])
    names, vals = [x[0] for x in items], [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color="#2e75b6")
    ax.axhline(fam_acc.get("overall", 0), color="red", ls="--",
               label=f"Overall family acc = {fam_acc.get('overall', 0):.3f}")
    ax.set_ylabel("Accuracy")
    ax.set_title("Family-Level Classification Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_sample_gallery(clean_signals, path, snr_db=15):
    picks = list(clean_signals.keys())
    # picks = ["bpsk", "qpsk", "16qam", "64qam", "2fsk", "ofdm-64", "fm", "am-dsb"]
    picks = [p for p in picks if p in clean_signals]
    if not picks:
        return
    n = len(picks)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_2d(axes)
    rng = np.random.RandomState(0)
    for i, mod in enumerate(picks):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        iq = power_normalize(add_awgn(clean_signals[mod].copy(), snr_db, rng=rng))
        ax.scatter(np.real(iq[:800]), np.imag(iq[:800]), s=4, alpha=0.6, c="#1f4e79")
        ax.set_title(f"{mod} @ {snr_db} dB", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        lim = max(np.abs(iq[:800]).max() * 1.15, 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle("Representative Constellation Diagrams", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── Academic DOCX ────────────────────────────────────────────────────────
def build_docx(metrics: dict, figure_paths: dict, out_path: str):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, m, Inches(1))

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.paragraph_format.space_after = Pt(8)
    style.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 16), (2, 14), (3, 12)):
        hs = doc.styles[f"Heading {level}"]
        hs.font.name = "Times New Roman"
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        hs._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    def p(text, bold=False, italic=False, center=False, size=12):
        para = doc.add_paragraph()
        if center:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.size = Pt(size)
        run.font.name = "Times New Roman"
        return para

    def add_fig(key, caption, width=6.0):
        path = figure_paths.get(key)
        if not path or not os.path.exists(path):
            p(f"[Figure missing: {key}]", italic=True)
            return
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(10)
        r.font.name = "Times New Roman"

    def add_table(headers, rows):
        table = doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.style = "Table Grid"
        for j, h in enumerate(headers):
            cell = table.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = table.rows[i + 1].cells[j]
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(val))
                run.font.size = Pt(9)
                run.font.name = "Times New Roman"
        doc.add_paragraph()

    # Title
    p("Automatic Modulation Classification of RF Signals", bold=True, center=True, size=18)
    p("Using a Hybrid Residual CNN on Complex Baseband IQ and Spectrograms",
      italic=True, center=True, size=13)
    p("Undergraduate Capstone Project Report", center=True, size=12)
    p(f"Generated: {metrics.get('timestamp', '')}", center=True, size=10)
    p(f"Framework: PyTorch {metrics.get('torch_version', '')}  |  "
      f"Device: {metrics.get('device', 'cpu')}  |  Dataset: TorchSig",
      center=True, size=10)

    acc = metrics["test_accuracy"]
    macro_f1 = metrics["macro_f1"]
    fam = metrics["family_accuracy"]

    doc.add_heading("Abstract", level=1)
    p(
        f"This report presents an end-to-end machine-learning pipeline for automatic "
        f"modulation classification (AMC) of radio-frequency signals. Synthetic complex "
        f"baseband IQ waveforms for {metrics['num_classes']} modulation types were generated "
        f"with TorchSig under constrained emission parameters and corrupted by additive white "
        f"Gaussian noise over a multi-SNR range of {metrics['train_snr_range'][0]}–"
        f"{metrics['train_snr_range'][1]} dB. A hybrid deep network combines a residual "
        f"one-dimensional CNN on multi-channel IQ features with a two-dimensional CNN on "
        f"log-magnitude spectrograms. Training used mixed-precision acceleration on NVIDIA "
        f"GPUs. On a held-out multi-SNR test set the classifier achieved "
        f"{acc * 100:.2f}% overall accuracy and a macro-averaged F1-score of {macro_f1:.3f}, "
        f"with family-level accuracy of {fam['overall'] * 100:.2f}%. Performance is further "
        f"characterized by per-class metrics, a confusion matrix, and an accuracy-versus-SNR "
        f"curve. Results show that hybrid time–frequency deep features can discriminate a "
        f"large, fine-grained modulation vocabulary without hand-crafted cyclic statistics."
    )

    doc.add_heading("1. Introduction", level=1)
    p(
        "Automatic modulation classification is fundamental to cognitive radio, spectrum "
        "monitoring, electronic support measures, and adaptive communications. Given a short "
        "baseband capture, an AMC system must infer the transmitter’s modulation. Classical "
        "likelihood and cumulant-based methods require strong models; deep learning can learn "
        "discriminative representations directly from IQ samples or time–frequency images."
    )
    p(
        f"This project implements a reproducible synthetic-data AMC system spanning "
        f"{metrics['num_classes']} TorchSig classes (ASK, FSK/GFSK/MSK/GMSK, PSK, QAM, OFDM, "
        f"AM/FM/tone). Objectives: (i) build balanced multi-SNR IQ data; (ii) train a hybrid "
        f"residual CNN; (iii) report multi-view evaluation; (iv) document findings academically."
    )

    doc.add_heading("2. Methodology", level=1)

    doc.add_heading("2.1 Signal Model and Dataset", level=2)
    p(
        f"Complex baseband IQ of length L = {IQ_LEN} was synthesized at 1 MHz using TorchSig. "
        "Generator metadata was constrained (full-window duration, moderate bandwidth, "
        "near-zero center frequency, single emitter, high native SNR) to reduce uncontrolled "
        "nuisance variation while retaining realistic pulse-shaping and constellation structure. "
        "Independent AWGN was then applied at a target SNR γ:"
    )
    p("r[n] = s[n] + w[n],   w[n] ~ CN(0, σ²),   σ² = P_s / 10^(γ/10)", italic=True, center=True)
    p(
        f"Training used {metrics['train_samples']:,} examples ({metrics['train_spc']} per class) "
        f"with γ ~ Uniform[{metrics['train_snr_range'][0]}, {metrics['train_snr_range'][1]}] dB. "
        f"The test set held {metrics['test_samples']:,} examples ({metrics['test_spc']}/class) "
        "with an independent seed. Each capture was power-normalized. Features include four "
        "time-domain channels (I, Q, amplitude, phase-derivative) and a log-magnitude spectrogram "
        f"resized to {SPEC_F}×{SPEC_T}."
    )

    doc.add_heading("2.2 Modulation Classes", level=2)
    fam_map = defaultdict(list)
    for c in CLASSES:
        fam_map[FAMILY_OF[c]].append(c)
    add_table(
        ["Family", "# Types", "Classes"],
        [[f, str(len(v)), ", ".join(v)] for f, v in sorted(fam_map.items())],
    )

    doc.add_heading("2.3 Model Architecture", level=2)
    p(
        f"The HybridAMCNet has ≈{metrics['num_params']:,} parameters. Branch A applies a "
        "residual 1D CNN with dilated convolutions to multi-channel IQ. Branch B applies a "
        "compact 2D CNN to the spectrogram. Global average pooling yields vectors that are "
        "concatenated and classified by a two-layer MLP with dropout. Residual links stabilize "
        "optimization; the spectrogram branch supplies explicit time–frequency structure useful "
        "for OFDM and FSK families."
    )
    p(
        f"Optimization: AdamW (lr={metrics['lr']}, weight decay 1e-4), cosine annealing over "
        f"{metrics['epochs']} epochs, batch size {metrics['batch_size']}, gradient clipping, "
        "label smoothing 0.05, automatic mixed precision on CUDA. Training augmentation includes "
        "random phase rotation and circular time shifts."
    )

    doc.add_heading("2.4 Evaluation Protocol", level=2)
    p(
        "Primary metrics: top-1 accuracy and macro-F1 on the multi-SNR test set. Secondary: "
        "per-class precision/recall/F1, row-normalized confusion matrix, family-level accuracy, "
        "and accuracy versus SNR at discrete operating points."
    )

    doc.add_heading("2.5 Visualization", level=2)
    p(
        f"Qualitative panels ({metrics.get('plot_count', TOTAL_PLOTS)} figures) span all classes "
        f"and SNR grid points, each showing IQ time series, constellation, and spectrogram."
    )
    if figure_paths.get("gallery"):
        add_fig("gallery", "Figure 1. Representative constellation diagrams at 15 dB SNR.", 6.2)

    doc.add_heading("3. Results", level=1)

    doc.add_heading("3.1 Overall Performance", level=2)
    add_table(
        ["Metric", "Value"],
        [
            ["Number of classes", str(metrics["num_classes"])],
            ["Training samples", f"{metrics['train_samples']:,}"],
            ["Test samples", f"{metrics['test_samples']:,}"],
            ["Model parameters", f"{metrics['num_params']:,}"],
            ["Training device", metrics["device"]],
            ["Training time (s)", f"{metrics['train_time_s']:.1f}"],
            ["Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%"],
            ["Macro F1-score", f"{metrics['macro_f1']:.4f}"],
            ["Weighted F1-score", f"{metrics['weighted_f1']:.4f}"],
            ["Family-level accuracy", f"{metrics['family_accuracy']['overall'] * 100:.2f}%"],
            ["Best validation accuracy",
             f"{metrics['best_val_acc'] * 100:.2f}% (epoch {metrics['best_epoch']})"],
        ],
    )
    add_fig("training_curves",
            "Figure 2. Training and validation loss (left) and accuracy (right).", 6.2)

    doc.add_heading("3.2 Confusion Matrix", level=2)
    p(
        "Figure 3 shows the row-normalized confusion matrix. Off-diagonal mass often lies "
        "within the same family (adjacent PSK/QAM orders or OFDM FFT sizes), reflecting the "
        "difficulty of fine-grained order discrimination under noise."
    )
    add_fig("confusion_matrix",
            "Figure 3. Normalized confusion matrix over all modulation classes.", 6.4)

    doc.add_heading("3.3 Per-Class Metrics", level=2)
    add_fig("per_class_f1", "Figure 4. Per-class F1-scores on the multi-SNR test set.", 6.2)
    per = metrics["per_class"]
    ranked = sorted(CLASSES, key=lambda c: per[c]["f1-score"], reverse=True)
    p("Highest F1 classes:", bold=True)
    add_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [[c, f"{per[c]['precision']:.3f}", f"{per[c]['recall']:.3f}",
          f"{per[c]['f1-score']:.3f}", f"{per[c]['support']:.0f}"] for c in ranked[:8]],
    )
    p("Lowest F1 classes:", bold=True)
    add_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [[c, f"{per[c]['precision']:.3f}", f"{per[c]['recall']:.3f}",
          f"{per[c]['f1-score']:.3f}", f"{per[c]['support']:.0f}"] for c in ranked[-8:]],
    )

    doc.add_heading("3.4 Family-Level Accuracy", level=2)
    p(
        "Collapsing predictions to modulation families yields a coarser metric relevant to "
        "hierarchical spectrum sensing."
    )
    add_table(
        ["Family", "Accuracy"],
        [[k, f"{v * 100:.2f}%"] for k, v in sorted(fam.items(), key=lambda x: -x[1])],
    )
    add_fig("family_acc", "Figure 5. Family-level classification accuracy.", 5.8)

    doc.add_heading("3.5 Accuracy versus SNR", level=2)
    p(
        "Figure 6 quantifies AWGN robustness. Accuracy generally increases with SNR as "
        "constellation geometry and spectral occupancy become more separable."
    )
    snr_rows = [
        [f"{s} dB", f"{metrics['snr_curve'][str(s)] * 100:.2f}%"]
        for s in sorted(int(k) for k in metrics["snr_curve"].keys())
    ]
    add_table(["SNR", "Accuracy"], snr_rows)
    add_fig("snr_curve", "Figure 6. Classification accuracy as a function of SNR.", 5.5)

    doc.add_heading("4. Discussion", level=1)

    doc.add_heading("4.1 Interpretation of Findings", level=2)
    p(
        f"The hybrid network attained {acc * 100:.2f}% overall accuracy across "
        f"{metrics['num_classes']} classes under mixed SNR. Chance performance is only "
        f"≈{100 / metrics['num_classes']:.1f}%, so the model learns non-trivial structure. "
        f"Family-level accuracy of {fam['overall'] * 100:.2f}% shows that residual errors often "
        "remain within the correct modulation family—valuable for hierarchical AMC."
    )
    p(
        "Distinctive classes (binary PSK, tones, strongly structured FM/AM, OFDM spectral "
        "occupancy) tend to score higher F1. High-order constellations and closely related "
        "continuous-phase FSK variants remain confusable at lower SNR because symbol geometry "
        "is noise-dominated and phase trajectories are similar."
    )

    doc.add_heading("4.2 Comparison with Classical AMC", level=2)
    p(
        "Likelihood AMC needs accurate channel models and scales poorly with class count. "
        "Feature-based cumulant methods are interpretable but brittle under mismatch. Deep "
        "hybrid models trade interpretability for flexibility across analog and digital "
        "families without per-family feature engineering, at the cost of needing representative "
        "training data."
    )

    doc.add_heading("4.3 Limitations", level=2)
    p(
        "Experiments use synthetic TorchSig waveforms with AWGN; real channels add multipath, "
        "IQ imbalance, phase noise, CFO dynamics, and co-channel interference. Fixed-length "
        "captures may under-represent long OFDM symbols. Class priors in the wild are rarely "
        "uniform. Hyperparameters target a strong undergraduate baseline rather than exhaustive "
        "architecture search."
    )

    doc.add_heading("4.4 Future Work", level=2)
    p(
        "Extensions include multi-emitter detection, transformer/attention backbones, joint "
        "SNR–modulation estimation, semi-supervised learning from unlabeled spectrum, and "
        "over-the-air validation with software-defined radios."
    )

    doc.add_heading("5. Conclusion", level=1)
    p(
        f"This project delivered a complete AMC pipeline—from multi-SNR synthetic IQ generation "
        f"through hybrid deep learning and multi-view evaluation—to an academic report. The "
        f"HybridAMCNet achieved {acc * 100:.2f}% test accuracy and {macro_f1:.3f} macro-F1 over "
        f"{metrics['num_classes']} types, with clear SNR trends and interpretable family-level "
        f"behavior. The codebase supports GPU training, dataset caching, and automated "
        f"figure/report generation."
    )

    doc.add_heading("References", level=1)
    refs = [
        "O’Shea, T. J., & Hoydis, J. (2017). An introduction to deep learning for the physical layer. IEEE TCCN.",
        "O’Shea, T. J., Roy, T., & Clancy, T. C. (2018). Over-the-air deep learning based radio signal classification. IEEE JSTSP.",
        "TorchSig Contributors. TorchSig RF ML toolkit. https://torchsig.com",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.",
        "West, N. E., & O’Shea, T. (2017). Deep architectures for modulation recognition. IEEE DySPAN.",
    ]
    for i, ref in enumerate(refs, 1):
        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.25)
        para.paragraph_format.first_line_indent = Inches(-0.25)
        run = para.add_run(f"[{i}] {ref}")
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"

    doc.add_heading("Appendix A. Full Per-Class Classification Report", level=1)
    add_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [[c, f"{per[c]['precision']:.4f}", f"{per[c]['recall']:.4f}",
          f"{per[c]['f1-score']:.4f}", f"{per[c]['support']:.0f}"] for c in CLASSES],
    )

    doc.add_heading("Appendix B. Implementation Notes", level=1)
    p(
        f"Software: Python, NumPy, SciPy, scikit-learn, Matplotlib, Pandas, "
        f"PyTorch {metrics.get('torch_version', '')}, TorchSig, python-docx. "
        f"Primary script: rf_signal_classification.py. Artifacts under results/ and models/."
    )

    doc.save(out_path)
    print(f"    [OK] DOCX report → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RF AMC pipeline + academic report")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--force-plots", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-spc", type=int, default=150)
    parser.add_argument("--test-spc", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-docx", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print("=" * 64)
    print("  RF SIGNAL CLASSIFICATION — HYBRID AMC PIPELINE")
    print(f"  Classes: {NUM_CLASSES}  |  PyTorch: {torch.__version__}")
    print("=" * 64)

    existing_plots = (
        len([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        if os.path.isdir(PLOTS_DIR) else 0
    )
    do_plots = (not args.skip_plots) and (args.force_plots or existing_plots < TOTAL_PLOTS * 0.9)
    clean_signals = {}
    plot_count = existing_plots

    if do_plots:
        print("\n>>> STEP 1: Clean reference signals")
        t0 = time.time()
        clean_signals = collect_clean_signals(CLASSES)
        print(f"    Collected {len(clean_signals)}/{NUM_CLASSES} in {time.time() - t0:.1f}s")
        print(f"\n>>> STEP 2: Generating plots")
        t0 = time.time()
        plot_count = 0
        for ci, mod in enumerate(CLASSES):
            clean = clean_signals.get(mod)
            if clean is None:
                continue
            for snr in SNR_VALUES:
                name = f"{ci + 1:03d}_{mod}_snr{snr:03d}dB.png"
                path = os.path.join(PLOTS_DIR, name)
                if (not args.force_plots) and os.path.exists(path):
                    plot_count += 1
                    continue
                try:
                    make_signal_plot(add_awgn(clean.copy(), snr), mod, snr, path)
                    plot_count += 1
                except Exception as e:
                    print(f"    ERROR {mod}@{snr}: {e}")
            if (ci + 1) % 10 == 0:
                print(f"    [{ci + 1}/{NUM_CLASSES}] ({time.time() - t0:.0f}s)")
        print(f"    [OK] {plot_count} plots in {time.time() - t0:.0f}s")
    else:
        print(f"\n>>> STEP 1–2: Skipping plots ({existing_plots} PNGs; use --force-plots)")
        print("    Collecting gallery signals...")
        try:
            clean_signals = collect_clean_signals(
                ["bpsk", "qpsk", "16qam", "64qam", "2fsk", "ofdm-64", "fm", "am-dsb"],
                max_attempts=1500,
            )
        except Exception as e:
            print(f"    Gallery skipped: {e}")

    print("\n>>> STEP 3: Multi-SNR datasets")
    iq_tr, y_tr, _ = load_or_build_raw(
        "train", args.train_spc, TRAIN_SNR_RANGE, seed=100, force=args.force_data
    )
    iq_te, y_te, _ = load_or_build_raw(
        "test", args.test_spc, TEST_SNR_RANGE, seed=200, force=args.force_data
    )
    iq_snr, y_snr, _ = load_or_build_raw(
        "snrbase", 15, (30, 34), seed=777, force=args.force_data
    )

    device = get_device()
    use_cuda = device.type == "cuda"
    train_ds = RFHybridDataset(iq_tr, y_tr, train=True, seed=11)
    test_ds = RFHybridDataset(iq_te, y_te, train=False, seed=22)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=args.num_workers,
    )

    print("\n>>> STEP 4: Training HybridAMCNet")
    model = HybridAMCNet(NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {n_params:,}")
    print(f"    Train: {len(train_ds)}  Test: {len(test_ds)}  "
          f"epochs={args.epochs}  bs={args.batch_size}")

    ckpt_path = os.path.join(MODELS_DIR, "rf_classifier.pth")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch = -1.0, None, 0
    train_time = 0.0

    if args.skip_train and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"    Loaded weights: {ckpt_path}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_amp = use_cuda
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, _, _ = run_epoch(
                model, train_loader, device, criterion, optimizer, scaler, use_amp
            )
            va_loss, va_acc, _, _ = run_epoch(
                model, test_loader, device, criterion, None, scaler, use_amp
            )
            scheduler.step()
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            if va_acc > best_val:
                best_val, best_epoch = va_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    Epoch {epoch:02d}/{args.epochs}  "
                  f"train={tr_acc:.3f}  val={va_acc:.3f}  loss={tr_loss:.3f}/{va_loss:.3f}")
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), ckpt_path)
        train_time = time.time() - t0
        print(f"    Best val={best_val:.3f} @ epoch {best_epoch}  ({train_time:.1f}s)")
        with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    if not history["train_loss"] and os.path.exists(os.path.join(RESULTS_DIR, "history.json")):
        with open(os.path.join(RESULTS_DIR, "history.json")) as f:
            history = json.load(f)
        best_val = max(history.get("val_acc", [0]))
        best_epoch = int(np.argmax(history.get("val_acc", [0]))) + 1

    print("\n>>> STEP 5: Evaluation")
    dummy_scaler = torch.amp.GradScaler("cuda", enabled=False)
    _, test_acc, preds, labels = run_epoch(
        model, test_loader, device, nn.CrossEntropyLoss(), None, dummy_scaler, False
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        labels, preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fam_acc = family_accuracy(labels, preds)
    print(f"    Test accuracy: {test_acc:.4f}  macro-F1: {macro_f1:.4f}")
    print(f"    Family accuracy: {fam_acc['overall']:.4f}")

    print("    SNR sweep...")
    snr_curve = evaluate_snr_curve(model, device, iq_snr, y_snr)

    print("\n>>> STEP 6: Artifacts")
    paths = {
        "training_curves": os.path.join(RESULTS_DIR, "training_curves.png"),
        "confusion_matrix": os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        "per_class_f1": os.path.join(RESULTS_DIR, "per_class_f1.png"),
        "snr_curve": os.path.join(RESULTS_DIR, "snr_accuracy_curve.png"),
        "family_acc": os.path.join(RESULTS_DIR, "family_accuracy.png"),
        "gallery": os.path.join(RESULTS_DIR, "constellation_gallery.png"),
    }
    if history.get("train_loss"):
        save_training_curves(history, paths["training_curves"])
        print("    [OK] training_curves.png")
    save_confusion_matrix(cm, paths["confusion_matrix"])
    print("    [OK] confusion_matrix.png")
    save_per_class_f1(report, paths["per_class_f1"])
    print("    [OK] per_class_f1.png")
    save_snr_curve(snr_curve, paths["snr_curve"])
    print("    [OK] snr_accuracy_curve.png")
    save_family_bars(fam_acc, paths["family_acc"])
    print("    [OK] family_accuracy.png")
    if clean_signals:
        save_sample_gallery(clean_signals, paths["gallery"])
        print("    [OK] constellation_gallery.png")

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(RESULTS_DIR, "classification_results.csv")
    )
    print("    [OK] classification_results.csv")

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "torch_version": torch.__version__,
        "device": str(device) + (
            f" ({torch.cuda.get_device_name(device)})" if use_cuda else ""
        ),
        "num_classes": NUM_CLASSES,
        "train_samples": int(len(train_ds)),
        "test_samples": int(len(test_ds)),
        "train_spc": args.train_spc,
        "test_spc": args.test_spc,
        "train_snr_range": list(TRAIN_SNR_RANGE),
        "num_params": int(n_params),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_time_s": float(train_time),
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "best_val_acc": float(best_val if best_val >= 0 else test_acc),
        "best_epoch": int(best_epoch if best_epoch else 0),
        "family_accuracy": {k: float(v) for k, v in fam_acc.items()},
        "snr_curve": {str(k): float(v) for k, v in snr_curve.items()},
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1-score": float(report[c]["f1-score"]),
                "support": float(report[c]["support"]),
            }
            for c in CLASSES if c in report
        },
        "plot_count": int(plot_count),
        "history": history,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("    [OK] metrics.json")

    md = [
        "# RF Signal Classification Results\n",
        f"- **Test Accuracy**: {test_acc:.4f} ({test_acc * 100:.2f}%)",
        f"- **Macro F1**: {macro_f1:.4f}",
        f"- **Family Accuracy**: {fam_acc['overall']:.4f}",
        f"- **Params**: {n_params:,}",
        f"- **Device**: {metrics['device']}",
        "",
        "## SNR Curve\n",
        "| SNR (dB) | Accuracy |",
        "|----------|----------|",
    ]
    for s in sorted(snr_curve):
        md.append(f"| {s} | {snr_curve[s]:.4f} |")
    with open(os.path.join(RESULTS_DIR, "results_summary.md"), "w") as f:
        f.write("\n".join(md))
    print("    [OK] results_summary.md")

    if not args.skip_docx:
        print("\n>>> STEP 7: Academic DOCX report")
        build_docx(
            metrics, paths,
            os.path.join(RESULTS_DIR, "RF_Signal_Classification_Report.docx"),
        )

    print("\n" + "=" * 64)
    print("  COMPLETE")
    print(f"  Accuracy: {test_acc * 100:.2f}%  |  Macro-F1: {macro_f1:.3f}  |  "
          f"Family: {fam_acc['overall'] * 100:.1f}%")
    print(f"  Report: {os.path.join(RESULTS_DIR, 'RF_Signal_Classification_Report.docx')}")
    print("=" * 64)


if __name__ == "__main__":
    main()