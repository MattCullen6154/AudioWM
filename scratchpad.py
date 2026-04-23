"""
Scratchpad experiments moved out of 123Final.ipynb.

This file keeps the batch sweep/tabulation and shared-key decoder ideas handy
without making them part of the main notebook flow. It is intended to be run
from the notebook after the core helper cells have been executed, so functions
such as load_audio, embed_spread_spectrum, apply_test_channel, and snr_db are
expected to already exist in the notebook namespace.
"""

import numpy as np
import pandas as pd


# Audio containers for batch sweeps. Keep generated outputs out of this list.
AUDIO_SOURCES = {
    "audio": "data/audio.wav",
    "cortes": "data/Cortes.wav",
    "cortes2": "data/Cortes2.wav",
    "mom_and_bald": "data/MomAndBald.wav",
    "moon_landing": "data/MoonLanding.wav",
    "sitting_outside": "data/SittingDownOutside.wav",
    "sitting_outside_2": "data/SittingDownOutside2.wav",
    "talking_to_bald": "data/TalkingToBald.wav",
    "talking_to_therese_1": "data/TalkingToTherese1.wav",
    "talking_to_therese_2": "data/TalkingToTherese2.wav",
    "talking_to_therese_3": "data/TalkingToTherese3.wav",
    "talking_to_therese_4": "data/TalkingToTherese4.wav",
}

WATERMARK_KEY_SEED = 12345

WATERMARK_LIBRARY = {
    "current": np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=int),
    "alternating": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int),
    "inverse_current": np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=int),
    "dense": np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=int),
}

DISTORTION_SWEEP = [
    {"name": "clean", "bandlimit": False, "awgn": False, "resample": False, "quantization": False},
    {
        "name": "bandlimit",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": False,
        "resample": False,
        "quantization": False,
    },
    {
        "name": "bandlimit_noise_low",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": True,
        "noise_std": 0.001,
        "resample": False,
        "quantization": False,
    },
    {
        "name": "bandlimit_noise_med",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": True,
        "noise_std": 0.003,
        "resample": False,
        "quantization": False,
    },
    {
        "name": "bandlimit_noise_high",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": True,
        "noise_std": 0.006,
        "resample": False,
        "quantization": False,
    },
    {
        "name": "resample_8k",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": True,
        "noise_std": 0.003,
        "resample": True,
        "target_fs": 8000,
        "quantization": False,
    },
    {
        "name": "quant_8bit",
        "bandlimit": True,
        "band_low": 300,
        "band_high": 3400,
        "awgn": True,
        "noise_std": 0.003,
        "resample": False,
        "quantization": True,
        "quant_bits": 8,
    },
]

SWEEP_AUDIO_NAMES = ["audio", "cortes2", "moon_landing"]
SWEEP_WATERMARK_NAMES = list(WATERMARK_LIBRARY)


def derive_watermark_key(fs, nfft=1024, f_low=600, f_high=2500, key_seed=WATERMARK_KEY_SEED):
    freqs = compute_fft_bins(fs, nfft)
    candidate_bins = np.where(get_bin_mask(freqs, f_low, f_high))[0]
    selected_bins = candidate_bins[::2]

    key_rng = np.random.default_rng(key_seed)
    spreading_code = make_spreading_code(len(selected_bins), rng=key_rng)
    return selected_bins, spreading_code


def decode_spread_spectrum_shared_key(
    y,
    fs,
    num_bits,
    key_seed=WATERMARK_KEY_SEED,
    frame_ms=20,
    hop_ms=10,
    nfft=1024,
    f_low=600,
    f_high=2500,
    repeat=4,
):
    frame_len = ms_to_samples(frame_ms, fs)
    hop_len = ms_to_samples(hop_ms, fs)
    frames = frame_signal(y, frame_len, hop_len)
    w = get_window(frame_len)
    Y = np.fft.rfft(frames * w[None, :], n=nfft, axis=1)

    selected_bins, spreading_code = derive_watermark_key(
        fs,
        nfft=nfft,
        f_low=f_low,
        f_high=f_high,
        key_seed=key_seed,
    )
    needed_frames = num_bits * repeat
    frame_indices, _, _ = choose_embedding_frames(
        frames,
        needed_frames=needed_frames,
        min_fraction_of_peak=0.05,
        min_spacing=max(1, frame_len // hop_len),
    )

    scores = []
    for frame_idx in frame_indices[:needed_frames]:
        mag = np.abs(Y[frame_idx, selected_bins])
        mag_centered = mag - np.mean(mag)
        score = np.sum(mag_centered * spreading_code) / len(selected_bins)
        scores.append(score)

    scores = np.array(scores)
    num_decoded = len(scores) // repeat
    bit_scores = np.array([
        np.sum(scores[i * repeat:(i + 1) * repeat]) for i in range(num_decoded)
    ])
    decoded_bits = (bit_scores > 0).astype(int)
    return decoded_bits, bit_scores, scores, frame_indices[:needed_frames]


def load_audio_source(name_or_path):
    path = AUDIO_SOURCES.get(name_or_path, name_or_path)
    x, fs = load_audio(path)
    return normalize_audio(x), fs, path


def run_watermark_trial(
    audio_name,
    watermark_name,
    channel_config,
    seed=12345,
    embed_strength=0.5,
    bit_repeat=4,
):
    x, fs, path = load_audio_source(audio_name)
    bits = np.asarray(WATERMARK_LIBRARY[watermark_name], dtype=int)

    embed_rng = np.random.default_rng(seed)
    channel_rng = np.random.default_rng(seed + 1)

    y_wm, meta = embed_spread_spectrum(
        x=x,
        fs=fs,
        bits=bits,
        frame_ms=FRAME_MS,
        hop_ms=HOP_MS,
        nfft=NFFT,
        f_low=FREQ_LOW,
        f_high=FREQ_HIGH,
        alpha=embed_strength,
        repeat=bit_repeat,
        rng=embed_rng,
    )
    y_rx = apply_test_channel(y_wm, fs, rng=channel_rng, channel_config=channel_config)
    decoded_bits, bit_scores, frame_scores = decode_spread_spectrum(
        y=y_rx,
        fs=fs,
        metadata=meta,
        frame_ms=FRAME_MS,
        hop_ms=HOP_MS,
        nfft=NFFT,
        repeat=bit_repeat,
    )

    decoded_slice = decoded_bits[:len(bits)]
    score_slice = bit_scores[:len(bits)]
    return {
        "audio": audio_name,
        "path": path,
        "watermark": watermark_name,
        "channel": channel_config.get("name", "custom"),
        "noise_std": channel_config.get("noise_std", 0.0) if channel_config.get("awgn", False) else 0.0,
        "band_low": channel_config.get("band_low", np.nan) if channel_config.get("bandlimit", False) else np.nan,
        "band_high": channel_config.get("band_high", np.nan) if channel_config.get("bandlimit", False) else np.nan,
        "target_fs": channel_config.get("target_fs", np.nan) if channel_config.get("resample", False) else np.nan,
        "quant_bits": channel_config.get("quant_bits", np.nan) if channel_config.get("quantization", False) else np.nan,
        "embed_strength": embed_strength,
        "bit_repeat": bit_repeat,
        "embedding_snr_db": snr_db(x, y_wm),
        "channel_snr_db": snr_db(y_wm, y_rx),
        "ber": bit_error_rate(bits, decoded_slice),
        "num_bit_errors": int(np.sum(bits[:len(decoded_slice)] != decoded_slice)),
        "score_margin": float(np.min(np.abs(score_slice))) if len(score_slice) else np.nan,
        "decoded_bits": "".join(str(int(b)) for b in decoded_slice),
        "true_bits": "".join(str(int(b)) for b in bits),
        "usable_frames": meta["usable_frames"],
    }


def run_watermark_sweep(
    audio_names=SWEEP_AUDIO_NAMES,
    watermark_names=SWEEP_WATERMARK_NAMES,
    channel_sweep=DISTORTION_SWEEP,
    seed=12345,
):
    rows = []
    for audio_name in audio_names:
        for watermark_name in watermark_names:
            for channel_config in channel_sweep:
                rows.append(run_watermark_trial(audio_name, watermark_name, channel_config, seed=seed))
    return pd.DataFrame(rows)


def summarize_sweep(results):
    summary_cols = [
        "channel",
        "noise_std",
        "band_low",
        "band_high",
        "target_fs",
        "quant_bits",
        "ber",
        "num_bit_errors",
        "score_margin",
        "embedding_snr_db",
        "channel_snr_db",
    ]
    return (
        results[summary_cols]
        .groupby(["channel", "noise_std", "band_low", "band_high", "target_fs", "quant_bits"], dropna=False)
        .agg(
            mean_ber=("ber", "mean"),
            max_ber=("ber", "max"),
            total_bit_errors=("num_bit_errors", "sum"),
            min_score_margin=("score_margin", "min"),
            mean_embedding_snr_db=("embedding_snr_db", "mean"),
            mean_channel_snr_db=("channel_snr_db", "mean"),
        )
        .reset_index()
        .sort_values(["mean_ber", "min_score_margin"], ascending=[False, True])
    )
