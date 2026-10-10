"""Audio feature extraction module for SHL English Grammar Scoring Engine.

Extracts signal properties, spectral features, MFCCs, prosodic/pitch metrics,
and pause/silence statistics from WAV audio files using efficient vectorized operations.
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis
from tqdm import tqdm


def extract_single_audio_features(filepath: str, hop_length: int = 1024) -> dict:
    """Extract a comprehensive numerical feature dictionary from a single audio file.
    
    Args:
        filepath (str): Absolute or relative path to the WAV audio file.
        hop_length (int): Hop length for frame-based audio transformations.
        
    Returns:
        dict: Extracted numerical features and status indicators.
    """
    feats = {
        'extraction_error': False,
        'error_msg': ''
    }
    
    try:
        if not os.path.exists(filepath):
            feats['extraction_error'] = True
            feats['error_msg'] = f"File not found: {filepath}"
            return _populate_fallback_features(feats)
        
        y, sr = sf.read(filepath)
        
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        if len(y) == 0 or sr <= 0:
            feats['extraction_error'] = True
            feats['error_msg'] = "Empty audio signal or invalid sample rate."
            return _populate_fallback_features(feats)
            
        duration = float(len(y) / sr)
        feats['duration_sec'] = duration
        feats['sample_rate'] = int(sr)
        feats['num_samples'] = int(len(y))
        
        # Raw Signal Amplitude Statistics
        amp_abs = np.abs(y)
        feats['amp_mean'] = float(np.mean(amp_abs))
        feats['amp_std'] = float(np.std(y))
        feats['amp_max'] = float(np.max(amp_abs))
        feats['amp_min'] = float(np.min(amp_abs))
        feats['amp_skew'] = float(skew(y)) if len(y) > 2 else 0.0
        feats['amp_kurtosis'] = float(kurtosis(y)) if len(y) > 2 else 0.0
        
        # RMS Energy & Zero Crossing Rate
        rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        feats['rms_mean'] = float(np.mean(rms_frames))
        feats['rms_std'] = float(np.std(rms_frames))
        feats['rms_max'] = float(np.max(rms_frames))
        feats['rms_min'] = float(np.min(rms_frames))
        
        zcr_frames = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        feats['zcr_mean'] = float(np.mean(zcr_frames))
        feats['zcr_std'] = float(np.std(zcr_frames))
        feats['zcr_max'] = float(np.max(zcr_frames))
        
        # Spectral Features
        sc_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        feats['spectral_centroid_mean'] = float(np.mean(sc_frames))
        feats['spectral_centroid_std'] = float(np.std(sc_frames))
        
        sb_frames = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        feats['spectral_bandwidth_mean'] = float(np.mean(sb_frames))
        feats['spectral_bandwidth_std'] = float(np.std(sb_frames))
        
        sroll_frames = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        feats['spectral_rolloff_mean'] = float(np.mean(sroll_frames))
        feats['spectral_rolloff_std'] = float(np.std(sroll_frames))
        
        sflat_frames = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        feats['spectral_flatness_mean'] = float(np.mean(sflat_frames))
        feats['spectral_flatness_std'] = float(np.std(sflat_frames))
        
        scontrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        for band_idx in range(scontrast.shape[0]):
            feats[f'spectral_contrast_band_{band_idx}_mean'] = float(np.mean(scontrast[band_idx]))
            feats[f'spectral_contrast_band_{band_idx}_std'] = float(np.std(scontrast[band_idx]))
            
        # MFCCs (13 coefficients), Deltas & Delta-Deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        for i in range(13):
            coeff_num = i + 1
            feats[f'mfcc_{coeff_num}_mean'] = float(np.mean(mfcc[i]))
            feats[f'mfcc_{coeff_num}_std'] = float(np.std(mfcc[i]))
            feats[f'mfcc_{coeff_num}_max'] = float(np.max(mfcc[i]))
            feats[f'mfcc_{coeff_num}_min'] = float(np.min(mfcc[i]))
            
            feats[f'mfcc_delta_{coeff_num}_mean'] = float(np.mean(mfcc_delta[i]))
            feats[f'mfcc_delta_{coeff_num}_std'] = float(np.std(mfcc_delta[i]))
            
            feats[f'mfcc_delta2_{coeff_num}_mean'] = float(np.mean(mfcc_delta2[i]))
            feats[f'mfcc_delta2_{coeff_num}_std'] = float(np.std(mfcc_delta2[i]))
            
        # Silence & Pause Dynamics
        mean_rms = np.mean(rms_frames) if len(rms_frames) > 0 else 0.0
        threshold = 0.02 * np.max(amp_abs) if np.max(amp_abs) > 0 else 0.01
        silent_frames = (rms_frames < (0.15 * mean_rms)) if mean_rms > 0 else (amp_abs < threshold)
        silence_ratio = float(np.mean(silent_frames))
        feats['silence_duration_ratio'] = silence_ratio
        feats['speech_duration_ratio'] = float(1.0 - silence_ratio)
        
        frame_dur = float(hop_length) / sr
        pause_min_frames = int(0.25 / frame_dur) if frame_dur > 0 else 5
        
        pauses = 0
        current_pause_len = 0
        pause_lengths = []
        for is_silent in silent_frames:
            if is_silent:
                current_pause_len += 1
            else:
                if current_pause_len >= pause_min_frames:
                    pauses += 1
                    pause_lengths.append(current_pause_len * frame_dur)
                current_pause_len = 0
        if current_pause_len >= pause_min_frames:
            pauses += 1
            pause_lengths.append(current_pause_len * frame_dur)
            
        feats['pause_count'] = int(pauses)
        feats['mean_pause_duration_sec'] = float(np.mean(pause_lengths)) if len(pause_lengths) > 0 else 0.0
        feats['pause_rate_per_min'] = float((pauses / duration) * 60.0) if duration > 0 else 0.0
        
        # Pitch (F0) & Voicing Estimation (Vectorized 2D FFT Autocorrelation)
        frame_len = 2048
        if len(y) > frame_len:
            frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_length)
            n_fft = 4096
            fft_f = np.fft.rfft(frames, n=n_fft, axis=0)
            power_f = np.abs(fft_f)**2
            corr_matrix = np.fft.irfft(power_f, axis=0)[:frame_len, :]
            
            fmin, fmax = 65, 500
            min_lag = max(1, int(sr / fmax))
            max_lag = min(frame_len - 1, int(sr / fmin))
            
            peak_indices = min_lag + np.argmax(corr_matrix[min_lag:max_lag, :], axis=0)
            r0 = corr_matrix[0, :]
            r_peak = corr_matrix[peak_indices, np.arange(corr_matrix.shape[1])]
            
            valid_mask = (r0 > 1e-4) & ((r_peak / r0) > 0.35)
            f0_voiced = (sr / peak_indices)[valid_mask]
        else:
            f0_voiced = np.array([])
            
        if len(f0_voiced) > 0:
            feats['pitch_mean_hz'] = float(np.mean(f0_voiced))
            feats['pitch_std_hz'] = float(np.std(f0_voiced))
            feats['pitch_min_hz'] = float(np.min(f0_voiced))
            feats['pitch_max_hz'] = float(np.max(f0_voiced))
            total_possible_frames = max(1, len(y) // hop_length)
            feats['voiced_frame_ratio'] = float(len(f0_voiced) / total_possible_frames)
        else:
            feats['pitch_mean_hz'] = 0.0
            feats['pitch_std_hz'] = 0.0
            feats['pitch_min_hz'] = 0.0
            feats['pitch_max_hz'] = 0.0
            feats['voiced_frame_ratio'] = 0.0
            
    except Exception as exc:
        feats['extraction_error'] = True
        feats['error_msg'] = f"Unhandled exception: {str(exc)}"
        return _populate_fallback_features(feats)
        
    # Clean non-finite values (inf, -inf, NaN)
    for k, v in feats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feats[k] = 0.0
            
    return feats


def _populate_fallback_features(feats: dict) -> dict:
    """Populate default zero/null values for all expected feature keys on failure."""
    default_keys = [
        'duration_sec', 'sample_rate', 'num_samples', 'amp_mean', 'amp_std', 'amp_max', 'amp_min',
        'amp_skew', 'amp_kurtosis', 'rms_mean', 'rms_std', 'rms_max', 'rms_min', 'zcr_mean', 'zcr_std',
        'zcr_max', 'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_bandwidth_mean',
        'spectral_bandwidth_std', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_flatness_mean',
        'spectral_flatness_std', 'silence_duration_ratio', 'speech_duration_ratio', 'pause_count',
        'mean_pause_duration_sec', 'pause_rate_per_min', 'pitch_mean_hz', 'pitch_std_hz', 'pitch_min_hz',
        'pitch_max_hz', 'voiced_frame_ratio'
    ]
    for b in range(7):
        default_keys.extend([f'spectral_contrast_band_{b}_mean', f'spectral_contrast_band_{b}_std'])
    for i in range(1, 14):
        default_keys.extend([
            f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_max', f'mfcc_{i}_min',
            f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
            f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
        ])
        
    for k in default_keys:
        if k not in feats:
            feats[k] = 0.0
            
    return feats


def extract_audio_features_dataset(df: pd.DataFrame, audio_dir: str, filename_col: str = 'filename', show_progress: bool = True) -> tuple[pd.DataFrame, list]:
    """Extract audio features across all samples in a dataframe.
    
    Args:
        df (pd.DataFrame): Dataset dataframe containing audio filenames.
        audio_dir (str): Folder containing physical .wav files.
        filename_col (str): Name of the column containing filenames.
        show_progress (bool): Display tqdm progress bar.
        
    Returns:
        tuple[pd.DataFrame, list]: Dataframe of extracted features and list of error dicts.
    """
    feature_rows = []
    error_log = []
    
    # Warmup librosa / Numba once to avoid first-call lag
    warmup_file = os.path.join(audio_dir, df[filename_col].iloc[0]) if len(df) > 0 else None
    if warmup_file and os.path.exists(warmup_file):
        _ = extract_single_audio_features(warmup_file)
        
    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Extracting audio features from {os.path.basename(audio_dir)}") if show_progress else df.iterrows()
    
    for _, row in iterator:
        fname = row[filename_col]
        fpath = os.path.join(audio_dir, fname)
        
        feats = extract_single_audio_features(fpath)
        feats[filename_col] = fname
        
        if feats['extraction_error']:
            error_log.append({
                'filename': fname,
                'filepath': fpath,
                'error': feats['error_msg']
            })
            
        feature_rows.append(feats)
        
    features_df = pd.DataFrame(feature_rows)
    
    # Reorder columns so filename is first
    cols = [filename_col] + [c for c in features_df.columns if c not in [filename_col, 'extraction_error', 'error_msg']] + ['extraction_error', 'error_msg']
    features_df = features_df[cols]
    
    return features_df, error_log


def save_audio_features(features_df: pd.DataFrame, output_path: str) -> None:
    """Save extracted audio features dataframe to CSV.
    
    Args:
        features_df (pd.DataFrame): Dataframe of audio features.
        output_path (str): Target CSV file destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"[SAVED] Audio features saved to {output_path} (Shape: {features_df.shape})")
