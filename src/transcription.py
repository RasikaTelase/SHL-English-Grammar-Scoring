"""Speech transcription module for SHL English Grammar Scoring Engine.

Transcribes WAV speech audio files using OpenAI Whisper (tiny.en / base.en),
with PyTorch CPU multi-threading optimization, automatic caching, incremental checkpointing, and error handling.
"""

import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from tqdm import tqdm

# Optimize PyTorch CPU thread allocation for fast matrix math
try:
    torch.set_num_threads(4)
except Exception:
    pass


def load_whisper_model(model_name: str = 'tiny.en'):
    """Load and return an OpenAI Whisper model.
    
    Args:
        model_name (str): Model size name (e.g. 'tiny.en', 'tiny', 'base.en', 'base').
        
    Returns:
        whisper.Whisper: Loaded Whisper model instance.
    """
    print(f"[INFO] Loading Whisper model '{model_name}' (PyTorch CPU num_threads={torch.get_num_threads()})...")
    model = whisper.load_model(model_name)
    print(f"[SUCCESS] Whisper model '{model_name}' loaded successfully.")
    return model


def transcribe_single_audio(filepath: str, model, language: str = 'en') -> dict:
    """Transcribe a single WAV audio file into text using Whisper.
    
    Args:
        filepath (str): Absolute or relative path to the audio file.
        model (whisper.Whisper): Pre-loaded Whisper model instance.
        language (str): Target language code ('en').
        
    Returns:
        dict: Transcript text, error flag, and error message.
    """
    result = {
        'transcript': '',
        'transcription_error': False,
        'error_msg': ''
    }
    
    try:
        if not os.path.exists(filepath):
            result['transcription_error'] = True
            result['error_msg'] = f"File not found: {filepath}"
            return result
            
        # Read WAV using soundfile (bypasses ffmpeg system dependency)
        y, sr = sf.read(filepath)
        
        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        if len(y) == 0 or sr <= 0:
            result['transcription_error'] = True
            result['error_msg'] = "Empty audio signal or invalid sample rate."
            return result
            
        # Ensure float32 format for PyTorch
        y = y.astype(np.float32)
        
        # Transcribe with Whisper (fp16=False for CPU compatibility)
        if hasattr(model, 'name') and 'en' in getattr(model, 'name', ''):
            res = model.transcribe(y, fp16=False)
        else:
            res = model.transcribe(y, fp16=False, language=language)
            
        raw_text = res.get('text', '').strip()
        result['transcript'] = raw_text
        
        if len(raw_text) == 0:
            result['transcription_error'] = True
            result['error_msg'] = "Whisper returned empty transcript text."
            
    except Exception as exc:
        result['transcription_error'] = True
        result['error_msg'] = f"Exception during transcription: {str(exc)}"
        
    return result


def transcribe_dataset(df: pd.DataFrame, 
                       audio_dir: str, 
                       output_csv_path: str = None, 
                       model_name: str = 'tiny.en', 
                       filename_col: str = 'filename',
                       checkpoint_interval: int = 20,
                       show_progress: bool = True) -> tuple[pd.DataFrame, list]:
    """Transcribe a dataset of audio files with caching and checkpointing.
    
    Args:
        df (pd.DataFrame): Dataset dataframe containing filenames.
        audio_dir (str): Folder containing physical .wav files.
        output_csv_path (str): File destination path for saving/caching CSV.
        model_name (str): Whisper model variant ('tiny.en', 'base.en', etc.).
        filename_col (str): Filename column name.
        checkpoint_interval (int): Save progress every N samples.
        show_progress (bool): Display tqdm progress bar.
        
    Returns:
        tuple[pd.DataFrame, list]: Transcripts dataframe and error log.
    """
    cached_df = None
    if output_csv_path and os.path.exists(output_csv_path):
        try:
            cached_df = pd.read_csv(output_csv_path)
            print(f"[CACHE] Found existing transcript file at {output_csv_path} (Shape: {cached_df.shape})")
        except Exception as e:
            print(f"[WARN] Failed to read existing transcript cache: {e}")
            cached_df = None

    # Map existing cached transcripts by filename
    cache_dict = {}
    if cached_df is not None and filename_col in cached_df.columns and 'transcript' in cached_df.columns:
        for _, r in cached_df.iterrows():
            fname = r[filename_col]
            err = r.get('transcription_error', False)
            txt = str(r.get('transcript', '')) if pd.notna(r.get('transcript')) else ''
            # Count as valid cached if not marked error and not empty
            if not err and len(txt) > 0:
                cache_dict[fname] = {
                    'transcript': txt,
                    'transcription_error': False,
                    'error_msg': str(r.get('error_msg', '')) if pd.notna(r.get('error_msg')) else ''
                }
                
    missing_filenames = [fname for fname in df[filename_col] if fname not in cache_dict]
    print(f"[CACHE STATS] Total files: {len(df)} | Cached valid transcripts: {len(cache_dict)} | To process: {len(missing_filenames)}")
    
    model = None
    if len(missing_filenames) > 0:
        model = load_whisper_model(model_name)
        
    results_map = cache_dict.copy()
    error_log = []
    
    if len(missing_filenames) > 0:
        iterator = tqdm(missing_filenames, desc=f"Transcribing {os.path.basename(audio_dir)}") if show_progress else missing_filenames
        processed_count = 0
        
        for fname in iterator:
            fpath = os.path.join(audio_dir, fname)
            res = transcribe_single_audio(fpath, model)
            results_map[fname] = res
            
            if res['transcription_error']:
                error_log.append({
                    'filename': fname,
                    'error': res['error_msg']
                })
                
            processed_count += 1
            
            # Save periodic checkpoints
            if output_csv_path and (processed_count % checkpoint_interval == 0):
                temp_rows = []
                for orig_fname in df[filename_col]:
                    if orig_fname in results_map:
                        item = results_map[orig_fname]
                        temp_rows.append({
                            filename_col: orig_fname,
                            'transcript': item['transcript'],
                            'transcription_error': item['transcription_error'],
                            'error_msg': item['error_msg']
                        })
                temp_df = pd.DataFrame(temp_rows)
                os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
                temp_df.to_csv(output_csv_path, index=False)
                
    # Reconstruct final dataframe preserving exact original order of df
    final_rows = []
    for fname in df[filename_col]:
        item = results_map.get(fname, {
            'transcript': '',
            'transcription_error': True,
            'error_msg': 'Not processed'
        })
        final_rows.append({
            filename_col: fname,
            'transcript': item['transcript'],
            'word_count': len(item['transcript'].split()) if item['transcript'] else 0,
            'char_count': len(item['transcript']),
            'transcription_error': item['transcription_error'],
            'error_msg': item['error_msg']
        })
        
    transcripts_df = pd.DataFrame(final_rows)
    
    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        transcripts_df.to_csv(output_csv_path, index=False)
        print(f"[SAVED] Final transcripts saved to {output_csv_path} (Shape: {transcripts_df.shape})")
        
    return transcripts_df, error_log
