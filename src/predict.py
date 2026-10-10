"""
src/predict.py

Inference script for predicting continuous English grammar scores (0.0 to 5.0)
from a single raw WAV audio file using the trained SHL Grammar Scoring Engine pipeline.

Usage:
    python src/predict.py <path_to_audio_file.wav>

Example:
    python src/predict.py C:\\Users\\rasik\\Downloads\\shl-hiring-assessment-2026\\Dataset_Final\\test\\audio_128.wav
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Add src to sys.path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_features import extract_single_audio_features
from transcription import load_whisper_model, transcribe_single_audio
from text_features import extract_single_text_features
from grammar_features import extract_text_grammar_indicators


def predict_audio_file(
    wav_path: str,
    model_path: str = r'models/best_model.joblib',
    vectorizer_path: str = r'models/tfidf_vectorizer.joblib',
    whisper_model_name: str = 'tiny.en',
    verbose: bool = True
) -> dict:
    """
    Predict grammar score (0.0 to 5.0) for a single input WAV audio file.

    Args:
        wav_path (str): Path to the input WAV file.
        model_path (str): Path to the saved joblib model artifact.
        vectorizer_path (str): Path to the saved TF-IDF vectorizer artifact.
        whisper_model_name (str): Whisper ASR model variant name ('tiny.en').
        verbose (bool): Whether to print progress messages.

    Returns:
        dict: Result dictionary containing filename, transcript, predicted_score, and features.
    """
    # 1. Error handling: Verify input file existence
    if not os.path.exists(wav_path):
        err_msg = f"ERROR: Input audio file does not exist: {wav_path}"
        if verbose:
            print(err_msg, file=sys.stderr)
        return {'filename': os.path.basename(wav_path), 'predicted_score': 0.0, 'transcript': '', 'error': err_msg}

    # 2. Error handling: Verify model artifacts
    if not os.path.exists(model_path):
        err_msg = f"ERROR: Model artifact not found at {model_path}. Train the model first."
        if verbose:
            print(err_msg, file=sys.stderr)
        return {'filename': os.path.basename(wav_path), 'predicted_score': 0.0, 'transcript': '', 'error': err_msg}

    if not os.path.exists(vectorizer_path):
        err_msg = f"ERROR: TF-IDF vectorizer artifact not found at {vectorizer_path}."
        if verbose:
            print(err_msg, file=sys.stderr)
        return {'filename': os.path.basename(wav_path), 'predicted_score': 0.0, 'transcript': '', 'error': err_msg}

    filename = os.path.basename(wav_path)

    # 3. Load artifacts
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    whisper_model = load_whisper_model(whisper_model_name)

    # 4. Extract Audio Features
    audio_feats = extract_single_audio_features(wav_path)
    if audio_feats.get('extraction_error', False):
        if verbose:
            print(f"[WARNING] Audio feature extraction warning: {audio_feats.get('error_msg')}")

    # 5. ASR Speech Transcription
    trans_res = transcribe_single_audio(wav_path, whisper_model)
    text = trans_res.get('transcript', '')

    # 6. Text & Linguistic Features
    text_feats = extract_single_text_features(text, filename=filename)

    # 7. TF-IDF Features
    tfidf_arr = tfidf_vectorizer.transform([text]).toarray()[0]
    tfidf_feature_names = [f'tfidf_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()]
    tfidf_feats = dict(zip(tfidf_feature_names, tfidf_arr))

    # 8. Grammar & Fluency Indicators
    grammar_text_feats = extract_text_grammar_indicators(text)

    # Speech-rate alignment metrics
    duration = float(audio_feats.get('duration_sec', 0.0))
    speech_ratio = float(audio_feats.get('speech_duration_ratio', 0.0))
    silence_ratio = float(audio_feats.get('silence_duration_ratio', 0.0))
    pause_count = float(audio_feats.get('pause_count', 0.0))
    mean_pause_sec = float(audio_feats.get('mean_pause_duration_sec', 0.0))

    words = text.split()
    word_count = len(words)

    if word_count == 0 or duration == 0:
        speech_rate_wpm = 0.0
        articulation_rate_wps = 0.0
        pause_per_word_ratio = 0.0
        silence_sec_per_word = 0.0
        words_per_pause = 0.0
    else:
        speech_rate_wpm = (word_count / duration) * 60.0
        active_speech_sec = duration * speech_ratio
        articulation_rate_wps = (word_count / active_speech_sec) if active_speech_sec > 0 else 0.0
        pause_per_word_ratio = pause_count / max(1, word_count)
        silence_sec_per_word = (duration * silence_ratio) / max(1, word_count)
        words_per_pause = word_count / max(1.0, pause_count)

    alignment_feats = {
        'speech_rate_wpm': float(speech_rate_wpm),
        'articulation_rate_wps': float(articulation_rate_wps),
        'pause_per_word_ratio': float(pause_per_word_ratio),
        'silence_sec_per_word': float(silence_sec_per_word),
        'words_per_pause': float(words_per_pause),
        'mean_pause_duration_sec': float(mean_pause_sec)
    }

    # Combine all feature dicts
    combined = {}
    combined.update(audio_feats)
    combined.update(text_feats)
    combined.update(alignment_feats)
    combined.update(grammar_text_feats)
    combined.update(tfidf_feats)

    # 9. Extract expected feature column names from fitted scaler
    scaler = model.named_steps['scaler']
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = list(scaler.feature_names_in_)
    else:
        # Fallback to loading train data columns if scaler doesn't store feature_names_in_
        from train import load_and_prepare_data
        X_train, _, _, _, _ = load_and_prepare_data()
        expected_cols = list(X_train.columns)

    single_df = pd.DataFrame([combined])

    # Ensure all expected columns are present
    for col in expected_cols:
        if col not in single_df.columns:
            single_df[col] = 0.0

    X_single = single_df[expected_cols].select_dtypes(include=[np.number])
    X_single = X_single.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 10. Predict & Clip to [0.0, 5.0]
    raw_pred = model.predict(X_single)[0]
    final_score = float(np.clip(raw_pred, 0.0, 5.0))

    if verbose:
        print("\n==================================================")
        print("  SHL English Grammar Scoring Engine — Inference")
        print("==================================================")
        print(f"Filename:          {filename}")
        print(f"Transcript:        \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"Predicted Score:   {final_score:.4f} / 5.00")
        print("==================================================\n")

    return {
        'filename': filename,
        'predicted_score': final_score,
        'transcript': text,
        'error': None
    }


def main():
    parser = argparse.ArgumentParser(description="Predict English Grammar Score for a single WAV audio file.")
    parser.add_argument('wav_path', type=str, help="Path to the WAV audio file.")
    parser.add_argument('--model_path', type=str, default=r'models/best_model.joblib', help="Path to saved model pipeline.")
    parser.add_argument('--vectorizer_path', type=str, default=r'models/tfidf_vectorizer.joblib', help="Path to saved TF-IDF vectorizer.")
    parser.add_argument('--whisper_model', type=str, default='tiny.en', help="Whisper ASR model size.")

    args = parser.parse_args()
    predict_audio_file(
        wav_path=args.wav_path,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        whisper_model_name=args.whisper_model,
        verbose=True
    )


if __name__ == '__main__':
    main()
