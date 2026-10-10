"""
src/train.py

Module for loading combined audio, text, grammar, and TF-IDF features,
performing reproducible train/validation splits and 5-fold cross-validation,
evaluating baseline regression models, selecting the best model,
and saving fitted model artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)


def load_and_prepare_data(data_dir=r'd:\AI-Grammar-Scoring\data', features_dir=r'd:\AI-Grammar-Scoring\artifacts\features'):
    """
    Load train.csv, test.csv, and all audio/text/grammar/TF-IDF feature files.
    Merge by explicit filename key without column duplication and return numeric feature matrices X_train, X_test,
    labels y_train, and filename series.
    """
    train_csv_path = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(train_csv_path):
        train_csv_path = r'C:\Users\rasik\Downloads\shl-hiring-assessment-2026\Dataset_Final\train.csv'
        
    test_csv_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(test_csv_path):
        test_csv_path = r'C:\Users\rasik\Downloads\shl-hiring-assessment-2026\Dataset_Final\test.csv'

    train_meta = pd.read_csv(train_csv_path)
    test_meta = pd.read_csv(test_csv_path)

    # Load feature files
    audio_tr = pd.read_csv(os.path.join(features_dir, 'train_audio_features.csv'))
    audio_te = pd.read_csv(os.path.join(features_dir, 'test_audio_features.csv'))

    text_tr = pd.read_csv(os.path.join(features_dir, 'train_text_features.csv'))
    text_te = pd.read_csv(os.path.join(features_dir, 'test_text_features.csv'))

    grammar_tr = pd.read_csv(os.path.join(features_dir, 'train_grammar_features.csv'))
    grammar_te = pd.read_csv(os.path.join(features_dir, 'test_grammar_features.csv'))

    tfidf_tr = pd.read_csv(os.path.join(features_dir, 'train_tfidf_features.csv'))
    tfidf_te = pd.read_csv(os.path.join(features_dir, 'test_tfidf_features.csv'))

    def merge_features(meta_df, audio_df, text_df, grammar_df, tfidf_df):
        merged = meta_df[['filename']].merge(audio_df, on='filename', how='left')
        for df_next in [text_df, grammar_df, tfidf_df]:
            existing_cols = set(merged.columns) - {'filename'}
            new_cols = [c for c in df_next.columns if c == 'filename' or c not in existing_cols]
            merged = merged.merge(df_next[new_cols], on='filename', how='left')

        exclude_cols = [
            'filename', 'label', 'transcript', 'transcription_error', 'error_msg',
            'grammar_feature_status', 'text_feature_status'
        ]
        drop_cols = [c for c in merged.columns if c in exclude_cols]
        clean_df = merged.drop(columns=drop_cols)
        
        # Select strictly numeric columns
        num_cols = clean_df.select_dtypes(include=[np.number]).columns
        return clean_df[num_cols]

    X_train = merge_features(train_meta, audio_tr, text_tr, grammar_tr, tfidf_tr)
    X_test = merge_features(test_meta, audio_te, text_te, grammar_te, tfidf_te)

    # Ensure exact feature column alignment between train and test
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = train_meta['label'].values
    train_filenames = train_meta['filename']
    test_filenames = test_meta['filename']

    return X_train, y_train, train_filenames, X_test, test_filenames


def build_pipeline(model=None):
    """
    Construct a scikit-learn Pipeline with SimpleImputer, StandardScaler, and model.
    """
    if model is None:
        model = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42)
        
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])


def evaluate_model(pipeline, X_train, y_train, X_val, y_val):
    """
    Fit pipeline on training fold/split and evaluate on train and validation splits.
    Returns dictionary with compulsory Training RMSE, Validation RMSE, and Pearson correlation.
    """
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)

    train_rmse = float(np.sqrt(np.mean((y_train - y_train_pred) ** 2)))
    val_rmse = float(np.sqrt(np.mean((y_val - y_val_pred) ** 2)))

    # Safe Pearson calculation
    if np.std(y_val_pred) > 1e-6:
        val_pearson_r, _ = pearsonr(y_val, y_val_pred)
        val_pearson_r = float(val_pearson_r)
    else:
        val_pearson_r = 0.0

    return {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'val_pearson_r': val_pearson_r,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred
    }


def cross_validate_model(pipeline_factory, X, y, n_splits=5, random_state=42):
    """
    Perform 5-Fold Cross Validation and return out-of-fold predictions and metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_rmses = []
    
    for tr_idx, va_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        
        pipe = pipeline_factory()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        oof_preds[va_idx] = preds
        
        f_rmse = np.sqrt(np.mean((y_va - preds) ** 2))
        fold_rmses.append(f_rmse)
        
    cv_rmse = float(np.sqrt(np.mean((y - oof_preds) ** 2)))
    cv_r, _ = pearsonr(y, oof_preds)
    
    return {
        'cv_rmse': cv_rmse,
        'cv_rmse_std': float(np.std(fold_rmses)),
        'cv_pearson_r': float(cv_r),
        'oof_preds': oof_preds
    }


def save_model_artifact(model_or_pipeline, filepath=r'd:\AI-Grammar-Scoring\models\best_model.joblib'):
    """
    Save fitted model or pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_or_pipeline, filepath)
    print(f"Saved model artifact to: {filepath}")
