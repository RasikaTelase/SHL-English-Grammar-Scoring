"""Text and linguistic feature extraction module for SHL English Grammar Scoring Engine.

Extracts text length metrics, lexical diversity (TTR, Hapax Legomena),
part-of-speech category ratios, punctuation statistics, readability indices,
and TF-IDF n-gram feature representations.
"""

import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define lexical category word sets
ARTICLES = {'a', 'an', 'the'}
PREPOSITIONS = {
    'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'from', 'about',
    'into', 'through', 'over', 'under', 'between', 'after', 'before', 'during', 'without'
}
CONJUNCTIONS = {
    'and', 'but', 'or', 'so', 'because', 'if', 'when', 'while', 'although',
    'since', 'as', 'than', 'unless', 'whereas'
}
PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those'
}
AUXILIARY_VERBS = {
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'
}
FILLER_WORDS = {
    'um', 'uh', 'like', 'so', 'well', 'actually', 'basically', 'honestly', 'y know', 'ya', 'yeah', 'right'
}


def _estimate_syllables_word(word: str) -> int:
    """Estimate syllable count of a single word using vowel cluster heuristics."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)


def extract_single_text_features(text: str, filename: str = '') -> dict:
    """Extract a dictionary of numerical text features from a single transcript text.
    
    Args:
        text (str): Raw text transcript string.
        filename (str): Optional audio filename reference.
        
    Returns:
        dict: Extracted linguistic features.
    """
    feats = {
        'filename': filename,
        'is_empty_transcript': False,
        'has_text_error': False
    }
    
    # Handle NaN, None, or empty text gracefully
    if pd.isna(text) or text is None:
        text = ''
    text_str = str(text).strip()
    
    if len(text_str) == 0:
        feats['is_empty_transcript'] = True
        feats['has_text_error'] = True
        return _populate_fallback_text_features(feats)
        
    # Extract words (case-insensitive) and sentences
    words = re.findall(r'\b\w+\b', text_str.lower())
    raw_words = re.findall(r'\b\w+\b', text_str)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text_str) if s.strip()]
    
    num_words = len(words)
    num_chars = len(text_str)
    num_sentences = max(1, len(sentences))
    
    if num_words == 0:
        feats['is_empty_transcript'] = True
        feats['has_text_error'] = True
        return _populate_fallback_text_features(feats)
        
    # 1. Length & Structural Features
    feats['word_count'] = int(num_words)
    feats['char_count'] = int(num_chars)
    feats['avg_word_length'] = float(num_chars / num_words)
    feats['sentence_count'] = int(num_sentences)
    
    sentence_word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    feats['avg_sentence_length_words'] = float(np.mean(sentence_word_counts)) if len(sentence_word_counts) > 0 else float(num_words)
    feats['sentence_length_std_words'] = float(np.std(sentence_word_counts)) if len(sentence_word_counts) > 0 else 0.0
    
    # 2. Lexical Diversity & Repetition Features
    word_freq = pd.Series(words).value_counts()
    num_unique = len(word_freq)
    
    feats['unique_word_count'] = int(num_unique)
    feats['unique_word_ratio'] = float(num_unique / num_words)  # Type Token Ratio (TTR)
    feats['repeated_word_ratio'] = float(1.0 - (num_unique / num_words))
    
    hapax_legomena = (word_freq == 1).sum()
    feats['hapax_legomena_count'] = int(hapax_legomena)
    feats['hapax_legomena_ratio'] = float(hapax_legomena / num_words)
    
    max_rep = int(word_freq.max()) if len(word_freq) > 0 else 0
    feats['max_word_repetition_count'] = max_rep
    feats['max_word_repetition_ratio'] = float(max_rep / num_words)
    
    # 3. Part-of-Speech & Word Category Ratios
    art_count = sum(1 for w in words if w in ARTICLES)
    prep_count = sum(1 for w in words if w in PREPOSITIONS)
    conj_count = sum(1 for w in words if w in CONJUNCTIONS)
    pron_count = sum(1 for w in words if w in PRONOUNS)
    aux_count = sum(1 for w in words if w in AUXILIARY_VERBS)
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    
    feats['article_count'] = int(art_count)
    feats['article_ratio'] = float(art_count / num_words)
    
    feats['preposition_count'] = int(prep_count)
    feats['preposition_ratio'] = float(prep_count / num_words)
    
    feats['conjunction_count'] = int(conj_count)
    feats['conjunction_ratio'] = float(conj_count / num_words)
    
    feats['pronoun_count'] = int(pron_count)
    feats['pronoun_ratio'] = float(pron_count / num_words)
    
    feats['auxiliary_verb_count'] = int(aux_count)
    feats['auxiliary_verb_ratio'] = float(aux_count / num_words)
    
    feats['filler_word_count'] = int(filler_count)
    feats['filler_word_ratio'] = float(filler_count / num_words)
    
    # 4. Punctuation & Capitalization Statistics
    comma_count = text_str.count(',')
    period_count = text_str.count('.')
    question_count = text_str.count('?')
    exclamation_count = text_str.count('!')
    capital_count = sum(1 for c in text_str if c.isupper())
    
    feats['comma_count'] = int(comma_count)
    feats['comma_ratio'] = float(comma_count / num_words)
    feats['period_count'] = int(period_count)
    feats['period_ratio'] = float(period_count / num_words)
    feats['question_count'] = int(question_count)
    feats['exclamation_count'] = int(exclamation_count)
    feats['capital_letter_count'] = int(capital_count)
    feats['capital_letter_ratio'] = float(capital_count / max(1, num_chars))
    
    # 5. Readability & Complexity Indices
    total_syllables = sum(_estimate_syllables_word(w) for w in words)
    feats['syllable_count'] = int(total_syllables)
    feats['avg_syllables_per_word'] = float(total_syllables / num_words)
    
    # Automated Readability Index (ARI)
    ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
    feats['automated_readability_index'] = float(ari)
    
    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
    feats['flesch_reading_ease'] = float(flesch)
    
    # Coleman-Liau Index
    L = (num_chars / num_words) * 100.0
    S = (num_sentences / num_words) * 100.0
    coleman_liau = 0.0588 * L - 0.296 * S - 15.8
    feats['coleman_liau_index'] = float(coleman_liau)
    
    # Clean non-finite values
    for k, v in feats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feats[k] = 0.0
            
    return feats


def _populate_fallback_text_features(feats: dict) -> dict:
    """Populate default zero/null values for text feature dictionary on error."""
    default_keys = [
        'word_count', 'char_count', 'avg_word_length', 'sentence_count',
        'avg_sentence_length_words', 'sentence_length_std_words',
        'unique_word_count', 'unique_word_ratio', 'repeated_word_ratio',
        'hapax_legomena_count', 'hapax_legomena_ratio', 'max_word_repetition_count',
        'max_word_repetition_ratio', 'article_count', 'article_ratio',
        'preposition_count', 'preposition_ratio', 'conjunction_count', 'conjunction_ratio',
        'pronoun_count', 'pronoun_ratio', 'auxiliary_verb_count', 'auxiliary_verb_ratio',
        'filler_word_count', 'filler_word_ratio', 'comma_count', 'comma_ratio',
        'period_count', 'period_ratio', 'question_count', 'exclamation_count',
        'capital_letter_count', 'capital_letter_ratio', 'syllable_count',
        'avg_syllables_per_word', 'automated_readability_index', 'flesch_reading_ease',
        'coleman_liau_index'
    ]
    for k in default_keys:
        if k not in feats:
            feats[k] = 0.0
    return feats


def extract_text_features_dataset(transcripts_df: pd.DataFrame, 
                                 text_col: str = 'transcript', 
                                 filename_col: str = 'filename') -> pd.DataFrame:
    """Extract numerical text features for all samples in a transcripts dataframe.
    
    Args:
        transcripts_df (pd.DataFrame): Dataframe containing filename and transcript columns.
        text_col (str): Column name containing text transcripts.
        filename_col (str): Column name containing audio filenames.
        
    Returns:
        pd.DataFrame: Dataframe of extracted text features.
    """
    feature_rows = []
    
    for _, row in transcripts_df.iterrows():
        fname = row[filename_col]
        txt = row.get(text_col, '')
        
        feats = extract_single_text_features(txt, filename=fname)
        feature_rows.append(feats)
        
    features_df = pd.DataFrame(feature_rows)
    
    # Reorder columns so filename is first
    cols = [filename_col] + [c for c in features_df.columns if c != filename_col]
    features_df = features_df[cols]
    
    # Clean any remaining non-finite values
    num_cols = features_df.select_dtypes(include=['float64', 'int64', 'float32']).columns
    features_df[num_cols] = features_df[num_cols].fillna(0.0)
    for c in num_cols:
        features_df[c] = features_df[c].replace([np.inf, -np.inf], 0.0)
        
    return features_df


def fit_extract_tfidf_features(train_transcripts: list, 
                               test_transcripts: list, 
                               train_filenames: list, 
                               test_filenames: list, 
                               max_features: int = 100, 
                               vectorizer_save_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """Fit TF-IDF vectorizer strictly on training transcripts and transform train and test sets.
    
    Args:
        train_transcripts (list): List of training transcript strings.
        test_transcripts (list): List of test transcript strings.
        train_filenames (list): List of training filenames.
        test_filenames (list): List of test filenames.
        max_features (int): Maximum TF-IDF vocabulary features.
        vectorizer_save_path (str): Destination path to save fitted vectorizer.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]: Train TF-IDF DF, Test TF-IDF DF, fitted vectorizer.
    """
    clean_train_texts = [str(t) if pd.notna(t) else '' for t in train_transcripts]
    clean_test_texts = [str(t) if pd.notna(t) else '' for t in test_transcripts]
    
    print(f"[INFO] Fitting TfidfVectorizer (max_features={max_features}, ngram_range=(1,2)) strictly on training set...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    train_tfidf_matrix = vectorizer.fit_transform(clean_train_texts)
    test_tfidf_matrix = vectorizer.transform(clean_test_texts)
    
    feature_names = [f"tfidf_{name.replace(' ', '_')}" for name in vectorizer.get_feature_names_out()]
    
    train_tfidf_df = pd.DataFrame(train_tfidf_matrix.toarray(), columns=feature_names)
    train_tfidf_df.insert(0, 'filename', train_filenames)
    
    test_tfidf_df = pd.DataFrame(test_tfidf_matrix.toarray(), columns=feature_names)
    test_tfidf_df.insert(0, 'filename', test_filenames)
    
    if vectorizer_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(vectorizer_save_path)), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_save_path)
        print(f"[SAVED] Fitted TF-IDF vectorizer saved to {vectorizer_save_path}")
        
    return train_tfidf_df, test_tfidf_df, vectorizer


def save_text_features(features_df: pd.DataFrame, output_path: str) -> None:
    """Save extracted text features dataframe to CSV.
    
    Args:
        features_df (pd.DataFrame): Text features dataframe.
        output_path (str): Target CSV file destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"[SAVED] Text features saved to {output_path} (Shape: {features_df.shape})")
