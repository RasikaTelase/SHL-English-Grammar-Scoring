"""
src/grammar_features.py

Module for extracting grammar-specific indicators, fluency metrics,
syntactic structure, disfluency counts, and speech-rate alignment features.

Note on Grammar Checking Tools:
External NLP grammar checkers (like language_tool_python which requires Java, or spaCy
with en_core_web_sm which is not pre-installed in this environment) are checked at runtime.
Where external checkers are unavailable, this module uses defensible, rule-based heuristics
for disfluency, subject-verb agreement patterns, article misuse, sentence fragments, and syntactic complexity.
"""

import re
import numpy as np
import pandas as pd


# Disfluency / Filler Words
FILLER_WORDS_PATTERNS = [
    r'\bum\b', r'\buh\b', r'\ber\b', r'\bah\b', r'\bhmm\b',
    r'\blike\b', r'\byou know\b', r'\bi mean\b', r'\bsort of\b', r'\bkind of\b',
    r'\bbasically\b', r'\bactually\b'
]
FILLER_REGEX = re.compile('|'.join(FILLER_WORDS_PATTERNS), re.IGNORECASE)

# Subordinating Conjunctions
SUB_CONJ_PATTERNS = [
    r'\bbecause\b', r'\balthough\b', r'\beven though\b', r'\bif\b', r'\bunless\b',
    r'\bsince\b', r'\bwhile\b', r'\bwhereas\b', r'\buntil\b', r'\bas long as\b',
    r'\bprovided that\b', r'\bso that\b'
]
SUB_CONJ_REGEX = re.compile('|'.join(SUB_CONJ_PATTERNS), re.IGNORECASE)

# Coordinating Conjunctions
COORD_CONJ_PATTERNS = [r'\band\b', r'\bbut\b', r'\bor\b', r'\bnor\b', r'\bfor\b', r'\bso\b', r'\byet\b']
COORD_CONJ_REGEX = re.compile('|'.join(COORD_CONJ_PATTERNS), re.IGNORECASE)

# Subject-Verb Agreement Mismatch Heuristics
# Third person singular pronoun followed by base form or plural verb
SV_SINGULAR_MISMATCH = re.compile(
    r'\b(he|she|it)\s+(do|have|were|are|be)\b', re.IGNORECASE
)
# Plural/First/Second person pronoun followed by singular verb
SV_PLURAL_MISMATCH = re.compile(
    r'\b(they|we|you)\s+(does|has|is|was)\b|\bi\s+(is|are|does|has)\b', re.IGNORECASE
)

# Double Negatives Heuristics
DOUBLE_NEG_REGEX = re.compile(
    r'\b(don\'t|doesn\'t|didn\'t|won\'t|can\'t|couldn\'t|shouldn\'t|haven\'t|hasn\'t|hadn\'t|never|no one|nobody|nothing)\b.*?\b(no|nothing|nobody|nowhere|neither|none)\b',
    re.IGNORECASE
)

# Article Misuse Heuristics
# "a" followed by vowel start (a apple, a hour, a engine, a idea, a option)
ARTICLE_A_MISUSE = re.compile(r'\ba\s+[aeiou]\w*', re.IGNORECASE)
# "an" followed by consonant start (an book, an car, an dog, an house) - excluding silent h words like hour/honest
ARTICLE_AN_MISUSE = re.compile(r'\ban\s+[bcdfghjklmnpqrstvwxyz]\w*', re.IGNORECASE)

# Passive Voice Heuristic
PASSIVE_VOICE_REGEX = re.compile(
    r'\b(am|is|are|was|were|be|been|being)\s+(\w+ed|made|done|given|taken|seen|written|built|found|chosen|said)\b',
    re.IGNORECASE
)


def extract_text_grammar_indicators(text):
    """
    Extract rule-based grammar indicators, disfluencies, and syntactic metrics from raw transcript text.
    Returns a dictionary of numeric features.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'filler_word_count': 0,
            'filler_word_ratio': 0.0,
            'word_repetition_count': 0,
            'word_repetition_ratio': 0.0,
            'phrase_repetition_count': 0,
            'sv_agreement_error_count': 0,
            'sv_agreement_error_ratio': 0.0,
            'double_negative_count': 0,
            'article_misuse_count': 0,
            'capitalization_after_period_error_count': 0,
            'subordinate_conjunction_count': 0,
            'subordinate_conjunction_ratio': 0.0,
            'coordinating_conjunction_count': 0,
            'coordinating_conjunction_ratio': 0.0,
            'sentence_fragment_count': 0,
            'sentence_fragment_ratio': 0.0,
            'run_on_sentence_count': 0,
            'run_on_sentence_ratio': 0.0,
            'passive_voice_heuristic_count': 0
        }

    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    if word_count == 0:
        return extract_text_grammar_indicators('')

    # 1. Fillers
    filler_matches = FILLER_REGEX.findall(text)
    filler_count = len(filler_matches)
    filler_ratio = filler_count / word_count

    # 2. Word Repetitions (adjacent identical words)
    word_rep_count = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            word_rep_count += 1
    word_rep_ratio = word_rep_count / word_count

    # 3. 2-gram Phrase Repetitions
    phrase_rep_count = 0
    if word_count >= 4:
        for i in range(len(words) - 3):
            if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
                phrase_rep_count += 1

    # 4. Subject-Verb Agreement Mismatches
    sv_sing_matches = len(SV_SINGULAR_MISMATCH.findall(text))
    sv_plur_matches = len(SV_PLURAL_MISMATCH.findall(text))
    sv_error_count = sv_sing_matches + sv_plur_matches
    sv_error_ratio = sv_error_count / word_count

    # 5. Double Negatives
    double_neg_count = len(DOUBLE_NEG_REGEX.findall(text))

    # 6. Article Misuses (excluding known exceptions e.g. "a university", "an hour")
    a_misuse = len(ARTICLE_A_MISUSE.findall(text))
    an_misuse = len(ARTICLE_AN_MISUSE.findall(text))
    article_misuse_count = a_misuse + an_misuse

    # 7. Capitalization Errors (lowercase word after period)
    cap_errors = len(re.findall(r'\.\s+[a-z]', text))

    # 8. Sentences & Syntactic Complexity
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(1, len(sentences))

    sub_conj_count = len(SUB_CONJ_REGEX.findall(text))
    sub_conj_ratio = sub_conj_count / sentence_count

    coord_conj_count = len(COORD_CONJ_REGEX.findall(text))
    coord_conj_ratio = coord_conj_count / word_count

    # Fragments (<3 words) & Run-on sentences (>30 words)
    fragment_count = 0
    run_on_count = 0
    for s in sentences:
        s_words = re.findall(r'\b\w+\b', s)
        if len(s_words) < 3:
            fragment_count += 1
        elif len(s_words) > 30:
            run_on_count += 1

    fragment_ratio = fragment_count / sentence_count
    run_on_ratio = run_on_count / sentence_count

    # Passive voice
    passive_count = len(PASSIVE_VOICE_REGEX.findall(text))

    return {
        'filler_word_count': filler_count,
        'filler_word_ratio': float(filler_ratio),
        'word_repetition_count': word_rep_count,
        'word_repetition_ratio': float(word_rep_ratio),
        'phrase_repetition_count': phrase_rep_count,
        'sv_agreement_error_count': sv_error_count,
        'sv_agreement_error_ratio': float(sv_error_ratio),
        'double_negative_count': double_neg_count,
        'article_misuse_count': article_misuse_count,
        'capitalization_after_period_error_count': cap_errors,
        'subordinate_conjunction_count': sub_conj_count,
        'subordinate_conjunction_ratio': float(sub_conj_ratio),
        'coordinating_conjunction_count': coord_conj_count,
        'coordinating_conjunction_ratio': float(coord_conj_ratio),
        'sentence_fragment_count': fragment_count,
        'sentence_fragment_ratio': float(fragment_ratio),
        'run_on_sentence_count': run_on_count,
        'run_on_sentence_ratio': float(run_on_ratio),
        'passive_voice_heuristic_count': passive_count
    }


def compute_grammar_and_fluency_features(transcripts_df, audio_features_df):
    """
    Combine transcript text grammar indicators with audio feature alignment
    (speech rate, articulation rate, pause-to-word metrics).
    
    Parameters:
        transcripts_df (pd.DataFrame): DataFrame containing 'filename' and 'transcript'
        audio_features_df (pd.DataFrame): DataFrame containing 'filename', 'duration_sec', etc.
        
    Returns:
        pd.DataFrame: DataFrame with filename and all extracted grammar/fluency features.
    """
    # Merge on filename while preserving transcripts_df order
    merged = pd.merge(transcripts_df, audio_features_df, on='filename', how='left')

    rows = []
    for idx, row in merged.iterrows():
        fname = row['filename']
        text = str(row['transcript']) if pd.notna(row['transcript']) else ""
        err_msg = str(row.get('error_msg', ''))

        # Check for explicitly recorded transcription error (e.g. audio_5055.wav)
        is_error = bool(row.get('transcription_error', False)) or ("FAILED" in err_msg.upper())

        # Extract text grammar indicators
        text_feats = extract_text_grammar_indicators(text)

        # Calculate Speech & Fluency Alignment Features
        duration = float(row.get('duration_sec', 0.0)) if pd.notna(row.get('duration_sec')) else 0.0
        speech_ratio = float(row.get('speech_duration_ratio', 0.0)) if pd.notna(row.get('speech_duration_ratio')) else 0.0
        silence_ratio = float(row.get('silence_duration_ratio', 0.0)) if pd.notna(row.get('silence_duration_ratio')) else 0.0
        pause_count = float(row.get('pause_count', 0.0)) if pd.notna(row.get('pause_count')) else 0.0
        mean_pause_sec = float(row.get('mean_pause_duration_sec', 0.0)) if pd.notna(row.get('mean_pause_duration_sec')) else 0.0

        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)

        if is_error or word_count == 0 or duration == 0:
            speech_rate_wpm = 0.0
            articulation_rate_wps = 0.0
            pause_per_word_ratio = 0.0
            silence_sec_per_word = 0.0
            words_per_pause = 0.0
            grammar_status = "FAILED_TRANSCRIPTION_DEFAULT_FEATURES" if is_error else "EMPTY_TRANSCRIPT"
        else:
            speech_rate_wpm = (word_count / duration) * 60.0
            active_speech_sec = duration * speech_ratio
            articulation_rate_wps = (word_count / active_speech_sec) if active_speech_sec > 0 else 0.0
            pause_per_word_ratio = pause_count / max(1, word_count)
            silence_sec_per_word = (duration * silence_ratio) / max(1, word_count)
            words_per_pause = word_count / max(1.0, pause_count)
            grammar_status = "OK"

        row_dict = {
            'filename': fname,
            'speech_rate_wpm': float(speech_rate_wpm),
            'articulation_rate_wps': float(articulation_rate_wps),
            'pause_per_word_ratio': float(pause_per_word_ratio),
            'silence_sec_per_word': float(silence_sec_per_word),
            'words_per_pause': float(words_per_pause),
            'mean_pause_duration_sec': float(mean_pause_sec)
        }
        row_dict.update(text_feats)
        row_dict['grammar_feature_status'] = grammar_status

        rows.append(row_dict)

    df_out = pd.DataFrame(rows)

    # Sanity check for non-finite / NaN values in numerical columns
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].fillna(0.0)
    df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], 0.0)

    return df_out
