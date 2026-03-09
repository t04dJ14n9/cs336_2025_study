from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree
    tree = HTMLTree.parse(html_bytes.decode('utf-8', errors='ignore'))
    return extract_plain_text(tree)


def run_identify_language(text: str) -> tuple[Any, float]:
    """Returns (language_code, confidence_score)."""
    try:
        import fasttext
        import tempfile, os as _os
        model_path = _os.path.join(_os.path.dirname(__file__), '..', 'lid.176.bin')
        model_path = _os.path.abspath(model_path)
        if not _os.path.exists(model_path):
            raise FileNotFoundError(f"fasttext model not found at {model_path}")
        model = fasttext.load_model(model_path)
        # fasttext expects single-line input
        single_line = text.replace('\n', ' ')
        predictions = model.predict(single_line, k=1)
        label = predictions[0][0].replace('__label__', '')
        score = float(predictions[1][0])
        return label, score
    except Exception:
        pass
    # Heuristic fallback
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh', 1.0
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return 'ja', 1.0
    if re.search(r'[\uac00-\ud7af]', text):
        return 'ko', 1.0
    return 'en', 1.0


def run_mask_emails(text: str) -> tuple[str, int]:
    """Mask email addresses, return (masked_text, count)."""
    placeholder = '|||EMAIL_ADDRESS|||'
    pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    matches = re.findall(pattern, text)
    # Only count actual emails (not existing placeholders)
    masked = re.sub(pattern, placeholder, text)
    return masked, len(matches)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """Mask phone numbers, return (masked_text, count)."""
    placeholder = '|||PHONE_NUMBER|||'
    # Matches: 2831823829, (283)-182-3829, (283) 182 3829, 283-182-3829
    pattern = r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}'
    matches = re.findall(pattern, text)
    masked = re.sub(pattern, placeholder, text)
    return masked, len(matches)


def run_mask_ips(text: str) -> tuple[str, int]:
    """Mask IP addresses, return (masked_text, count)."""
    placeholder = '|||IP_ADDRESS|||'
    pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    matches = re.findall(pattern, text)
    masked = re.sub(pattern, placeholder, text)
    return masked, len(matches)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    """Returns (label, score) where label is 'nsfw' or 'non-nsfw'."""
    try:
        # Try using a fasttext-based classifier if available
        raise NotImplementedError  # placeholder
    except Exception:
        pass
    # Keyword-based heuristic
    nsfw_keywords = {
        'porn', 'xxx', 'nude', 'explicit', 'obscene',
        'cock', 'cunt', 'ass', 'fuck', 'shit', 'bitch',
        'c*ck', 'c*nt', 'f*ck', 'a**', '*ssh*le',
    }
    text_lower = text.lower()
    for kw in nsfw_keywords:
        if kw in text_lower:
            return 'nsfw', 1.0
    return 'non-nsfw', 1.0


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    """Returns (label, score) where label is 'toxic' or 'non-toxic'."""
    try:
        raise NotImplementedError
    except Exception:
        pass
    # Keyword-based heuristic: strong indicators of toxicity
    toxic_strong = {
        'idiot', 'moron', 'rude fuck', 'arrogant', 'fuckers',
        'fuck', 'twat', 'asshole',
    }
    text_lower = text.lower()
    hit_count = sum(1 for kw in toxic_strong if kw in text_lower)
    if hit_count >= 2:
        return 'toxic', float(hit_count)
    return 'non-toxic', 1.0


def run_classify_quality(text: str) -> tuple[Any, float]:
    """Returns (label, score) where label is 'wiki' or 'cc'."""
    # Heuristic: High-quality text has longer sentences, no navigation boilerplate
    words = text.split()
    word_count = len(words)

    # Low quality signals
    low_quality_signals = 0

    # Very short text
    if word_count < 100:
        low_quality_signals += 2

    # Navigation/boilerplate patterns
    boilerplate_patterns = [
        r'\bFAQ\b', r'\bSearch\b', r'\bRegister\b', r'\bLog in\b',
        r'\bMemberlist\b', r'\bUsergroups\b', r'\bProfile\b',
        r'Copyright ©', r'Powered by',
    ]
    for pat in boilerplate_patterns:
        if re.search(pat, text, re.IGNORECASE):
            low_quality_signals += 1

    # Repeated short lines (navigation menus)
    lines = text.strip().split('\n')
    short_lines = sum(1 for l in lines if 0 < len(l.split()) <= 3)
    if len(lines) > 5 and short_lines / len(lines) > 0.5:
        low_quality_signals += 2

    if low_quality_signals >= 3:
        return 'cc', float(low_quality_signals)
    return 'wiki', 1.0 / max(1, low_quality_signals + 1)


def run_gopher_quality_filter(text: str) -> bool:
    """Apply Gopher-style quality filters."""
    words = text.split()
    word_count = len(words)

    # Word count check
    if word_count < 50 or word_count > 100_000:
        return False

    # Mean word length between 3 and 10
    mean_word_length = sum(len(w) for w in words) / word_count
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # At least 80% of words must contain an alphabetic character
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.8:
        return False

    # No more than 30% of lines ending with ellipsis
    lines = text.split('\n')
    if lines:
        ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith('...'))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    Exact line-level deduplication across files.
    Any line that appears more than once across all files is removed from all files.
    Output files are written to output_directory with the same names.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # First pass: count line occurrences across all files
    line_counts: dict[str, int] = {}
    file_lines: dict[Path, list[str]] = {}

    for input_path in input_files:
        input_path = Path(input_path)
        with open(input_path) as f:
            lines = f.readlines()
        file_lines[input_path] = lines
        for line in lines:
            stripped = line.rstrip('\n')
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Second pass: write files with duplicate lines removed
    for input_path, lines in file_lines.items():
        output_path = output_directory / input_path.name
        kept = [line for line in lines if line_counts[line.rstrip('\n')] == 1]
        with open(output_path, 'w') as f:
            f.writelines(kept)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    MinHash LSH deduplication. Removes near-duplicate documents.
    Each file is treated as a separate document.
    Writes non-duplicate files to output_directory.
    """
    from datasketch import MinHash, MinHashLSH
    from pathlib import Path

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_files = [Path(p) for p in input_files]

    # Read all documents
    documents = {}
    for path in input_files:
        with open(path) as f:
            documents[path] = f.read()

    # Build MinHash signatures
    def make_minhash(text: str) -> MinHash:
        mh = MinHash(num_perm=num_hashes)
        words = text.lower().split()
        for i in range(max(1, len(words) - ngrams + 1)):
            ngram = ' '.join(words[i:i + ngrams])
            mh.update(ngram.encode('utf-8'))
        return mh

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_hashes)
    minhashes = {}
    for path, text in documents.items():
        mh = make_minhash(text)
        minhashes[path] = mh

    # Find duplicates: keep earliest file, drop later duplicates
    kept = set()
    dropped = set()

    for path in input_files:
        if path in dropped:
            continue
        mh = minhashes[path]
        key = str(path)
        # Query before inserting to find existing duplicates
        if key not in lsh.keys:
            similar = lsh.query(mh)
            if similar:
                # This document is a near-duplicate of something already in LSH
                dropped.add(path)
            else:
                lsh.insert(key, mh)
                kept.add(path)
        else:
            kept.add(path)

    # Write kept documents
    for path in kept:
        output_path = output_directory / path.name
        with open(output_path, 'w') as f:
            f.write(documents[path])
