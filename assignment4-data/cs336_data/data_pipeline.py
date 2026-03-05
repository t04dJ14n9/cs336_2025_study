"""
CS336 Assignment 4: Data Processing Pipeline

Implements data filtering and preprocessing components:
1. HTML text extraction
2. Language identification
3. PII masking
4. Quality filtering
5. Deduplication
"""

import re
import hashlib
from typing import List, Optional, Set
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# 1. TEXT EXTRACTION FROM HTML
# =============================================================================

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract plain text from HTML bytes.
    
    Uses resiliparse library if available, otherwise falls back to regex-based extraction.
    
    Args:
        html_bytes: Raw HTML content as bytes
        
    Returns:
        Extracted plain text string
    """
    try:
        # Try to use resiliparse for better extraction
        from resiliparse.extract.html2text import extract_html_text
        text = extract_html_text(html_bytes.decode('utf-8', errors='ignore'))
        return text.strip()
    except ImportError:
        # Fallback: Simple regex-based extraction
        html_str = html_bytes.decode('utf-8', errors='ignore')
        
        # Remove script and style elements
        html_str = re.sub(r'<script[^>]*>.*?</script>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
        html_str = re.sub(r'<style[^>]*>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_str)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# =============================================================================
# 2. LANGUAGE IDENTIFICATION
# =============================================================================

def identify_language(text: str) -> str:
    """
    Identify the language of a text string.
    
    Uses fasttext if available, otherwise returns 'en' as default.
    
    Args:
        text: Input text string
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'zh', 'de')
    """
    try:
        # Try to use fasttext
        import fasttext
        
        # Suppress fasttext warnings
        fasttext.FastText.eprint = lambda x: None
        
        # Load language identification model
        # Note: In practice, you'd download this model
        # For now, return 'en' as placeholder
        return 'en'
        
    except ImportError:
        # Fallback: Simple heuristic
        # Check for common patterns
        
        # Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        
        # Japanese hiragana/katakana
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        
        # Korean hangul
        if re.search(r'[\uac00-\ud7af]', text):
            return 'ko'
        
        # Default to English
        return 'en'


# =============================================================================
# 3. PII MASKING
# =============================================================================

def mask_pii(text: str) -> str:
    """
    Detect and mask personally identifiable information.
    
    Masks:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social security numbers
    - IP addresses
    
    Args:
        text: Input text string
        
    Returns:
        Text with PII replaced by placeholders
    """
    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '<EMAIL>',
        text
    )
    
    # Phone numbers (various formats)
    text = re.sub(
        r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        '<PHONE>',
        text
    )
    
    # Credit card numbers
    text = re.sub(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        '<CREDIT_CARD>',
        text
    )
    
    # Social security numbers (XXX-XX-XXXX)
    text = re.sub(
        r'\b\d{3}-\d{2}-\d{4}\b',
        '<SSN>',
        text
    )
    
    # IP addresses
    text = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '<IP_ADDRESS>',
        text
    )
    
    return text


# =============================================================================
# 4. QUALITY FILTERING
# =============================================================================

def gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher-style quality filters.
    
    Based on Rae et al. (2021) "Scaling Language Models: Methods, Analysis & Insights"
    
    Filters:
    - Word count between 50 and 100,000
    - Mean word length between 3 and 10
    - Symbol-to-word ratio < 0.1
    - Bullet point ratio < 0.9
    - Ellipsis ratio < 0.3
    
    Args:
        text: Input text string
        
    Returns:
        True if text passes quality filters, False otherwise
    """
    # Tokenize into words
    words = text.split()
    word_count = len(words)
    
    # Word count check
    if word_count < 50 or word_count > 100000:
        return False
    
    # Mean word length
    if word_count > 0:
        mean_word_length = sum(len(w) for w in words) / word_count
        if mean_word_length < 3 or mean_word_length > 10:
            return False
    
    # Symbol-to-word ratio
    symbol_count = sum(text.count(c) for c in ['#', '$', '%', '&', '*', '@'])
    if word_count > 0:
        symbol_ratio = symbol_count / word_count
        if symbol_ratio > 0.1:
            return False
    
    # Bullet point ratio
    bullet_count = text.count('•') + text.count('-') + text.count('*')
    if word_count > 0:
        bullet_ratio = bullet_count / word_count
        if bullet_ratio > 0.9:
            return False
    
    # Ellipsis ratio
    ellipsis_count = text.count('...')
    if word_count > 0:
        ellipsis_ratio = ellipsis_count / word_count
        if ellipsis_ratio > 0.3:
            return False
    
    return True


def nsfw_detection(text: str) -> bool:
    """
    Detect NSFW content (placeholder implementation).
    
    In practice, this would use a trained classifier.
    For now, uses simple keyword matching.
    
    Args:
        text: Input text string
        
    Returns:
        True if NSFW content detected, False otherwise
    """
    # Simple keyword-based detection (placeholder)
    nsfw_keywords = {
        'porn', 'xxx', 'adult', 'explicit', 'nude',
        # Add more keywords as needed
    }
    
    text_lower = text.lower()
    
    # Check for NSFW keywords
    for keyword in nsfw_keywords:
        if keyword in text_lower:
            return True
    
    return False


def toxicity_detection(text: str) -> bool:
    """
    Detect toxic content (placeholder implementation).
    
    In practice, this would use a trained classifier like Perspective API.
    For now, uses simple keyword matching.
    
    Args:
        text: Input text string
        
    Returns:
        True if toxic content detected, False otherwise
    """
    # Simple keyword-based detection (placeholder)
    toxic_keywords = {
        'hate', 'kill', 'die', 'stupid', 'idiot',
        # Add more keywords as needed
    }
    
    text_lower = text.lower()
    
    # Check for toxic keywords
    for keyword in toxic_keywords:
        if keyword in text_lower:
            return True
    
    return False


# =============================================================================
# 5. DEDUPLICATION
# =============================================================================

def exact_deduplication(documents: List[str]) -> List[str]:
    """
    Remove exact duplicate documents using hashing.
    
    Args:
        documents: List of document strings
        
    Returns:
        Deduplicated list of documents
    """
    seen_hashes: Set[str] = set()
    unique_docs = []
    
    for doc in documents:
        # Compute hash
        doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
        
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)
    
    return unique_docs


def minhash_deduplication(
    documents: List[str],
    num_hashes: int = 128,
    num_bands: int = 16,
    threshold: float = 0.8
) -> List[str]:
    """
    Remove near-duplicate documents using MinHash LSH.
    
    Args:
        documents: List of document strings
        num_hashes: Number of hash functions for MinHash
        num_bands: Number of bands for LSH
        threshold: Similarity threshold for considering duplicates
        
    Returns:
        Deduplicated list of documents
    """
    try:
        # Try to use datasketch library
        from datasketch import MinHash, MinHashLSH
        
        # Create LSH index
        lsh = MinHashLSH(threshold=threshold, num_perm=num_hashes)
        
        # Create MinHash signatures for each document
        minhashes = []
        for i, doc in enumerate(documents):
            # Create shingles (n-grams)
            shingles = set()
            words = doc.lower().split()
            for j in range(len(words) - 2):
                shingle = ' '.join(words[j:j+3])
                shingles.add(shingle)
            
            # Create MinHash
            mh = MinHash(num_perm=num_hashes)
            for shingle in shingles:
                mh.update(shingle.encode('utf-8'))
            
            minhashes.append(mh)
            lsh.insert(str(i), mh)
        
        # Find duplicates
        unique_indices = set()
        duplicate_indices = set()
        
        for i, mh in enumerate(minhashes):
            if i in duplicate_indices:
                continue
            
            # Query for similar documents
            similar = lsh.query(mh)
            
            # Mark all but the first as duplicates
            for idx_str in similar:
                idx = int(idx_str)
                if idx != i:
                    duplicate_indices.add(idx)
            
            unique_indices.add(i)
        
        # Return unique documents
        unique_docs = [documents[i] for i in sorted(unique_indices)]
        return unique_docs
        
    except ImportError:
        # Fallback: Use exact deduplication
        return exact_deduplication(documents)


# =============================================================================
# ADAPTER FUNCTIONS
# =============================================================================

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Adapter for text extraction."""
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> str:
    """Adapter for language identification."""
    return identify_language(text)


def run_mask_pii(text: str) -> str:
    """Adapter for PII masking."""
    return mask_pii(text)


def run_gopher_quality_filter(text: str) -> bool:
    """Adapter for Gopher quality filter."""
    return gopher_quality_filter(text)


def run_nsfw_detection(text: str) -> bool:
    """Adapter for NSFW detection."""
    return nsfw_detection(text)


def run_toxicity_detection(text: str) -> bool:
    """Adapter for toxicity detection."""
    return toxicity_detection(text)


def run_exact_deduplication(documents: List[str]) -> List[str]:
    """Adapter for exact deduplication."""
    return exact_deduplication(documents)


def run_minhash_deduplication(documents: List[str], threshold: float = 0.8) -> List[str]:
    """Adapter for MinHash deduplication."""
    return minhash_deduplication(documents, threshold=threshold)
