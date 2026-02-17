import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import re

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}" if prefix else timestamp

def hash_content(content: str) -> str:
    """Generate hash for content"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction - can be enhanced with NLP
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON from text"""
    try:
        return json.loads(text)
    except:
        return None

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for display"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (can be enhanced)"""
    # Simple Jaccard similarity
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0