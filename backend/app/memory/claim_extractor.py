import re

def extract_claims(text: str) -> list:
    """
    Extracts atomic factual sentences from chunk text.
    Uses sentence splitting as primary strategy — works for all document types.
    KV parsing is a secondary fallback only for strictly structured records.
    """
    if not text or len(text.strip()) < 10:
        return []

    # Primary: split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    # If we got good sentences, return top 5 by length (longer = more informative)
    if len(sentences) >= 2:
        sentences.sort(key=len, reverse=True)
        return sentences[:5]

    # Secondary fallback: KV pattern only for short structured records
    kv_pattern = r'([A-Za-z][A-Za-z ]{1,20})\s*:\s*([^:\n]{3,80})'
    matches = re.findall(kv_pattern, text)
    if matches:
        claims = [f"{k.strip()} is {v.strip()}." for k, v in matches]
        return claims[:5]

    # Last resort: return the whole text as one claim
    return [text.strip()[:300]]
