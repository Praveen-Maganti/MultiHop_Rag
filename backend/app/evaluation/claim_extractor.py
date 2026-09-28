import re
from typing import List

def extract_claims_rule_based(text: str) -> List[str]:
    """
    Extract atomic claims from structured Key:Value text.
    """
    if not text:
        return []
        
    # Remove memory references
    clean_text = re.sub(
        r'\(mem_[a-zA-Z0-9_,\s]+\)',
        '',
        text
    ).strip()
    
    claims = []
    
    # Matches patterns like "Name:Praveen" or "Age: 21"
    kv_pattern = r'([A-Za-z]+)\s*:\s*([^:\n]+)(?=\s+[A-Za-z]+\s*:|$)'
    matches = re.findall(kv_pattern, clean_text)
    
    if matches:
        for k, v in matches:
            claims.append(f"{k.strip()}: {v.strip()}")
    else:
        # Fallback to sentence splitting if not KV formatted
        sentences = re.split(r'(?<=[.!?])\s+', clean_text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
        
    return claims
