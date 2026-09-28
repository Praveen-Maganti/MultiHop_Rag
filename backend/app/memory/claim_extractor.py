import re

def extract_claims(text: str) -> list:
    """
    Rule-based KV parser for structured text.
    Replaces the LLM extractor to reduce latency.
    """
    claims = []
    
    # Matches patterns like "Name:Praveen" or "Age: 21"
    kv_pattern = r'([A-Za-z]+)\s*:\s*([^:\n]+)(?=\s+[A-Za-z]+\s*:|$)'
    matches = re.findall(kv_pattern, text)
    
    if matches:
        name_val = "The entity"
        for k, v in matches:
            if k.lower() == "name":
                name_val = v.strip()
                break
                
        for k, v in matches:
            k = k.strip()
            v = v.strip()
            
            if k.lower() == "name":
                claims.append(f"{v}'s name is {v}.")
            elif k.lower() == "age":
                claims.append(f"{name_val} is {v} years old.")
            elif k.lower() == "university":
                claims.append(f"{name_val} attends {v} University.")
            elif k.lower() == "year":
                claims.append(f"{name_val} is a {v}-year student.")
            elif k.lower() == "cgpa":
                claims.append(f"{name_val} has a CGPA of {v}.")
            else:
                claims.append(f"{name_val}'s {k} is {v}.")
    else:
        # Fallback to sentence splitting if not KV formatted
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
        
    return claims[:5]
