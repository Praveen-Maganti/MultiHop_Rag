import sqlite3
import pandas as pd

DB_PATH = "experiments.db"

def generate_report():
    conn = sqlite3.connect(DB_PATH)
    
    # Read all experiments
    try:
        df = pd.read_sql_query("SELECT * FROM experiments", conn)
    except Exception as e:
        print("No experiments found or DB error:", e)
        return
        
    if df.empty:
        print("No data available to generate report.")
        return
        
    # Generate aggregated metrics
    summary = df.groupby('system_type').agg({
        'evidence_coverage': 'mean',
        'unsupported_claim_rate': 'mean',
        'tokens': 'mean',
        'latency': lambda x: x.str.replace('s','').astype(float).mean(),
        'retrieval_calls': 'mean',
        'memory_items': 'mean'
    }).round(3)
    
    # Calculate Improvements
    if 'proposed' in summary.index and 'standard' in summary.index:
        std = summary.loc['standard']
        prop = summary.loc['proposed']
        
        coverage_inc = ((prop['evidence_coverage'] - std['evidence_coverage']) / std['evidence_coverage']) * 100 if std['evidence_coverage'] else 0
        unsupported_dec = ((std['unsupported_claim_rate'] - prop['unsupported_claim_rate']) / std['unsupported_claim_rate']) * 100 if std['unsupported_claim_rate'] else 0
        tokens_dec = ((std['tokens'] - prop['tokens']) / std['tokens']) * 100 if std['tokens'] else 0
        
        report_md = f"""# Research Experiment Report: Adaptive Memory RAG

## Aggregated Performance Summary
{summary.to_markdown()}

## Observed Improvements
- **Evidence Coverage:** {'+' if coverage_inc >= 0 else ''}{coverage_inc:.1f}%
- **Unsupported Claims:** {'-' if unsupported_dec >= 0 else '+'}{abs(unsupported_dec):.1f}%
- **Memory Token Reduction:** {'-' if tokens_dec >= 0 else '+'}{abs(tokens_dec):.1f}%

## Conclusion
The proposed Adaptive Memory system {'successfully' if unsupported_dec > 0 and tokens_dec > 0 else 'failed to'} demonstrate the hypothesized reduction in error propagation (unsupported claims) and memory noise (tokens) while {'improving' if coverage_inc > 0 else 'maintaining'} evidence coverage.
"""
        
        with open("research_report.md", "w") as f:
            f.write(report_md)
            
        print("Report successfully generated: research_report.md")
    else:
        print("Need both 'standard' and 'proposed' data to generate comparison report.")

if __name__ == "__main__":
    generate_report()
