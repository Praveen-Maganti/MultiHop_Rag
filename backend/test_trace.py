import json
from dotenv import load_dotenv
load_dotenv()
from app.rag.proposed_rag import run_proposed_rag
from app.tracing.reasoning_trace import ReasoningTrace
import traceback

from app.retrieval.vector_store import initialize_vector_store

def run_tests():
    print("--- TEST 1 & 2: Basic Trace Creation and JSON Serialization ---")
    try:
        mock_chunks = [
            {"chunk_id": "doc_b03364cc_0", "text": "Name:Praveen Age:21 University:VIT-AP Year:4th CGPA:9.44"}
        ]
        initialize_vector_store("test_doc_praveen", mock_chunks)
        
        print("\n--- RUNNING USER'S EXACT SCENARIO ---")
        result = run_proposed_rag("test_doc_praveen", "What is Praveen's CGPA?", max_hops=3)
        trace_json = json.dumps(result["reasoning_trace"], indent=2)
        print("SUCCESS! Trace output:\n", trace_json[:1000], "\n...[truncated]\n")
    except Exception as e:
        print("FAILED!")
        traceback.print_exc()

    print("--- TEST 8: _check_active after finalize_hop ---")
    try:
        trace = ReasoningTrace()
        trace.start_hop(1, "Test query")
        trace.finalize_hop()
        try:
            trace.record_timing(1.2)
            print("FAILED! Should have raised RuntimeError")
        except RuntimeError as e:
            print(f"SUCCESS! Caught expected error: {e}")
    except Exception as e:
        print("FAILED!")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
