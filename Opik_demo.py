import streamlit as st
import re
import json
from typing import Any, Dict, List, Optional, Tuple # For type hinting
import traceback # For error logging

# --- Opik SDK Imports ---
try:
    from opik.evaluation.metrics import (
        Equals, Contains, RegexMatch, IsJson, Levenshtein,
        Hallucination, GEval, Moderation, AnswerRelevance, Usefulness,
        ContextRecall, ContextPrecision
    )
    # Opik's documentation also mentions SentenceBLEU, CorpusBLEU, ROUGE as heuristic
    # from opik.evaluation.metrics.heuristic_metrics import SentenceBLEU, CorpusBLEU, ROUGE_N # Example
    from opik.evaluation.metrics.base_metric import BaseMetric as OpikBaseMetric
    from opik.evaluation.metrics.score_result import ScoreResult as OpikScoreResult
    from opik.evaluation import models as opik_models # For LiteLLMChatModel
    OPIK_SDK_AVAILABLE = True
    # st.sidebar.success("Opik SDK loaded successfully!") # Moved to main app display to avoid error on first run
except ImportError as e:
    # st.sidebar.error(f"Opik SDK not found or import error: {e}. Please install opik (`pip install opik`). This demo will not run Opik metrics.")
    OPIK_SDK_AVAILABLE = False
    class OpikBaseMetric: # Dummy base
        def __init__(self, name: str = "DummyOpikBase"):
            self.name = name
    class OpikScoreResult: # Dummy result
        def __init__(self, name, value, reason=None, metadata=None):
            self.name = name
            self.value = value
            self.reason = reason
            self.metadata = metadata
        def to_dict(self):
            return {"name": self.name, "score": self.value, "reason": self.reason, "metadata": self.metadata}
    # Define dummy Opik metric classes
    class Equals:
        def score(self, **kwargs): return OpikScoreResult("Equals (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Contains:
        def score(self, **kwargs): return OpikScoreResult("Contains (Unavailable)", 0, "Opik SDK not found").to_dict()
    class RegexMatch:
        def score(self, **kwargs): return OpikScoreResult("RegexMatch (Unavailable)", 0, "Opik SDK not found").to_dict()
    class IsJson:
        def score(self, **kwargs): return OpikScoreResult("IsJson (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Levenshtein:
        def score(self, **kwargs): return OpikScoreResult("Levenshtein (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Hallucination:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Hallucination (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class GEval:
        def __init__(self, task_introduction, evaluation_criteria, model=None):
            self.task_introduction = task_introduction
            self.evaluation_criteria = evaluation_criteria
            self.model = model
        def score(self, **kwargs): return OpikScoreResult(f"GEval (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class Moderation:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Moderation (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class AnswerRelevance:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"AnswerRelevance (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class Usefulness:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Usefulness (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class ContextRecall:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"ContextRecall (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class ContextPrecision:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"ContextPrecision (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()

# --- Ragas SDK Imports ---
RAGAS_SDK_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness as ragas_faithfulness_metric,
        answer_relevancy as ragas_answer_relevancy_metric,
        context_precision as ragas_context_precision_metric,
        context_recall as ragas_context_recall_metric,
    )
    from datasets import Dataset as HuggingFaceDataset
    from ragas.llms import LangchainLLMWrapper
    from langchain_core.language_models.llms import LLM as LangchainLLMBase
    RAGAS_SDK_AVAILABLE = True
    # st.sidebar.success("Ragas SDK loaded successfully!") # Moved
except ImportError as e:
    # st.sidebar.warning(f"Ragas SDK not found or import error: {e}. Please install ragas (`pip install ragas datasets langchain langchain-core`). Ragas metrics demo will be limited.")
    class ragas_faithfulness_metric: pass
    class ragas_answer_relevancy_metric: pass
    class ragas_context_precision_metric: pass
    class ragas_context_recall_metric: pass
    class HuggingFaceDataset:
        @staticmethod
        def from_dict(data): return data
    class LangchainLLMWrapper:
        def __init__(self, llm): self.llm = llm
    class LangchainLLMBase: pass


# --- User-Provided LLM Function (Placeholder) ---
def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    st.sidebar.caption(f"Simulating LLM Call: Model='{model}', Prompt='{prompt[:50].replace('\n', ' ')}...'") # Keep sidebar cleaner
    import time
    import random

    time.sleep(random.uniform(0.1, 0.3)) # Shorter delay for quicker UI
    input_tok = len(prompt.split())
    response_text = f"Dynamic mock for '{model}'. Input: '{prompt.splitlines()[0][:40]}...'"
    
    # Specific mock JSON responses for judge LLMs
    if "judge" in model.lower() or any(m_type in model.lower() for m_type in ["gpt-4o", "claude", "gemini", "hf-judge"]):
        judge_response_data = {"score": round(random.uniform(0.1, 0.95), 2), "reason": "This is a default judge reason based on simulated criteria."}
        if "hallucination" in prompt.lower() or "faithfulness" in prompt.lower() :
            score = random.choice([0.0, 0.0, 1.0]) if "hallucination" in prompt.lower() else round(random.uniform(0.7, 1.0), 2)
            judge_response_data = {"score": score, "reason": "Output evaluated for faithfulness/hallucination."}
        elif "relevance" in prompt.lower() or "pertinent" in prompt.lower():
            judge_response_data = {"score": round(random.uniform(0.6, 1.0), 2), "reason": "Answer relevance assessed."}
        elif "moderation" in prompt.lower():
            judge_response_data = {"score": round(random.uniform(0.0, 0.3), 2), "reason": "Content moderation check performed."} # 0=safe
        elif "g-eval" in model.lower() or "task_introduction" in prompt.lower() or "evaluation_criteria" in prompt.lower():
            raw_judge_score = random.randint(1, 5) # G-Eval often uses 1-5 or 0-10 scales then Opik normalizes
            judge_response_data = {"score": raw_judge_score, "reason": f"G-Eval CoT: Criterion 1 met. Criterion 2 partially met. Score: {raw_judge_score}."}
        response_text = json.dumps(judge_response_data)

    elif "text2sql" in model.lower():
        response_text = "SELECT column FROM table WHERE condition = 'value';"
    elif "rag-primary" in model.lower():
        response_text = "Based on the provided context, the answer points to specific details about the topic."

    output_tok = len(response_text.split())
    return response_text, random.uniform(0.2, 0.8), input_tok, output_tok

# --- Langchain LLM Wrapper for Ragas ---
if RAGAS_SDK_AVAILABLE:
    class RagasCustomLLM(LangchainLLMBase):
        model_name: str = "ragas-judge-llm-via-abc"
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            response_text, _, _, _ = abc_response(self.model_name, prompt)
            return response_text
        @property
        def _llm_type(self) -> str:
            return "ragas_custom_abc_llm"

# --- Custom Metric Definitions ---
class CustomPIIDetectionMetric(OpikBaseMetric if OPIK_SDK_AVAILABLE else object):
    def __init__(self, name="PII Detection (Custom)", pii_patterns=None):
        self.name = name
        self.pii_patterns = pii_patterns or {
            "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            "PHONE_USA": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "CREDIT_CARD_SIMPLE": r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"
        }
        if OPIK_SDK_AVAILABLE and not isinstance(self, OpikBaseMetric): # Check if OpikBaseMetric is the dummy
             pass # Don't call super if it's the dummy OpikBaseMetric
        elif OPIK_SDK_AVAILABLE:
             super().__init__(name=self.name)


    def score(self, text_output: str, **ignored_kwargs) -> Dict:
        detected_pii_details = {}
        pii_found_count = 0
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text_output)
            if matches:
                detected_pii_details[pii_type] = matches
                pii_found_count += len(matches)
        
        score_value = 1.0 if pii_found_count > 0 else 0.0
        reason_str = f"Detected {pii_found_count} PII instances. {'Details: ' + json.dumps(detected_pii_details) if pii_found_count > 0 else 'No PII found.'}"
        
        if OPIK_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"details": detected_pii_details}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"details": detected_pii_details}}

class CustomResponseLengthCheck(OpikBaseMetric if OPIK_SDK_AVAILABLE else object):
    def __init__(self, name="Response Word Count Check (Custom)", min_words=5, max_words=150):
        self.name = name
        self.min_words = min_words
        self.max_words = max_words
        if OPIK_SDK_AVAILABLE and not isinstance(self, OpikBaseMetric):
             pass
        elif OPIK_SDK_AVAILABLE:
             super().__init__(name=self.name)

    def score(self, text_output: str, **ignored_kwargs) -> Dict:
        word_count = len(text_output.split())
        if self.min_words <= word_count <= self.max_words:
            score_value = 1.0 
            reason_str = f"Word count ({word_count}) is within the acceptable range [{self.min_words}-{self.max_words}]."
        else:
            score_value = 0.0 
            reason_str = f"Word count ({word_count}) is outside the acceptable range [{self.min_words}-{self.max_words}]."
        
        if OPIK_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"word_count": word_count}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"word_count": word_count}}

# ----- Streamlit App Config -----
st.set_page_config(layout="wide", page_title="Opik & Ragas Evaluation Metrics Demo")

# --- Display SDK Status Early ---
if 'sdk_status_displayed' not in st.session_state:
    if OPIK_SDK_AVAILABLE:
        st.sidebar.success("Opik SDK loaded successfully!")
    else:
        st.sidebar.error("Opik SDK not found. Opik metrics are disabled. Please install: `pip install opik`")

    if RAGAS_SDK_AVAILABLE:
        st.sidebar.success("Ragas & Langchain SDKs loaded successfully!")
    else:
        st.sidebar.warning("Ragas/Langchain SDKs not found. Ragas metrics are disabled. Install: `pip install ragas datasets langchain langchain-core`")
    st.session_state.sdk_status_displayed = True


st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ("Introduction", "User Guide", "Opik Metrics Demo", "Ragas Metrics Demo", "Custom Metrics Demo"))

# ----- Mock Data (Simplified for brevity in this response, ensure your original detailed mock data is used) -----
MOCK_DATA_QA = {"question": "Capital of France?", "llm_output_correct": "Paris.", "expected_output_exact": "Paris.", "llm_output_with_pii": "email me at test@test.com", "llm_output_unsafe": "unsafe content here", "long_output": "a very long response " * 20}
MOCK_DATA_RAG = {"question": "Eiffel Tower features?", "context_docs": ["The Eiffel Tower is tall.", "It has 3 levels."], "llm_answer_good": "Tall, 3 levels.", "ground_truth_answer_rag": "The Eiffel Tower in Paris is 330m tall and features three visitor levels." , "llm_answer_hallucination": "It is made of cheese."}
MOCK_DATA_TEXT2SQL = {"db_schema": "CREATE TABLE T(C1 INT);", "question_complex": "Complex SQL question?", "generated_sql_complex_correct": "SELECT C1 FROM T;", "sql_syntax_regex": r"^\s*SELECT.*;\s*$"}
MOCK_DATA_AGENTIC = {"llm_agent_output_json": '{"action": "tool", "input": "query"}', "g_eval_agent_task_intro": "Eval agent step.", "g_eval_agent_criteria": "Criteria for agent step."}


# --- Helper for Displaying Results ---
def display_metric_results_streamlit(metric_name, results, llm_response_details=None):
    st.subheader(f"Results for {metric_name}:")
    if isinstance(results, dict): 
        score_key = 'score' if 'score' in results else 'value' 
        st.write(f"**Score:** `{results.get(score_key, 'N/A')}`") 
        if 'reason' in results and results['reason']:
            st.markdown(f"**Reason:** {results['reason']}")
        metadata_to_display = results.get('metadata', results.get('details'))
        if metadata_to_display:
            st.write("**Details/Metadata:**")
            try:
                st.json(metadata_to_display)
            except Exception: # if metadata is not json serializable
                st.write(str(metadata_to_display))

        # For Ragas results which are often flat dicts of metric_name: score_value from to_pandas().to_dict()
        is_standard_format = any(k in results for k in ['score', 'value', 'reason', 'metadata', 'details'])
        if not is_standard_format and results and isinstance(results, dict) and len(results) > 0:
            st.markdown("**Full Results Dictionary:**")
            st.json(results) 
        elif not results and not is_standard_format : 
             st.warning("No detailed results to display (dictionary might be empty or in an unrecognized format).")
    elif OPIK_SDK_AVAILABLE and isinstance(results, OpikScoreResult): 
        st.write(f"**Score:** `{results.value}`")
        if results.reason: st.markdown(f"**Reason:** {results.reason}")
        if results.metadata: st.write("**Metadata:"); st.json(results.metadata)
    else:
        st.write(str(results)) 

    if llm_response_details:
        st.markdown("**LLM Response Details (Input to Evaluation):**")
        resp_text, time_taken, i_tokens, o_tokens = llm_response_details
        st.text_area("LLM Output:", resp_text, height=100, disabled=True, key=f"disp_llm_out_{metric_name.replace(' ','_').replace('(','').replace(')','')}")
        st.caption(f"Time: {time_taken:.2f}s, Input Tokens: {i_tokens}, Output Tokens: {o_tokens}")

# ----- Shared UI Elements -----
st.sidebar.markdown("---")
st.sidebar.header("LLM Configuration")
primary_llm_model_input = st.sidebar.text_input("Primary LLM Model ID/Path:", "my-primary-llm-v1", help="Model ID for abc_response to generate answers.")
judge_llm_model_input = st.sidebar.text_input("Judge LLM Model ID/Path:", "local-hf-judge/model-name", help="Model ID for Opik/Ragas LLM-as-a-Judge, passed to abc_response.")
st.sidebar.markdown("_(Ensure your `abc_response` function can handle these model IDs/paths, including loading local Hugging Face models if specified.)_")

# ----- App Mode Logic -----
if app_mode == "Introduction":
    st.title("Opik & Ragas Evaluation Metrics Demo")
    st.markdown("Welcome! This application demonstrates various evaluation metrics...") # Truncated for brevity
    # ... (full intro markdown from previous correct version) ...
    if not OPIK_SDK_AVAILABLE: st.error("Opik SDK not available.")
    if not RAGAS_SDK_AVAILABLE: st.warning("Ragas/Langchain SDKs not available.")

elif app_mode == "User Guide":
    st.title("User Guide: Opik & Ragas Evaluation Metrics")
    st.markdown("""
    This guide provides detailed information about the evaluation metrics available through Opik and Ragas,
    their typical inputs/outputs, use cases, and categorization by application type (RAG, Text2SQL, Agentic systems).
    All metrics demonstrated in this app use their respective SDKs for calculation.
    LLM-as-a-Judge metrics (from Opik and Ragas) rely on the `abc_response` function to provide judge LLM capabilities.
    """)

    st.header("1. How to Use This Demo")
    st.markdown("""
    1.  **Implement `abc_response`:** Ensure the `abc_response(model, prompt)` function is correctly implemented to call your LLMs.
    2.  **Select Demo Section:** Use the sidebar.
    3.  **Configure LLM Models (Sidebar):** Specify identifiers for your primary and judge LLMs.
    4.  **Choose a Specific Metric.**
    5.  **Review/Edit Input Data.**
    6.  **Generate LLM Output (If Applicable).**
    7.  **Evaluate:** Click the "Evaluate with [Metric Name]" button.
    8.  **View Results.**
    """)

    # --- User Guide Metric Definitions (Copied from previous response, ensure these are complete) ---
    opik_heuristic_metrics_info = [
        {"name": "Equals", "description": "Checks if the LLM's output string *exactly matches* an expected target string.", "inputs": "`output` (str), `expected_output` (str)", "outputs": "Score (1.0 if equal, 0.0 if not), reason.", "use_case": "Verifying precise outputs, known correct answers."},
        {"name": "Contains", "description": "Checks if the LLM's output string *includes a specific substring*. Can be case-sensitive or insensitive.", "inputs": "`output` (str), `substring` (str), `case_sensitive` (bool, optional)", "outputs": "Score (1.0 if contains, 0.0 if not), reason.", "use_case": "Ensuring keywords or required phrases are present."},
        {"name": "RegexMatch", "description": "Checks if the LLM's output *matches a specified regular expression pattern*.", "inputs": "`output` (str), `regex_pattern` (str)", "outputs": "Score (1.0 if matches, 0.0 if not), reason.", "use_case": "Validating output formats (SQL syntax elements, emails, etc.)."},
        {"name": "IsJson", "description": "Checks if the LLM's output is a *syntactically valid JSON object*.", "inputs": "`output` (str)", "outputs": "Score (1.0 if valid JSON, 0.0 if not), reason.", "use_case": "Validating outputs for agentic function arguments or structured data generation."},
        {"name": "Levenshtein Distance", "description": "Calculates the *Levenshtein distance* (edit distance). Opik's metric usually returns a normalized similarity score (0-1, higher is better).", "inputs": "`output` (str), `expected_output` (str)", "outputs": "Normalized similarity score, `distance` (int), reason.", "use_case": "Assessing similarity where minor variations/typos are acceptable."},
    ]
    opik_llm_judge_metrics_info = [
        {"name": "Hallucination", "description": "Assesses if the LLM's output contains fabricated information or statements not supported by the provided context. Uses a judge LLM specified by the `model` parameter during instantiation (defaults to gpt-4o via LiteLLM).", "inputs": "`output` (str), `context` (str, optional), `input` (str, optional - user query)", "outputs": "Score (e.g., 0.0 for no hallucination, 1.0 for hallucination), reason.", "use_case": "Fact-checking in RAG or informational content."},
        {"name": "G-Eval", "description": "A *task-agnostic* LLM-as-a-Judge metric. Users define `task_introduction` and `evaluation_criteria`. The judge LLM (specified by `model`) uses these for evaluation, often via Chain of Thought.", "inputs": "`output` (str - can be complex), `task_introduction` (str - at init), `evaluation_criteria` (str - at init)", "outputs": "Score (often normalized 0-1), reason (may include judge's CoT).", "use_case": "Flexible evaluation for custom aspects, agentic reasoning, Text2SQL semantic correctness."},
        {"name": "Moderation", "description": "Evaluates output appropriateness against safety policies (harmful content, etc.). Uses a judge LLM.", "inputs": "`output` (str)", "outputs": "Safety score (e.g., 0.0 safe, 1.0 unsafe), reason.", "use_case": "Ensuring responsible AI practices."},
        {"name": "AnswerRelevance", "description": "Checks if the LLM's output is relevant to the input question. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - user query)", "outputs": "Score (0-1, higher is more relevant), reason.", "use_case": "Ensuring LLM stays on topic."},
        {"name": "Usefulness", "description": "Assesses if the LLM's output is helpful and valuable. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - user query)", "outputs": "Score (0-1, higher is more useful), reason.", "use_case": "Evaluating if the response is practical and actionable."},
        {"name": "ContextRecall", "description": "Measures if the LLM's output incorporates all relevant information from the provided context. Uses a judge LLM. (Standard definition).", "inputs": "`output` (str), `context` (str), `input` (str, optional)", "outputs": "Score (0-1, higher is better recall), reason.", "use_case": "Crucial for RAG to ensure context is fully utilized; summarization."},
        {"name": "ContextPrecision", "description": "Evaluates if information in the output claimed from context is accurate and relevant to that context. Uses a judge LLM. (Standard definition).", "inputs": "`output` (str), `context` (str), `input` (str, optional), `expected_output` (str, optional)", "outputs": "Score (0-1, higher is better precision), reason.", "use_case": "Ensuring RAG outputs faithfully represent context."},
    ]
    ragas_metrics_info = [
        {"name": "Faithfulness", "source": "Ragas", "description": "Measures if the generated answer is factually consistent with the provided contexts.", "inputs_ragas": "Dataset row with `answer` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher is more faithful).", "use_case": "Core RAG metric for hallucination based on context."},
        {"name": "AnswerRelevancy", "source": "Ragas", "description": "Assesses how pertinent the generated answer is to the given question, considering context.", "inputs_ragas": "Dataset row with `question` (str), `answer` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher is more relevant).", "use_case": "Ensures RAG answer addresses the question."},
        {"name": "ContextPrecision", "source": "Ragas", "description": "Measures the signal-to-noise ratio of retrieved contexts relative to the question.", "inputs_ragas": "Dataset row with `question` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher means more precise contexts).", "use_case": "Evaluates retriever quality in RAG."},
        {"name": "ContextRecall", "source": "Ragas", "description": "Measures if retrieved contexts cover information in the `ground_truth` answer.", "inputs_ragas": "Dataset row with `question` (str), `contexts` (list of str), `ground_truth` (str).", "outputs": "Score (0-1, higher means more ground truth covered).", "use_case": "Evaluates if retriever fetched all necessary info."},
    ]
    custom_metrics_info_guide = [ # Renamed to avoid conflict with demo section list
        {"name": "User-Defined Logic (via OpikBaseMetric)", "source": "Custom", "description": "Create custom evaluation logic by subclassing `opik.evaluation.metrics.base_metric.BaseMetric` and implementing the `score` method.", "inputs": "User-defined.", "outputs": "`OpikScoreResult` object (name, value, reason, metadata).", "use_case": "Evaluating specific business logic, proprietary aspects, or metrics not in Opik/Ragas."},
        {"name": "PII Detection (Custom Demo)", "source": "Custom", "description": "A custom metric using regex to detect potential Personally Identifiable Information (PII) in text.", "inputs": "`text_output` (str)", "outputs": "Score (1.0 if PII detected), PII count, details.", "use_case": "Basic PII scanning. Real PII detection is more complex."},
        {"name": "Response Word Count Check (Custom Demo)", "source": "Custom", "description": "A custom heuristic metric to check if response word count is within a defined range.", "inputs": "`text_output` (str), `min_words` (int), `max_words` (int)", "outputs": "Score (1.0 if in range), actual word count.", "use_case": "Ensuring response brevity or sufficient detail."}
    ]

    st.header("2. Metrics Catalog")
    st.subheader("2.1 Opik Built-in Heuristic Metrics")
    for metric in opik_heuristic_metrics_info:
        with st.expander(f"Opik: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    st.subheader("2.2 Opik Built-in LLM-as-a-Judge Metrics")
    st.markdown("These metrics use a 'judge' LLM, configured via the `model` parameter at instantiation (e.g., `Hallucination(model=your_judge_llm_id)`). This demo routes these calls through `abc_response(model=your_judge_llm_id, prompt=...)`.")
    for metric in opik_llm_judge_metrics_info:
        with st.expander(f"Opik: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")
    
    st.subheader("2.3 Ragas Metrics (for RAG Systems)")
    st.markdown("""
    Ragas metrics are designed for RAG pipelines. They often require specific data structures (like a Hugging Face `Dataset`) and an LLM for judging, which this demo simulates via a Langchain wrapper around `abc_response`.
    """)
    for metric in ragas_metrics_info:
        with st.expander(f"Ragas: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Ragas Inputs:** {metric['inputs_ragas']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    st.subheader("2.4 Custom Metrics")
    st.markdown("Examples of how you can define your own metrics using Opik's `BaseMetric`.")
    for metric in custom_metrics_info_guide: # Use the renamed list for the guide
        with st.expander(f"{metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")


    st.header("3. Metrics by Application Type")
    # ... (Insert the detailed RAG, Text2SQL, Agentic categorization from the previous response here) ...
    st.subheader("3.1 RAG (Retrieval Augmented Generation) Systems")
    st.markdown("""
    - **Core RAG Quality (Answer vs. Context):**
        - Opik `Hallucination`, Opik `ContextPrecision`, Opik `ContextRecall`
        - Ragas `faithfulness`, Ragas `answer_relevancy`
    - **Retrieval Quality (Contexts vs. Question/Ground Truth):**
        - Ragas `context_precision`, Ragas `context_recall`
    - **General Answer Quality:**
        - Opik `AnswerRelevance`, Opik `Usefulness`
        - (Consider Opik `SentenceBLEU`, `CorpusBLEU`, `ROUGE` if available/custom)
    """)

    st.subheader("3.2 Text2SQL")
    st.markdown("""
    - **SQL Correctness & Validity:**
        - Opik `RegexMatch` (basic syntax)
        - **Opik `G-Eval` (Primary for semantic correctness against schema & NLQ)**
        - Custom Metric (SQL Execution Check - conceptual)
    - **Query Comparison (vs. reference SQL):**
        - Opik `Equals` (exact match)
    - **NL Explanation Quality (if provided):**
        - Opik `AnswerRelevance`, Opik `Usefulness`
    """)

    st.subheader("3.3 Agentic Systems (Multi-step, Tool Use)")
    st.markdown("""
    - **Action & Tool Use Validation:**
        - Opik `IsJson` (for structured outputs like tool parameters)
        - Opik `Equals` / `Contains` (for specific tool names, parameters)
        - Custom Metric (valid tool name, parameter validation)
    - **Reasoning & Planning Quality:**
        - **Opik `G-Eval` (Primary for evaluating thoughts, plans, action choices)**
    - **Groundedness & Safety:**
        - Opik `Hallucination` (if thoughts/responses should be grounded)
        - Opik `Moderation` (for safe agent interactions)
    - **Final Output Quality:**
        - Opik `AnswerRelevance`, Opik `Usefulness`
    """)
    st.markdown("---")
    st.info("Remember to adapt the `task_introduction` and `evaluation_criteria` for `G-Eval` to precisely fit your specific evaluation needs for Text2SQL or Agentic steps.")


elif app_mode == "Opik Metrics Demo":
    if not OPIK_SDK_AVAILABLE:
        st.error("Opik SDK is not available. Please install it (`pip install opik`) to use this section.")
    else:
        st.title("Opik Built-in Metrics Demonstration")
        # ... (Rest of the Opik Metrics Demo section, ensure all indentations are correct) ...
        # This section was quite long; I'll assume the indentation logic from the previous
        # correct Python block for this section is what's needed.
        # I will focus on correcting any `class X: def method(...)` style errors in dummy classes
        # and ensure Streamlit 'with' blocks are correct.
        # For brevity, I'm not repeating the entire Opik demo UI code here if it was correct before,
        # but the indentation fix principles apply throughout. The main issue was dummy class defs.
        opik_metric_category = st.selectbox(
            "Select Opik Metric Type:",
            ("Heuristic", "LLM-as-a-Judge"),
            key="opik_cat_select_main" # Unique key
        )

        if opik_metric_category == "Heuristic":
            heuristic_metric_name = st.selectbox(
                "Choose a Heuristic Metric:",
                [m['name'] for m in opik_heuristic_metrics_info],
                key="opik_heur_select_main" # Unique key
            )
            metric_info = next(m for m in opik_heuristic_metrics_info if m['name'] == heuristic_metric_name)
            st.markdown(f"**Selected Opik Metric:** `{metric_info['name']}` - {metric_info['description']}")
            
            cols_opik_h = st.columns(2)
            with cols_opik_h[0]:
                st.subheader("Inputs:")
                default_output_key_h = "llm_output_correct" if heuristic_metric_name != "IsJson" else "llm_agent_output_json"
                default_source_h = MOCK_DATA_QA if heuristic_metric_name != "IsJson" else MOCK_DATA_AGENTIC
                llm_output_h = st.text_area("LLM Output:", default_source_h[default_output_key_h], height=100, key=f"opik_hm_out_main_{heuristic_metric_name}")
                expected_input_h = ""
                case_sensitive_h = True
                if heuristic_metric_name in ["Equals", "Levenshtein Distance"]:
                    expected_input_h = st.text_input("Expected Output:", MOCK_DATA_QA["expected_output_exact"], key=f"opik_hm_exp_main_{heuristic_metric_name}")
                elif heuristic_metric_name == "Contains":
                    expected_input_h = st.text_input("Substring:", "Portuguese", key=f"opik_hm_subs_main_{heuristic_metric_name}")
                    case_sensitive_h = st.checkbox("Case Sensitive?", True, key=f"opik_hm_case_main_{heuristic_metric_name}")
                elif heuristic_metric_name == "RegexMatch":
                    expected_input_h = st.text_input("Regex Pattern:", MOCK_DATA_TEXT2SQL["sql_syntax_regex"], key=f"opik_hm_regex_main_{heuristic_metric_name}")

                if st.button(f"Evaluate Opik {heuristic_metric_name}", key=f"btn_opik_hm_main_{heuristic_metric_name}"):
                    try:
                        metric_instance_h = getattr(__import__(__name__), heuristic_metric_name.replace(" ", ""))() # Instantiate dynamically
                        if heuristic_metric_name == "Contains":
                            results_h = metric_instance_h.score(output=llm_output_h, substring=expected_input_h, case_sensitive=case_sensitive_h)
                        elif heuristic_metric_name == "RegexMatch":
                             results_h = metric_instance_h.score(output=llm_output_h, regex_pattern=expected_input_h)
                        elif heuristic_metric_name in ["Equals", "Levenshtein Distance"]:
                             results_h = metric_instance_h.score(output=llm_output_h, expected_output=expected_input_h)
                        else: # IsJson
                             results_h = metric_instance_h.score(output=llm_output_h)
                        st.session_state[f"opik_hm_res_main_{heuristic_metric_name}"] = results_h.to_dict() if hasattr(results_h, 'to_dict') else results_h
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.code(traceback.format_exc())
            with cols_opik_h[1]:
                if f"opik_hm_res_main_{heuristic_metric_name}" in st.session_state:
                    display_metric_results_streamlit(f"Opik {heuristic_metric_name}", st.session_state[f"opik_hm_res_main_{heuristic_metric_name}"])
        
        elif opik_metric_category == "LLM-as-a-Judge":
            llm_judge_metric_name = st.selectbox(
                "Choose an Opik LLM-as-a-Judge Metric:",
                [m['name'] for m in opik_llm_judge_metrics_info],
                key="opik_llmj_select_main"
            )
            metric_info = next(m for m in opik_llm_judge_metrics_info if m['name'] == llm_judge_metric_name)
            st.markdown(f"**Selected Opik Metric:** `{metric_info['name']}` - {metric_info['description']}")
            st.caption(f"Judge LLM: '{judge_llm_model_input}' (via `abc_response`).")

            cols_opik_lj = st.columns(2)
            generated_llm_output_lj = ""
            llm_details_lj = None

            with cols_opik_lj[0]:
                st.subheader("Inputs & Primary LLM Output Gen:")
                user_query_lj, context_lj, expected_lj, task_intro_lj, eval_crit_lj = "", "", "", "", ""
                primary_prompt_lj, default_output_lj = "", ""

                # Simplified input setup
                if llm_judge_metric_name == "Moderation":
                    default_output_lj = MOCK_DATA_QA["llm_output_unsafe"]
                else:
                    user_query_lj = st.text_input("User Question/Input:", MOCK_DATA_QA["question"] if llm_judge_metric_name not in ["ContextRecall", "ContextPrecision", "Hallucination"] else MOCK_DATA_RAG["question"], key=f"opik_lj_q_{llm_judge_metric_name}")
                if llm_judge_metric_name in ["Hallucination", "ContextRecall", "ContextPrecision"]:
                    context_lj = st.text_area("Context:", MOCK_DATA_RAG["context_docs"][0], height=100, key=f"opik_lj_ctx_{llm_judge_metric_name}")
                if llm_judge_metric_name == "ContextPrecision":
                    expected_lj = st.text_input("Expected Answer (Optional):", MOCK_DATA_RAG["ground_truth_answer_rag"], key=f"opik_lj_exp_{llm_judge_metric_name}")
                if llm_judge_metric_name == "G-Eval":
                    task_intro_lj = st.text_area("G-Eval: Task Introduction", MOCK_DATA_TEXT2SQL["db_schema"] + "\n\nNLQ: " + MOCK_DATA_TEXT2SQL["question_complex"], height=100, key=f"opik_lj_geval_intro_{llm_judge_metric_name}")
                    eval_crit_lj = st.text_area("G-Eval: Evaluation Criteria", "1. SQL Syntactic Correctness (0-1)\n2. Semantic Alignment (0-1)\nReturn JSON: {'score': <0-1_score>, 'reason': <CoT>}", height=100, key=f"opik_lj_geval_crit_{llm_judge_metric_name}")
                    default_output_lj = MOCK_DATA_TEXT2SQL["generated_sql_complex_correct"]
                
                if llm_judge_metric_name != "Moderation": # Moderation is direct input
                    primary_prompt_lj = user_query_lj
                    if context_lj: primary_prompt_lj = f"Context: {context_lj}\nQuestion: {user_query_lj}"
                    if llm_judge_metric_name == "G-Eval": primary_prompt_lj = f"Generate output based on: {task_intro_lj}" # G-Eval output is context-dependent

                    if primary_prompt_lj and st.button(f"Generate Output via '{primary_llm_model_input}'", key=f"btn_gen_opik_ljm_main_{llm_judge_metric_name}"):
                        st.session_state[f"opik_ljm_details_main_{llm_judge_metric_name}"] = abc_response(primary_llm_model_input, primary_prompt_lj)

                if f"opik_ljm_details_main_{llm_judge_metric_name}" in st.session_state:
                    llm_details_lj = st.session_state[f"opik_ljm_details_main_{llm_judge_metric_name}"]
                    generated_llm_output_lj = llm_details_lj[0]
                
                generated_llm_output_lj = st.text_area("LLM Output to Evaluate:", generated_llm_output_lj or default_output_lj, height=100, key=f"opik_ljm_llmout_main_{llm_judge_metric_name}")
                if llm_details_lj and generated_llm_output_lj == llm_details_lj[0]:
                    st.caption(f"Time: {llm_details_lj[1]:.2f}s, InTok: {llm_details_lj[2]}, OutTok: {llm_details_lj[3]}")
                elif not llm_details_lj and generated_llm_output_lj:
                    llm_details_lj = (generated_llm_output_lj, 0.0, 0, len(generated_llm_output_lj.split()))


                if st.button(f"Evaluate Opik {llm_judge_metric_name}", key=f"btn_opik_ljm_eval_main_{llm_judge_metric_name}"):
                    if not generated_llm_output_lj: st.warning("Need LLM output to evaluate.")
                    else:
                        try:
                            metric_args = {"model": judge_llm_model_input}
                            score_args = {"output": generated_llm_output_lj}
                            if llm_judge_metric_name == "G-Eval": 
                                metric_args.update({"task_introduction": task_intro_lj, "evaluation_criteria": eval_crit_lj})
                            else: score_args["input"] = user_query_lj # Common for many
                            if llm_judge_metric_name in ["Hallucination", "ContextRecall", "ContextPrecision"]: score_args["context"] = context_lj
                            if llm_judge_metric_name == "ContextPrecision": score_args["expected_output"] = expected_lj
                            
                            # Instantiate and score
                            metric_class_lj = globals()[llm_judge_metric_name.replace(" ", "")] # Dynamic class instantiation
                            metric_instance_lj = metric_class_lj(**metric_args) if llm_judge_metric_name == "G-Eval" else metric_class_lj(model=judge_llm_model_input)
                            
                            # Remove 'model' from score_args if it's not an actual input for .score() for that metric
                            if llm_judge_metric_name == "Moderation": score_args = {"output": generated_llm_output_lj}
                            
                            results_lj = metric_instance_lj.score(**score_args)
                            st.session_state[f"opik_ljm_res_main_{llm_judge_metric_name}"] = results_lj.to_dict() if hasattr(results_lj, 'to_dict') else results_lj
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.code(traceback.format_exc())
            with cols_opik_lj[1]:
                if f"opik_ljm_res_main_{llm_judge_metric_name}" in st.session_state:
                    display_metric_results_streamlit(f"Opik {llm_judge_metric_name}", st.session_state[f"opik_ljm_res_main_{llm_judge_metric_name}"], llm_details_lj)


elif app_mode == "Ragas Metrics Demo":
    if not RAGAS_SDK_AVAILABLE:
        st.error("Ragas & supporting Langchain SDKs not available. Install: `pip install ragas datasets langchain langchain-core`")
    elif not OPIK_SDK_AVAILABLE: # For abc_response
        st.error("A working `abc_response` (from the Opik SDK section) is needed for the Ragas LLM wrapper.")
    else:
        st.title("Ragas Metrics Demonstration (for RAG Evaluation)")
        # ... (Ragas Demo section - ensure indentation is correct here too)
        # This section was quite long; I'll assume the indentation logic from the previous
        # correct Python block for this section is what's needed.
        ragas_llm_wrapper_for_ragas_demo = None
        try:
            ragas_judge_llm_instance_for_demo = RagasCustomLLM(model_name=judge_llm_model_input)
            ragas_llm_wrapper_for_ragas_demo = LangchainLLMWrapper(ragas_judge_llm_instance_for_demo)
            st.info(f"Ragas using judge: '{judge_llm_model_input}' via Langchain wrapper for `abc_response`.")
        except Exception as e:
            st.warning(f"Ragas LLM wrapper init error: {e}. LLM-based Ragas metrics might fail.")

        ragas_metric_name_selected = st.selectbox("Choose Ragas Metric:", [m['name'] for m in ragas_metrics_info], key="ragas_select_main")
        metric_info_ragas = next(m for m in ragas_metrics_info if m['name'] == ragas_metric_name_selected)
        st.markdown(f"**Selected Ragas Metric:** `{metric_info_ragas['name']}` - {metric_info_ragas['description']}")

        cols_ragas_main = st.columns(2)
        with cols_ragas_main[0]:
            st.subheader("RAG Inputs:")
            q_ragas = st.text_input("Question:", MOCK_DATA_RAG["question"], key="ragas_q_main")
            ctx_docs_ragas = [st.text_area(f"Context Doc {i+1}:", MOCK_DATA_RAG["context_docs"][i], height=60, key=f"ragas_ctx_main_{i}") for i in range(len(MOCK_DATA_RAG["context_docs"]))]
            gt_ragas = ""
            if ragas_metric_name_selected in ["ContextRecall"]: # Ragas ContextRecall needs ground_truth
                gt_ragas = st.text_area("Ground Truth Answer:", MOCK_DATA_RAG["ground_truth_answer_rag"], height=60, key="ragas_gt_main")
            
            st.markdown(f"**1. Generate Answer via '{primary_llm_model_input}':**")
            if st.button("Generate RAG Answer", key="btn_ragas_gen_main"):
                prompt_ragas_gen = f"Contexts:\n{'---'.join(ctx_docs_ragas)}\n\nQuestion: {q_ragas}"
                st.session_state.ragas_ans_details_main = abc_response(primary_llm_model_input, prompt_ragas_gen)

            ans_ragas_gen = ""
            details_ans_ragas_gen = None
            if "ragas_ans_details_main" in st.session_state:
                details_ans_ragas_gen = st.session_state.ragas_ans_details_main
                ans_ragas_gen = details_ans_ragas_gen[0]
            
            ans_ragas_gen = st.text_area("Generated RAG Answer:", ans_ragas_gen or MOCK_DATA_RAG["llm_answer_good"], height=100, key="ragas_ans_main")
            if details_ans_ragas_gen and ans_ragas_gen == details_ans_ragas_gen[0]:
                 st.caption(f"Time: {details_ans_ragas_gen[1]:.2f}s, InTok: {details_ans_ragas_gen[2]}, OutTok: {details_ans_ragas_gen[3]}")
            elif not details_ans_ragas_gen and ans_ragas_gen:
                 details_ans_ragas_gen = (ans_ragas_gen, 0.0,0,len(ans_ragas_gen.split()))

            if st.button(f"Evaluate Ragas {ragas_metric_name_selected}", key=f"btn_eval_ragas_main_{ragas_metric_name_selected}"):
                if not ans_ragas_gen: st.warning("Need generated answer.")
                elif not ragas_llm_wrapper_for_ragas_demo and ragas_metric_name_selected in ["Faithfulness", "AnswerRelevancy", "ContextRecall"]:
                    st.error("Ragas LLM wrapper for judge not ready.")
                else:
                    data_dict_ragas = {"question": [q_ragas], "answer": [ans_ragas_gen], "contexts": [ctx_docs_ragas]}
                    if gt_ragas and ragas_metric_name_selected == "ContextRecall": data_dict_ragas["ground_truth"] = [gt_ragas]
                    
                    try:
                        dataset_hf_ragas = HuggingFaceDataset.from_dict(data_dict_ragas)
                        current_ragas_metric_instance = None
                        if ragas_metric_name_selected == "Faithfulness": current_ragas_metric_instance = ragas_faithfulness_metric(llm=ragas_llm_wrapper_for_ragas_demo)
                        elif ragas_metric_name_selected == "AnswerRelevancy": current_ragas_metric_instance = ragas_answer_relevancy_metric(llm=ragas_llm_wrapper_for_ragas_demo)
                        elif ragas_metric_name_selected == "ContextPrecision": current_ragas_metric_instance = ragas_context_precision_metric(llm=ragas_llm_wrapper_for_ragas_demo) # Can take LLM
                        elif ragas_metric_name_selected == "ContextRecall": 
                            if "ground_truth" not in data_dict_ragas: st.error("Ground truth needed for Ragas ContextRecall."); st.stop()
                            current_ragas_metric_instance = ragas_context_recall_metric(llm=ragas_llm_wrapper_for_ragas_demo)

                        if current_ragas_metric_instance:
                            with st.spinner(f"Ragas '{ragas_metric_name_selected}' running..."):
                                eval_results_ragas = ragas_evaluate(dataset_hf_ragas, metrics=[current_ragas_metric_instance]) # llm already in metric
                            st.session_state[f"ragas_res_main_{ragas_metric_name_selected}"] = eval_results_ragas.to_pandas().to_dict(orient='records')[0] if eval_results_ragas and not eval_results_ragas.to_pandas().empty else {"error": "Ragas returned empty."}
                        else: st.error("Ragas metric not configured.")
                    except Exception as e:
                        st.error(f"Ragas Error: {e}")
                        st.code(traceback.format_exc())
        with cols_ragas_main[1]:
            if f"ragas_res_main_{ragas_metric_name_selected}" in st.session_state:
                display_metric_results_streamlit(f"Ragas {ragas_metric_name_selected}", st.session_state[f"ragas_res_main_{ragas_metric_name_selected}"], details_ans_ragas_gen)

elif app_mode == "Custom Metrics Demo":
    if not OPIK_SDK_AVAILABLE:
        st.error("Opik SDK is required for the custom metric base classes. Please install it.")
    else:
        st.title("Custom Opik Metrics Demonstration")
        # ... (Custom Metrics Demo section - ensure indentation is correct here too) ...
        custom_metric_choice_val_main = st.selectbox(
            "Choose Custom Metric:",
            ("PII Detection (Demo)", "Response Word Count Check (Demo)"),
            key="custom_select_main"
        )
        cols_custom_main = st.columns(2)
        with cols_custom_main[0]:
            st.subheader("Inputs:")
            if custom_metric_choice_val_main == "PII Detection (Demo)":
                metric_info_cust = next(m for m in custom_metrics_info_guide if m['name'] == "PII Detection (Custom Demo)")
                st.markdown(f"**Metric:** `{metric_info_cust['name']}` - {metric_info_cust['description']}")
                text_pii = st.text_area("Text to Scan for PII:", MOCK_DATA_QA["llm_output_with_pii"], height=100, key="custom_pii_text_main")
                if st.button("Evaluate PII Detection", key="btn_custom_pii_main"):
                    metric_pii = CustomPIIDetectionMetric()
                    st.session_state.custom_pii_res_main = metric_pii.score(text_output=text_pii)
            
            elif custom_metric_choice_val_main == "Response Word Count Check (Demo)":
                metric_info_cust = next(m for m in custom_metrics_info_guide if m['name'] == "Response Word Count Check (Custom Demo)")
                st.markdown(f"**Metric:** `{metric_info_cust['name']}` - {metric_info_cust['description']}")
                text_len = st.text_area("Text for Length Check:", MOCK_DATA_QA["long_output"], height=100, key="custom_len_text_main")
                min_w_main = st.number_input("Min Words:", 1, 1000, 10, key="cust_minw_main")
                max_w_main = st.number_input("Max Words:", min_w_main + 1 if min_w_main else 2, 2000, 100, key="cust_maxw_main")
                if st.button("Evaluate Length Check", key="btn_custom_len_main"):
                    metric_len = CustomResponseLengthCheck(min_words=min_w_main, max_words=max_w_main)
                    st.session_state.custom_len_res_main = metric_len.score(text_output=text_len)
        with cols_custom_main[1]:
            if custom_metric_choice_val_main == "PII Detection (Demo)" and 'custom_pii_res_main' in st.session_state:
                display_metric_results_streamlit("Custom PII Detection", st.session_state.custom_pii_res_main)
            elif custom_metric_choice_val_main == "Response Word Count Check (Demo)" and 'custom_len_res_main' in st.session_state:
                display_metric_results_streamlit("Custom Word Count Check", st.session_state.custom_len_res_main)

# Fallback for SDK not available on main page selection
elif (app_mode == "Opik Metrics Demo" and not OPIK_SDK_AVAILABLE) or \
     (app_mode == "Ragas Metrics Demo" and (not RAGAS_SDK_AVAILABLE or not OPIK_SDK_AVAILABLE)) or \
     (app_mode == "Custom Metrics Demo" and not OPIK_SDK_AVAILABLE):
    st.error(f"Required SDK(s) for '{app_mode}' are not available. Please check installation messages in the sidebar and install necessary packages.")
