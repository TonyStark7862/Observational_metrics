import streamlit as st
import re
import json
from typing import Any, Dict, List, Optional, Tuple # For type hinting
import traceback # For error logging

# --- Opik SDK Imports ---
try:
    from opik.evaluation.metrics import (
        Equals, Contains, RegexMatch, IsJson, LevenshteinRatio, # Corrected LevenshteinRatio
        Hallucination, GEval, Moderation, AnswerRelevance, Usefulness,
        ContextRecall, ContextPrecision
    )
    # Potentially add Opik's built-in BLEU and ROUGE if confirmed available
    # from opik.evaluation.metrics.heuristic_metrics import SentenceBLEU, CorpusBLEU, ROUGE_N
    from opik.evaluation.metrics.base_metric import BaseMetric as OpikBaseMetric
    from opik.evaluation.metrics.score_result import ScoreResult as OpikScoreResult
    from opik.evaluation import models as opik_models # For LiteLLMChatModel
    PRIMARY_SDK_AVAILABLE = True # Renamed for generic UI
except ImportError as e:
    PRIMARY_SDK_AVAILABLE = False
    class OpikBaseMetric: # Dummy base
        def __init__(self, name: str = "DummyPrimaryBase"):
            self.name = name
    class OpikScoreResult: # Dummy result
        def __init__(self, name, value, reason=None, metadata=None):
            self.name = name
            self.value = value
            self.reason = reason
            self.metadata = metadata
        def to_dict(self):
            return {"name": self.name, "score": self.value, "reason": self.reason, "metadata": self.metadata}

    # Define dummy Primary SDK metric classes
    class Equals:
        def score(self, **kwargs): return OpikScoreResult("Equals (SDK Unavailable)", 0, "Primary SDK not found").to_dict()
    class Contains:
        def score(self, **kwargs): return OpikScoreResult("Contains (SDK Unavailable)", 0, "Primary SDK not found").to_dict()
    class RegexMatch:
        def score(self, **kwargs): return OpikScoreResult("RegexMatch (SDK Unavailable)", 0, "Primary SDK not found").to_dict()
    class IsJson:
        def score(self, **kwargs): return OpikScoreResult("IsJson (SDK Unavailable)", 0, "Primary SDK not found").to_dict()
    class LevenshteinRatio: # Corrected dummy name
        def score(self, **kwargs): return OpikScoreResult("LevenshteinRatio (SDK Unavailable)", 0, "Primary SDK not found").to_dict()
    class Hallucination:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Hallucination (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class GEval:
        def __init__(self, task_introduction, evaluation_criteria, model=None):
            self.task_introduction = task_introduction
            self.evaluation_criteria = evaluation_criteria
            self.model = model
        def score(self, **kwargs): return OpikScoreResult(f"GEval (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class Moderation:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Moderation (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class AnswerRelevance:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"AnswerRelevance (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class Usefulness:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"Usefulness (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class ContextRecall:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"ContextRecall (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()
    class ContextPrecision:
        def __init__(self, model=None): self.model=model
        def score(self, **kwargs): return OpikScoreResult(f"ContextPrecision (SDK Unavailable - Judge: {self.model})", 0, "Primary SDK not found").to_dict()

# --- RAG-Focused Library SDK Imports (e.g., Ragas) ---
RAG_LIB_SDK_AVAILABLE = False
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
    RAG_LIB_SDK_AVAILABLE = True
except ImportError as e:
    # Define dummy classes if Ragas is not available for UI rendering
    class ragas_faithfulness_metric: pass
    class ragas_answer_relevancy_metric: pass
    class ragas_context_precision_metric: pass
    class ragas_context_recall_metric: pass
    if 'HuggingFaceDataset' not in globals(): # Avoid re-defining if datasets was imported for other reasons
        class HuggingFaceDataset:
            @staticmethod
            def from_dict(data): return data # Simplistic mock
    if 'LangchainLLMWrapper' not in globals():
        class LangchainLLMWrapper:
            def __init__(self, llm): self.llm = llm
    if 'LangchainLLMBase' not in globals():
        class LangchainLLMBase: pass


# --- User-Provided LLM Function (Placeholder) ---
def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    st.sidebar.caption(f"Simulating LLM Call: Model='{model}', Prompt='{prompt[:50].replace('\n', ' ')}...'")
    import time, random
    time.sleep(random.uniform(0.1, 0.3))
    input_tok = len(prompt.split())
    response_text = f"Dynamic mock for '{model}'. Input: '{prompt.splitlines()[0][:40]}...'"
    if "judge" in model.lower() or any(m_type in model.lower() for m_type in ["gpt-4o", "claude", "gemini", "hf-judge"]):
        judge_response_data = {"score": round(random.uniform(0.1, 0.95), 2), "reason": "Default judge reason."}
        if "hallucination" in prompt.lower() or "faithfulness" in prompt.lower():
            score = random.choice([0.0, 1.0]) if "hallucination" in prompt.lower() else round(random.uniform(0.7, 1.0), 2)
            judge_response_data = {"score": score, "reason": "Evaluated for faithfulness/hallucination."}
        elif "relevance" in prompt.lower() or "pertinent" in prompt.lower():
            judge_response_data = {"score": round(random.uniform(0.6, 1.0), 2), "reason": "Relevance assessed."}
        elif "moderation" in prompt.lower():
            judge_response_data = {"score": round(random.uniform(0.0, 0.3), 2), "reason": "Moderation check done."}
        elif "g-eval" in model.lower() or "task_introduction" in prompt.lower() or "evaluation_criteria" in prompt.lower():
            raw_judge_score = random.randint(1, 5)
            judge_response_data = {"score": raw_judge_score, "reason": f"G-Eval CoT: Score: {raw_judge_score}."}
        response_text = json.dumps(judge_response_data)
    elif "text2sql" in model.lower(): response_text = "SELECT column FROM table WHERE condition = 'value';"
    elif "rag-primary" in model.lower(): response_text = "Based on context, the answer is X."
    output_tok = len(response_text.split())
    return response_text, random.uniform(0.2, 0.8), input_tok, output_tok

# --- Langchain LLM Wrapper for RAG-Focused Library (using abc_response) ---
ragas_llm_wrapper_for_demo = None
if RAG_LIB_SDK_AVAILABLE:
    class RagasCustomLLMForDemo(LangchainLLMBase): # Renamed for clarity
        model_name: str = "ragas-judge-via-abc"
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            response_text, _, _, _ = abc_response(self.model_name, prompt)
            return response_text
        @property
        def _llm_type(self) -> str: return "ragas_custom_abc_llm_for_demo"
    try:
        ragas_llm_wrapper_for_demo = LangchainLLMWrapper(RagasCustomLLMForDemo())
    except Exception: # Broad exception if LangchainLLMWrapper itself fails
        RAG_LIB_SDK_AVAILABLE = False # Downgrade if wrapper fails


# --- Custom Metric Definitions ---
class CustomPIIDetectionMetric(OpikBaseMetric if PRIMARY_SDK_AVAILABLE else object):
    def __init__(self, name="PII Detection (Custom)", pii_patterns=None):
        self.name = name
        self.pii_patterns = pii_patterns or {
            "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            "PHONE_USA": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        }
        if PRIMARY_SDK_AVAILABLE and isinstance(self, OpikBaseMetric):
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
        if PRIMARY_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"details": detected_pii_details}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"details": detected_pii_details}}

class CustomResponseLengthCheck(OpikBaseMetric if PRIMARY_SDK_AVAILABLE else object):
    def __init__(self, name="Response Word Count Check (Custom)", min_words=5, max_words=150):
        self.name = name
        self.min_words = min_words
        self.max_words = max_words
        if PRIMARY_SDK_AVAILABLE and isinstance(self, OpikBaseMetric):
            super().__init__(name=self.name)

    def score(self, text_output: str, **ignored_kwargs) -> Dict:
        word_count = len(text_output.split())
        score_value = 1.0 if self.min_words <= word_count <= self.max_words else 0.0
        reason_str = f"Word count ({word_count}) is {'within' if score_value == 1.0 else 'outside'} acceptable range [{self.min_words}-{self.max_words}]."
        if PRIMARY_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"word_count": word_count}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"word_count": word_count}}


# ----- Metric Info Lists (Defined Globally) -----
heuristic_metrics_info_list = [
    {"name": "Equals", "description": "Checks exact string match.", "inputs": "`output` (str), `reference` (str)", "outputs": "Score (1.0 if equal, 0.0 if not), reason.", "use_case": "Precise outputs."},
    {"name": "Contains", "description": "Checks for substring presence.", "inputs": "`output` (str), `substring` (str), `case_sensitive` (bool, opt)", "outputs": "Score (1.0 if contains, 0.0 if not), reason.", "use_case": "Keyword presence."},
    {"name": "RegexMatch", "description": "Matches output against regex pattern.", "inputs": "`output` (str), `regex_pattern` (str)", "outputs": "Score (1.0 if matches, 0.0 if not), reason.", "use_case": "Format validation (SQL, email)."},
    {"name": "IsJson", "description": "Checks for valid JSON object.", "inputs": "`output` (str)", "outputs": "Score (1.0 if valid, 0.0 if not), reason.", "use_case": "Agentic outputs, structured data."},
    {"name": "LevenshteinRatio", "description": "Normalized Levenshtein distance (similarity). Higher is better.", "inputs": "`output` (str), `reference` (str)", "outputs": "Similarity score (0-1), distance, reason.", "use_case": "Similarity with typo tolerance."},
]
llm_judge_metrics_info_list = [
    {"name": "Hallucination", "description": "Assesses if output fabricates info or contradicts context. Uses a judge LLM.", "inputs": "`output` (str), `context` (str, opt), `input` (str, opt)", "outputs": "Score (0.0 no hallucination, 1.0 hallucination), reason.", "use_case": "Fact-checking in RAG."},
    {"name": "GEval", "description": "Task-agnostic LLM-as-a-Judge. User defines `task_introduction` & `evaluation_criteria`.", "inputs": "`output` (str), `task_introduction` (init), `evaluation_criteria` (init)", "outputs": "Score (0-1), reason (CoT).", "use_case": "Custom aspects, agentic steps, Text2SQL."},
    {"name": "Moderation", "description": "Evaluates output appropriateness (safety, harm). Uses a judge LLM.", "inputs": "`output` (str)", "outputs": "Safety score (0.0 safe, 1.0 unsafe), reason.", "use_case": "Responsible AI."},
    {"name": "AnswerRelevance", "description": "Checks output relevance to input question. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - query)", "outputs": "Score (0-1), reason.", "use_case": "LLM stays on topic."},
    {"name": "Usefulness", "description": "Assesses if output is helpful/valuable. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - query)", "outputs": "Score (0-1), reason.", "use_case": "Practicality of response."},
    {"name": "ContextRecall", "description": "Measures if output uses all relevant info from context. Uses judge LLM.", "inputs": "`output` (str), `context` (str), `input` (str, opt)", "outputs": "Score (0-1), reason.", "use_case": "RAG context utilization."},
    {"name": "ContextPrecision", "description": "Evaluates if output info claimed from context is accurate/relevant to context. Uses judge LLM.", "inputs": "`output` (str), `context` (str), `input` (str, opt), `expected_output` (str, opt)", "outputs": "Score (0-1), reason.", "use_case": "RAG faithfulness to context."},
]
rag_focused_metrics_info_list = [
    {"name": "Faithfulness", "source": "RAG Library", "description": "Measures factual consistency of answer with contexts.", "inputs_lib": "Dataset row: `answer` (str), `contexts` (list[str]).", "outputs": "Score (0-1).", "use_case": "Core RAG hallucination check."},
    {"name": "AnswerRelevancy", "source": "RAG Library", "description": "Assesses answer pertinence to question, given context.", "inputs_lib": "Dataset row: `question` (str), `answer` (str), `contexts` (list[str]).", "outputs": "Score (0-1).", "use_case": "Ensures RAG answer is on topic."},
    {"name": "ContextPrecision", "source": "RAG Library", "description": "Measures signal-to-noise of retrieved contexts for the question.", "inputs_lib": "Dataset row: `question` (str), `contexts` (list[str]).", "outputs": "Score (0-1).", "use_case": "RAG retriever quality."},
    {"name": "ContextRecall", "source": "RAG Library", "description": "Measures if retrieved contexts cover `ground_truth` info.", "inputs_lib": "Dataset row: `question` (str), `contexts` (list[str]), `ground_truth` (str).", "outputs": "Score (0-1).", "use_case": "RAG retriever completeness."},
]
custom_metrics_info_list_guide = [
    {"name": "User-Defined Logic", "source": "Custom", "description": "Via `BaseMetric` for primary SDK.", "inputs": "User-defined.", "outputs": "ScoreResult (name, value, reason, metadata).", "use_case": "Specific business logic."},
    {"name": "PII Detection (Demo)", "source": "Custom", "description": "Regex PII detection.", "inputs": "`text_output` (str)", "outputs": "Score (1.0 if PII found), details.", "use_case": "Basic PII scan."},
    {"name": "Response Word Count Check (Demo)", "source": "Custom", "description": "Heuristic word count range.", "inputs": "`text_output` (str), `min_words`, `max_words`", "outputs": "Score (1.0 if in range), count.", "use_case": "Brevity/detail check."},
]

# ----- Streamlit App UI -----
st.set_page_config(layout="wide", page_title="LLM Evaluation Metrics Demo")

# --- Display SDK Status Early in Sidebar ---
if 'sdk_status_displayed' not in st.session_state:
    if PRIMARY_SDK_AVAILABLE: st.sidebar.success("Primary SDK loaded!")
    else: st.sidebar.error("Primary SDK (e.g., Opik) not found. Built-in metrics disabled.")
    if RAG_LIB_SDK_AVAILABLE: st.sidebar.success("RAG Library SDK (e.g., Ragas) loaded!")
    else: st.sidebar.warning("RAG Library SDK not found. RAG-focused metrics disabled.")
    st.session_state.sdk_status_displayed = True

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ("Introduction", "User Guide", "Built-in Metrics Demo", "RAG-Focused Metrics Demo", "Custom Metrics Demo"))

# (Mock Data definitions would be here - using simplified versions from the last corrected code for brevity)
MOCK_DATA_QA = {"question": "Capital of France?", "llm_output_correct": "Paris.", "expected_output_exact": "Paris.", "llm_output_with_pii": "email me at test@test.com", "llm_output_unsafe": "unsafe content here", "long_output": "a very long response " * 20}
MOCK_DATA_RAG = {"question": "Eiffel Tower features?", "context_docs": ["The Eiffel Tower is tall.", "It has 3 levels."], "llm_answer_good": "Tall, 3 levels.", "ground_truth_answer_rag": "The Eiffel Tower in Paris is 330m tall and features three visitor levels." , "llm_answer_hallucination": "It is made of cheese."}
MOCK_DATA_TEXT2SQL = {"db_schema": "CREATE TABLE T(C1 INT);", "question_complex": "Complex SQL question?", "generated_sql_complex_correct": "SELECT C1 FROM T;", "sql_syntax_regex": r"^\s*SELECT.*;\s*$"}
MOCK_DATA_AGENTIC = {"llm_agent_output_json": '{"action": "tool", "input": "query"}', "g_eval_agent_task_intro": "Eval agent step.", "g_eval_agent_criteria": "Criteria for agent step."}


# --- Helper for Displaying Results (ensure this is correctly indented) ---
def display_metric_results_streamlit(metric_name, results, llm_response_details=None):
    st.subheader(f"Results for {metric_name}:")
    # ... (The rest of this function from the previous correctly indented version)
    if isinstance(results, dict): 
        score_key = 'score' if 'score' in results else 'value' 
        st.write(f"**Score:** `{results.get(score_key, 'N/A')}`") 
        if 'reason' in results and results['reason']:
            st.markdown(f"**Reason:** {results['reason']}")
        metadata_to_display = results.get('metadata', results.get('details'))
        if metadata_to_display:
            st.write("**Details/Metadata:**")
            try: st.json(metadata_to_display)
            except Exception: st.write(str(metadata_to_display))
        is_standard_format = any(k in results for k in ['score', 'value', 'reason', 'metadata', 'details'])
        if not is_standard_format and results and isinstance(results, dict) and len(results) > 0:
            st.markdown("**Full Results Dictionary:**"); st.json(results) 
        elif not results and not is_standard_format : 
             st.warning("No detailed results to display.")
    elif PRIMARY_SDK_AVAILABLE and isinstance(results, OpikScoreResult): 
        st.write(f"**Score:** `{results.value}`")
        if results.reason: st.markdown(f"**Reason:** {results.reason}")
        if results.metadata: st.write("**Metadata:"); st.json(results.metadata)
    else: st.write(str(results)) 
    if llm_response_details:
        st.markdown("**LLM Response Details (Input to Evaluation):**")
        resp_text, time_taken, i_tokens, o_tokens = llm_response_details
        st.text_area("LLM Output:", resp_text, height=100, disabled=True, key=f"disp_llm_out_{metric_name.replace(' ','_').replace('(','').replace(')','')}")
        st.caption(f"Time: {time_taken:.2f}s, Input Tokens: {i_tokens}, Output Tokens: {o_tokens}")

# ----- Shared UI Elements -----
st.sidebar.markdown("---")
st.sidebar.header("LLM Configuration")
primary_llm_model_input = st.sidebar.text_input("Primary LLM Model ID/Path:", "my-primary-llm-v1", help="Model ID for abc_response to generate answers.")
judge_llm_model_input = st.sidebar.text_input("Judge LLM Model ID/Path:", "local-hf-judge/model-name", help="Model ID for LLM-as-a-Judge metrics, passed to abc_response.")
st.sidebar.markdown("_(Ensure `abc_response` handles these model IDs/paths.)_")

# ----- App Mode Logic -----
if app_mode == "Introduction":
    st.title("LLM Evaluation Metrics Demo") # Renamed
    st.markdown("Welcome! This application demonstrates various evaluation metrics for LLM applications...")
    if not PRIMARY_SDK_AVAILABLE: st.error("Primary SDK not available.")
    if not RAG_LIB_SDK_AVAILABLE: st.warning("RAG-focused Library SDK not available.")

elif app_mode == "User Guide":
    st.title("User Guide: LLM Evaluation Metrics") # Renamed
    st.markdown("This guide details evaluation metrics, inputs, outputs, use cases, and categorizations by application type.")
    st.header("1. How to Use This Demo")
    st.markdown("1. Implement `abc_response`...\n (full instructions from previous version)")

    st.header("2. Metrics Catalog")
    st.subheader("2.1 Built-in Heuristic Metrics")
    for metric in heuristic_metrics_info_list:
        with st.expander(f"Heuristic: {metric['name']}"): # Renamed
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    st.subheader("2.2 Built-in LLM-as-a-Judge Metrics")
    st.markdown("These metrics use a 'judge' LLM (via `abc_response`).")
    for metric in llm_judge_metrics_info_list:
        with st.expander(f"LLM-as-a-Judge: {metric['name']}"): # Renamed
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")
    
    st.subheader("2.3 RAG-Focused Metrics (from Integrated Library)") # Renamed
    st.markdown("These metrics are specialized for RAG pipelines and use an underlying library (e.g., Ragas).")
    for metric in rag_focused_metrics_info_list: # Use the correct list name
        with st.expander(f"{metric['source']}: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs_lib']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    st.subheader("2.4 Custom Metrics")
    for metric in custom_metrics_info_list_guide: # Use the correct list name
        with st.expander(f"{metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    st.header("3. Metrics by Application Type")
    st.subheader("3.1 RAG (Retrieval Augmented Generation) Systems")
    st.markdown("""
    - **Core RAG Quality (Answer vs. Context):**
        - LLM-as-a-Judge: `Hallucination`, `ContextPrecision`, `ContextRecall` (Built-in SDK)
        - RAG Library: `Faithfulness`, `AnswerRelevancy`
    - **Retrieval Quality (Contexts vs. Question/Ground Truth):**
        - RAG Library: `ContextPrecision`, `ContextRecall`
    - **General Answer Quality:**
        - LLM-as-a-Judge: `AnswerRelevance`, `Usefulness` (Built-in SDK)
    """)
    st.subheader("3.2 Text2SQL")
    st.markdown("""
    - **SQL Correctness & Validity:**
        - Heuristic: `RegexMatch` (Built-in SDK, for basic syntax)
        - **LLM-as-a-Judge: `GEval` (Built-in SDK, Primary for semantic correctness)**
    - **Query Comparison (vs. reference SQL):**
        - Heuristic: `Equals` (Built-in SDK)
    """)
    st.subheader("3.3 Agentic Systems (Multi-step, Tool Use)")
    st.markdown("""
    - **Action & Tool Use Validation:**
        - Heuristic: `IsJson`, `Equals`, `Contains` (Built-in SDK)
    - **Reasoning & Planning Quality:**
        - **LLM-as-a-Judge: `GEval` (Built-in SDK, Primary for thoughts, plans)**
    - **Groundedness & Safety:**
        - LLM-as-a-Judge: `Hallucination`, `Moderation` (Built-in SDK)
    - **Final Output Quality:**
        - LLM-as-a-Judge: `AnswerRelevance`, `Usefulness` (Built-in SDK)
    """)

elif app_mode == "Built-in Metrics Demo": # Renamed
    if not PRIMARY_SDK_AVAILABLE:
        st.error("Primary SDK (e.g., Opik) is not available. Please install it to use this section.")
    else:
        st.title("Built-in Evaluation Metrics Demonstration")
        metric_type_selected = st.selectbox(
            "Select Metric Type:", ("Heuristic", "LLM-as-a-Judge"), key="builtin_type_select"
        )
        if metric_type_selected == "Heuristic":
            metric_name_selected = st.selectbox("Choose Heuristic Metric:", [m['name'] for m in heuristic_metrics_info_list], key="builtin_heur_select")
            metric_info = next(m for m in heuristic_metrics_info_list if m['name'] == metric_name_selected)
            # ... (UI and logic for heuristic metrics - ensure this part is correctly indented)
            # For brevity, assuming this section's detailed UI code is correct from prior versions,
            # applying the general fix pattern for class instantiation and method calls.
            # The critical fix was defining *_metrics_info_list globally and correcting dummy classes.
            st.markdown(f"**Selected Metric:** `{metric_info['name']}` - {metric_info['description']}")
            cols_h = st.columns(2)
            with cols_h[0]:
                st.subheader("Inputs")
                output_h = st.text_area("LLM Output:", MOCK_DATA_QA["llm_output_correct"], key=f"h_out_{metric_name_selected}")
                ref_h, pattern_h, substr_h, case_h = "", "", "", True
                if metric_name_selected in ["Equals", "LevenshteinRatio"]: ref_h = st.text_input("Reference:", MOCK_DATA_QA["expected_output_exact"], key=f"h_ref_{metric_name_selected}")
                elif metric_name_selected == "Contains": substr_h = st.text_input("Substring:", "Paris", key=f"h_sub_{metric_name_selected}"); case_h = st.checkbox("Case Sensitive?", True, key=f"h_case_{metric_name_selected}")
                elif metric_name_selected == "RegexMatch": pattern_h = st.text_input("Regex Pattern:", MOCK_DATA_TEXT2SQL["sql_syntax_regex"], key=f"h_pat_{metric_name_selected}")
                
                if st.button(f"Evaluate {metric_name_selected}", key=f"btn_h_{metric_name_selected}"):
                    try:
                        metric_cls_h = globals()[metric_name_selected.replace(" ", "")]()
                        if metric_name_selected == "Equals": res_h = metric_cls_h.score(output=output_h, reference=ref_h)
                        elif metric_name_selected == "Contains": res_h = metric_cls_h.score(output=output_h, substring=substr_h, case_sensitive=case_h)
                        elif metric_name_selected == "RegexMatch": res_h = metric_cls_h.score(output=output_h, regex_pattern=pattern_h)
                        elif metric_name_selected == "IsJson": res_h = metric_cls_h.score(output=output_h)
                        elif metric_name_selected == "LevenshteinRatio": res_h = metric_cls_h.score(output=output_h, reference=ref_h)
                        st.session_state[f"res_h_{metric_name_selected}"] = res_h.to_dict() if hasattr(res_h, 'to_dict') else res_h
                    except Exception as e: st.error(f"Error: {e}"); st.code(traceback.format_exc())
            with cols_h[1]:
                if f"res_h_{metric_name_selected}" in st.session_state:
                    display_metric_results_streamlit(metric_name_selected, st.session_state[f"res_h_{metric_name_selected}"])

        elif metric_type_selected == "LLM-as-a-Judge":
            metric_name_selected = st.selectbox("Choose LLM-as-a-Judge Metric:", [m['name'] for m in llm_judge_metrics_info_list], key="builtin_llmj_select")
            metric_info = next(m for m in llm_judge_metrics_info_list if m['name'] == metric_name_selected)
            # ... (UI and logic for LLM-as-a-Judge metrics - ensure correct indentation)
            st.markdown(f"**Selected Metric:** `{metric_info['name']}` - {metric_info['description']}")
            st.caption(f"Judge LLM: '{judge_llm_model_input}' (via `abc_response`).")
            cols_lj = st.columns(2)
            output_lj, details_lj = "", None
            with cols_lj[0]:
                st.subheader("Inputs & Primary LLM Output Gen:")
                # Simplified common inputs for demo brevity; specific inputs were more detailed before
                query_lj = st.text_input("User Question/Input:", MOCK_DATA_QA["question"], key=f"lj_q_{metric_name_selected}")
                context_lj = st.text_area("Context (if applicable):", MOCK_DATA_RAG["context_docs"][0] if "Context" in metric_name_selected or "Hallucination" in metric_name_selected else "", height=70, key=f"lj_ctx_{metric_name_selected}")
                task_intro_lj, eval_crit_lj = "", ""
                if metric_name_selected == "G-Eval":
                    task_intro_lj = st.text_area("G-Eval Task Intro:", MOCK_DATA_AGENTIC["g_eval_agent_task_intro"], height=70, key=f"lj_geval_intro_{metric_name_selected}")
                    eval_crit_lj = st.text_area("G-Eval Criteria:", MOCK_DATA_AGENTIC["g_eval_agent_criteria"], height=100, key=f"lj_geval_crit_{metric_name_selected}")

                prompt_primary_lj = query_lj
                if context_lj: prompt_primary_lj = f"Context: {context_lj}\nQuestion: {query_lj}"
                if metric_name_selected == "G-Eval": prompt_primary_lj = f"Output for G-Eval based on: {task_intro_lj}" # Output for G-Eval is what's judged
                
                if metric_name_selected != "Moderation" and st.button(f"Generate Output via '{primary_llm_model_input}'", key=f"btn_gen_lj_{metric_name_selected}"):
                    st.session_state[f"details_lj_{metric_name_selected}"] = abc_response(primary_llm_model_input, prompt_primary_lj)

                if f"details_lj_{metric_name_selected}" in st.session_state:
                    details_lj = st.session_state[f"details_lj_{metric_name_selected}"]
                    output_lj = details_lj[0]
                
                output_lj = st.text_area("LLM Output to Evaluate:", output_lj or (MOCK_DATA_QA["llm_output_unsafe"] if metric_name_selected == "Moderation" else MOCK_DATA_QA["llm_output_correct"]), height=100, key=f"lj_out_{metric_name_selected}")
                if details_lj and output_lj == details_lj[0]: st.caption(f"Gen Time: {details_lj[1]:.2f}s")
                elif not details_lj and output_lj : details_lj = (output_lj,0,0,0)


                if st.button(f"Evaluate {metric_name_selected}", key=f"btn_eval_lj_{metric_name_selected}"):
                    if not output_lj: st.warning("Need output to evaluate.")
                    else:
                        try:
                            metric_cls_lj_actual = globals()[metric_name_selected.replace(" ", "")]
                            score_args_lj = {"output": output_lj}
                            if metric_name_selected == "G-Eval": metric_instance_lj = metric_cls_lj_actual(task_introduction=task_intro_lj, evaluation_criteria=eval_crit_lj, model=judge_llm_model_input)
                            else: metric_instance_lj = metric_cls_lj_actual(model=judge_llm_model_input)
                            
                            if metric_name_selected not in ["Moderation", "G-Eval"]: score_args_lj["input"] = query_lj
                            if metric_name_selected in ["Hallucination", "ContextRecall", "ContextPrecision"]: score_args_lj["context"] = context_lj
                            if metric_name_selected == "ContextPrecision": score_args_lj["expected_output"] = st.session_state.get(f"lj_exp_{metric_name_selected}", "") # Assuming an input for expected_output

                            res_lj = metric_instance_lj.score(**score_args_lj)
                            st.session_state[f"res_lj_{metric_name_selected}"] = res_lj.to_dict() if hasattr(res_lj, 'to_dict') else res_lj
                        except Exception as e: st.error(f"Error: {e}"); st.code(traceback.format_exc())
            with cols_lj[1]:
                if f"res_lj_{metric_name_selected}" in st.session_state:
                    display_metric_results_streamlit(metric_name_selected, st.session_state[f"res_lj_{metric_name_selected}"], details_lj)


elif app_mode == "RAG-Focused Metrics Demo": # Renamed
    if not RAG_LIB_SDK_AVAILABLE:
        st.error("RAG-focused Library SDK (e.g., Ragas) not available. Install relevant packages.")
    elif not PRIMARY_SDK_AVAILABLE: # For abc_response
        st.error("Primary SDK for `abc_response` wrapper is needed for this demo.")
    else:
        st.title("RAG-Focused Metrics Demonstration")
        # ... (Ragas Demo section - ensure indentation and logic are correct)
        if not ragas_llm_wrapper_for_demo:
            st.warning("Ragas LLM wrapper (for judge) could not be initialized. LLM-based Ragas metrics might fail.")
        
        rag_metric_name = st.selectbox("Choose RAG-focused Metric:", [m['name'] for m in rag_focused_metrics_info_list], key="rag_metric_select_main")
        metric_info_rag = next(m for m in rag_focused_metrics_info_list if m['name'] == rag_metric_name)
        st.markdown(f"**Selected Metric ({metric_info_rag['source']}):** `{metric_info_rag['name']}` - {metric_info_rag['description']}")

        cols_rag = st.columns(2)
        with cols_rag[0]:
            st.subheader("RAG Inputs:")
            q_rag = st.text_input("Question:", MOCK_DATA_RAG["question"], key="rag_q_main")
            ctx_rag = [st.text_area(f"Context Doc {i+1}:", MOCK_DATA_RAG["context_docs"][i], height=60, key=f"rag_ctx_main_{i}") for i in range(len(MOCK_DATA_RAG["context_docs"]))]
            gt_rag = ""
            if rag_metric_name == "ContextRecall": # Ragas ContextRecall needs ground_truth
                gt_rag = st.text_area("Ground Truth Answer:", MOCK_DATA_RAG["ground_truth_answer_rag"], height=60, key="rag_gt_main")

            st.markdown(f"**1. Generate Answer via '{primary_llm_model_input}':**")
            if st.button("Generate RAG Answer", key="btn_rag_gen_main"):
                prompt_rag_gen = f"Contexts:\n{'---'.join(ctx_rag)}\n\nQuestion: {q_rag}"
                st.session_state.rag_ans_details_main = abc_response(primary_llm_model_input, prompt_rag_gen)

            ans_rag, details_ans_rag = "", None
            if "rag_ans_details_main" in st.session_state:
                details_ans_rag = st.session_state.rag_ans_details_main; ans_rag = details_ans_rag[0]
            ans_rag = st.text_area("Generated RAG Answer:", ans_rag or MOCK_DATA_RAG["llm_answer_good"], height=100, key="rag_ans_main")
            if details_ans_rag and ans_rag == details_ans_rag[0]: st.caption(f"Gen Time: {details_ans_rag[1]:.2f}s")
            elif not details_ans_rag and ans_rag: details_ans_rag = (ans_rag,0,0,0)


            if st.button(f"Evaluate RAG {rag_metric_name}", key=f"btn_eval_rag_main_{rag_metric_name}"):
                if not ans_rag: st.warning("Need generated answer.")
                elif not ragas_llm_wrapper_for_demo and rag_metric_name in ["Faithfulness", "AnswerRelevancy", "ContextRecall"]:
                    st.error("RAG Library LLM wrapper for judge not ready.")
                else:
                    data_dict_for_rag_lib = {"question": [q_rag], "answer": [ans_rag], "contexts": [ctx_rag]}
                    if gt_rag and rag_metric_name == "ContextRecall": data_dict_for_rag_lib["ground_truth"] = [gt_rag]
                    
                    try:
                        dataset_hf_for_rag_lib = HuggingFaceDataset.from_dict(data_dict_for_rag_lib)
                        metric_instance_for_rag_lib = None
                        if rag_metric_name == "Faithfulness": metric_instance_for_rag_lib = ragas_faithfulness_metric(llm=ragas_llm_wrapper_for_demo)
                        elif rag_metric_name == "AnswerRelevancy": metric_instance_for_rag_lib = ragas_answer_relevancy_metric(llm=ragas_llm_wrapper_for_demo)
                        elif rag_metric_name == "ContextPrecision": metric_instance_for_rag_lib = ragas_context_precision_metric(llm=ragas_llm_wrapper_for_demo)
                        elif rag_metric_name == "ContextRecall": 
                            if "ground_truth" not in data_dict_for_rag_lib: st.error("Ground truth needed for RAG Library ContextRecall."); st.stop()
                            metric_instance_for_rag_lib = ragas_context_recall_metric(llm=ragas_llm_wrapper_for_demo)

                        if metric_instance_for_rag_lib:
                            with st.spinner(f"RAG Library '{rag_metric_name}' running..."):
                                eval_res_rag_lib = ragas_evaluate(dataset_hf_for_rag_lib, metrics=[metric_instance_for_rag_lib])
                            st.session_state[f"rag_res_main_{rag_metric_name}"] = eval_res_rag_lib.to_pandas().to_dict(orient='records')[0] if eval_res_rag_lib and not eval_res_rag_lib.to_pandas().empty else {"error": "RAG Library returned empty."}
                        else: st.error("RAG Library metric not configured.")
                    except Exception as e: st.error(f"RAG Library Error: {e}"); st.code(traceback.format_exc())
        with cols_rag[1]:
            if f"rag_res_main_{rag_metric_name}" in st.session_state:
                display_metric_results_streamlit(f"RAG Library {rag_metric_name}", st.session_state[f"rag_res_main_{rag_metric_name}"], details_ans_rag)


elif app_mode == "Custom Metrics Demo":
    if not PRIMARY_SDK_AVAILABLE:
        st.error("Primary SDK (e.g., Opik) is required for custom metrics base classes. Please install it.")
    else:
        st.title("Custom Evaluation Metrics Demonstration")
        # ... (Custom Metrics Demo section - ensure indentation is correct)
        custom_metric_choice_main = st.selectbox("Choose Custom Metric:", ("PII Detection (Demo)", "Response Word Count Check (Demo)"), key="custom_select_main_v2")
        cols_cust_main_v2 = st.columns(2)
        with cols_cust_main_v2[0]:
            st.subheader("Inputs:")
            metric_info_cust_main = None
            if custom_metric_choice_main == "PII Detection (Demo)":
                metric_info_cust_main = next(m for m in custom_metrics_info_list_guide if m['name'] == "PII Detection (Custom Demo)")
                st.markdown(f"**Metric:** `{metric_info_cust_main['name']}` - {metric_info_cust_main['description']}")
                text_pii_main = st.text_area("Text to Scan for PII:", MOCK_DATA_QA["llm_output_with_pii"], height=100, key="custom_pii_text_main_v2")
                if st.button("Evaluate PII Detection", key="btn_custom_pii_main_v2"):
                    metric_pii_instance = CustomPIIDetectionMetric(); st.session_state.custom_pii_res_main_v2 = metric_pii_instance.score(text_output=text_pii_main)
            elif custom_metric_choice_val_main == "Response Word Count Check (Demo)":
                metric_info_cust_main = next(m for m in custom_metrics_info_list_guide if m['name'] == "Response Word Count Check (Custom Demo)")
                st.markdown(f"**Metric:** `{metric_info_cust_main['name']}` - {metric_info_cust_main['description']}")
                text_len_main = st.text_area("Text for Length Check:", MOCK_DATA_QA["long_output"], height=100, key="custom_len_text_main_v2")
                min_w_main_v2 = st.number_input("Min Words:", 1, 1000, 10, key="cust_minw_main_v2")
                max_w_main_v2 = st.number_input("Max Words:", min_w_main_v2 + 1 if min_w_main_v2 else 2, 2000, 100, key="cust_maxw_main_v2")
                if st.button("Evaluate Length Check", key="btn_custom_len_main_v2"):
                    metric_len_instance = CustomResponseLengthCheck(min_words=min_w_main_v2, max_words=max_w_main_v2); st.session_state.custom_len_res_main_v2 = metric_len_instance.score(text_output=text_len_main)
        with cols_cust_main_v2[1]:
            if custom_metric_choice_main == "PII Detection (Demo)" and 'custom_pii_res_main_v2' in st.session_state:
                display_metric_results_streamlit("Custom PII Detection", st.session_state.custom_pii_res_main_v2)
            elif custom_metric_choice_main == "Response Word Count Check (Demo)" and 'custom_len_res_main_v2' in st.session_state:
                display_metric_results_streamlit("Custom Word Count Check", st.session_state.custom_len_res_main_v2)

# Fallback for SDK not available on main page selection
elif (app_mode == "Built-in Metrics Demo" and not PRIMARY_SDK_AVAILABLE) or \
     (app_mode == "RAG-Focused Metrics Demo" and (not RAG_LIB_SDK_AVAILABLE or not PRIMARY_SDK_AVAILABLE)) or \
     (app_mode == "Custom Metrics Demo" and not PRIMARY_SDK_AVAILABLE):
    st.error(f"Required SDK(s) for '{app_mode}' are not available. Please check installation messages in the sidebar and install necessary packages.")
