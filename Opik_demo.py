import streamlit as st
import re
import json
from typing import Any, Dict, List, Optional, Tuple # For type hinting

# --- Opik SDK Imports ---
try:
    from opik.evaluation.metrics import (
        Equals, Contains, RegexMatch, IsJson, Levenshtein,
        Hallucination, GEval, Moderation, AnswerRelevance, Usefulness,
        ContextRecall, ContextPrecision
    )
    from opik.evaluation.metrics.base_metric import BaseMetric as OpikBaseMetric
    from opik.evaluation.metrics.score_result import ScoreResult as OpikScoreResult
    from opik.evaluation import models as opik_models # For LiteLLMChatModel
    OPIK_SDK_AVAILABLE = True
    st.sidebar.success("Opik SDK loaded successfully!")
except ImportError as e:
    st.sidebar.error(f"Opik SDK not found or import error: {e}. Please install opik (`pip install opik`). This demo will not run Opik metrics.")
    OPIK_SDK_AVAILABLE = False
    # Define dummy classes if Opik is not available so the rest of the UI can load
    class OpikBaseMetric: pass
    class OpikScoreResult:
        def __init__(self, name, value, reason=None, metadata=None):
            self.name = name
            self.value = value
            self.reason = reason
            self.metadata = metadata
        def to_dict(self):
            return {"name": self.name, "score": self.value, "reason": self.reason, "metadata": self.metadata} # Match Opik's ScoreResult more closely
    # Define dummy Opik metric classes
    class Equals: def score(self, **kwargs): return OpikScoreResult("Equals (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Contains: def score(self, **kwargs): return OpikScoreResult("Contains (Unavailable)", 0, "Opik SDK not found").to_dict()
    class RegexMatch: def score(self, **kwargs): return OpikScoreResult("RegexMatch (Unavailable)", 0, "Opik SDK not found").to_dict()
    class IsJson: def score(self, **kwargs): return OpikScoreResult("IsJson (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Levenshtein: def score(self, **kwargs): return OpikScoreResult("Levenshtein (Unavailable)", 0, "Opik SDK not found").to_dict()
    class Hallucination: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"Hallucination (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class GEval:
        def __init__(self, task_introduction, evaluation_criteria, model=None):
            self.task_introduction = task_introduction
            self.evaluation_criteria = evaluation_criteria
            self.model = model
        def score(self, **kwargs): return OpikScoreResult(f"GEval (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class Moderation: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"Moderation (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class AnswerRelevance: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"AnswerRelevance (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class Usefulness: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"Usefulness (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class ContextRecall: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"ContextRecall (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()
    class ContextPrecision: def __init__(self, model=None): self.model=model; def score(self, **kwargs): return OpikScoreResult(f"ContextPrecision (Unavailable - Judge: {self.model})", 0, "Opik SDK not found").to_dict()


# --- Ragas SDK Imports ---
RAGAS_SDK_AVAILABLE = False
if OPIK_SDK_AVAILABLE: # Ragas often used alongside
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness as ragas_faithfulness,
            answer_relevancy as ragas_answer_relevancy,
            context_precision as ragas_context_precision,
            context_recall as ragas_context_recall,
            # answer_correctness as ragas_answer_correctness, # Requires ground_truth
            # answer_similarity as ragas_answer_similarity, # Requires ground_truth
        )
        from datasets import Dataset as HuggingFaceDataset # Ragas uses Hugging Face datasets
        from ragas.llms import LangchainLLMWrapper # To wrap abc_response for Ragas
        from langchain_core.language_models.llms import LLM as LangchainLLMBase # For custom LLM
        RAGAS_SDK_AVAILABLE = True
        st.sidebar.success("Ragas SDK loaded successfully!")
    except ImportError as e:
        st.sidebar.warning(f"Ragas SDK not found or import error: {e}. Please install ragas (`pip install ragas`). Ragas metrics demo will be limited.")
        # Define dummy classes if Ragas is not available
        class ragas_faithfulness: pass
        class ragas_answer_relevancy: pass
        class ragas_context_precision: pass
        class ragas_context_recall: pass

# --- User-Provided LLM Function (Placeholder) ---
# USER: Define your abc_response function here or import it.
# This function is CRITICAL for the demo to work with actual LLM judging.
# It should handle model identifiers for both primary tasks and judge LLMs,
# potentially loading local Hugging Face models based on path/name.

def abc_response(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Tuple[str, float, int, int]:
    """
    Placeholder for your actual LLM call.
    Args:
        model (str): The model name/identifier (e.g., "my-primary-llm", "local:/path/to/hf-judge-model", "openai/gpt-4o").
        prompt (str): The prompt to send to the LLM.
        temperature (float): The temperature for generation.
        max_tokens (int): Max tokens to generate.
    Returns:
        tuple: (response_text: str, time_taken: float, input_tokens: int, output_tokens: int)
    """
    st.sidebar.write(f"Mock LLM Call: Model='{model}', Prompt='{prompt[:70]}...'")
    # Simulate response based on model type or prompt content for demo
    import time
    import random

    time.sleep(random.uniform(0.5, 1.5)) # Simulate delay
    input_tok = len(prompt.split())
    
    # Default response
    response_text = f"This is a dynamic mock response for model '{model}'. Input started with: '{prompt.splitlines()[0][:50]}...'. The query seems to be about evaluating LLM outputs."
    
    # Specific mock responses for judge LLMs (Opik/Ragas might expect JSON from judge)
    if "judge" in model.lower() or "gpt-4o" in model.lower() or "claude" in model.lower() or "gemini" in model.lower() or "hf-judge" in model.lower():
        # Opik metrics often expect a JSON string with 'score' and 'reason'
        # Ragas metrics might also use LLMs that return structured or unstructured text parsed later
        if "hallucination" in prompt.lower() or "faithfulness" in prompt.lower() :
            # Opik Hallucination: score 0 (faithful), 1 (hallucinated)
            # Ragas Faithfulness: score 0-1 (higher is more faithful)
            score = random.choice([0.0, 0.0, 0.0, 1.0]) if "hallucination" in prompt.lower() else round(random.uniform(0.6, 1.0), 2)
            reason = "The output aligns well with the context." if score < 0.5 else "The output contains statements not supported by the context."
            response_text = json.dumps({"score": score, "reason": reason})
        elif "relevance" in prompt.lower() or "pertinent" in prompt.lower():
            score = round(random.uniform(0.5, 1.0), 2)
            reason = "The answer is highly relevant to the question." if score > 0.7 else "The answer is somewhat relevant but could be more focused."
            response_text = json.dumps({"score": score, "reason": reason})
        elif "moderation" in prompt.lower():
            # Opik Moderation: score 0 (safe) to 1 (unsafe)
            score = round(random.uniform(0.0, 0.4), 2) # Mostly safe for demo
            reason = "Content appears safe and appropriate." if score < 0.5 else "Content has potential moderation concerns."
            response_text = json.dumps({"score": score, "reason": reason})
        elif "g-eval" in model.lower() or "task_introduction" in prompt.lower() or "evaluation_criteria" in prompt.lower():
            # Opik GEval expects an integer score internally (often from LLM), then normalizes.
            # Let's mock the direct score (0-10) and a reason.
            raw_score = random.randint(6, 10)
            reason = f"CoT: Step 1 (Clarity): Good. Step 2 (Coverage): Excellent. Step 3 (Conciseness): Okay. Overall score {raw_score}/10."
            response_text = json.dumps({"score": raw_score, "reason": reason}) # Opik's GEval will process this
        elif "context_precision" in prompt.lower() or "context_recall" in prompt.lower():
            score = round(random.uniform(0.6, 1.0), 2)
            reason = "The context was used appropriately and covered relevant aspects."
            response_text = json.dumps({"score": score, "reason": reason})
        elif "useful" in prompt.lower():
            score = round(random.uniform(0.5, 0.95),2)
            reason = "The answer provides actionable and helpful information."
            response_text = json.dumps({"score": score, "reason": reason})
        else: # Generic judge response
            score = round(random.uniform(0.6, 0.9), 2)
            reason = "The evaluation is generally positive based on the criteria."
            response_text = json.dumps({"score": score, "reason": reason})

    elif "text2sql" in model.lower():
        if "employees in Sales hired after 2022" in prompt.lower():
            response_text = "SELECT EmployeeID, FirstName, LastName, HireDate FROM Employees WHERE Department = 'Sales' AND HireDate > '2022-12-31';"
        elif "total revenue per product" in prompt.lower():
            response_text = "SELECT ProductName, SUM(OrderAmount) AS TotalRevenue FROM Orders JOIN Products ON Orders.ProductID = Products.ProductID GROUP BY ProductName;"
        else:
            response_text = "SELECT COUNT(*) FROM Customers WHERE Country = 'USA';"

    elif "rag-primary" in model.lower(): # For generating answer in RAG
        if "Eiffel Tower features" in prompt.lower():
            response_text = "The Eiffel Tower, located in Paris, is 330m tall and features three visitor levels, restaurants, and an observation deck. It was designed by Gustave Eiffel for the 1889 World's Fair."
        else:
            response_text = "Based on the context, the main point is X, and supporting details include Y and Z."

    output_tok = len(response_text.split())
    return response_text, random.uniform(0.5, 2.5), input_tok, output_tok


# --- Langchain LLM Wrapper for Ragas (using abc_response) ---
if RAGAS_SDK_AVAILABLE:
    class RagasCustomLLM(LangchainLLMBase):
        model_name: str = "ragas-judge-llm" # Default, can be overridden

        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            # abc_response returns a tuple, Ragas LLM needs just the response string
            response_text, _, _, _ = abc_response(self.model_name, prompt)
            # Ragas metrics might parse JSON from this or use it directly.
            # Some Ragas judge prompts expect specific phrasing like "YES" or "NO" or a short explanation.
            # The abc_response mock for judge should be aware of this if possible.
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
        if OPIK_SDK_AVAILABLE:
            super().__init__(name=self.name)

    def score(self, text_output: str, **ignored_kwargs) -> Dict: # Opik expects ScoreResult or dict
        detected_pii_details = {}
        pii_found_count = 0
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text_output)
            if matches:
                detected_pii_details[pii_type] = matches
                pii_found_count += len(matches)
        
        score_value = 1.0 if pii_found_count > 0 else 0.0 # 1.0 if PII detected
        reason_str = f"Detected {pii_found_count} PII instances. {'Details: ' + json.dumps(detected_pii_details) if pii_found_count > 0 else 'No PII found.'}"
        
        if OPIK_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"details": detected_pii_details}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"details": detected_pii_details}}

class CustomResponseLengthCheck(OpikBaseMetric if OPIK_SDK_AVAILABLE else object):
    def __init__(self, name="Response Word Count Check (Custom)", min_words=5, max_words=150):
        self.name = name
        self.min_words = min_words
        self.max_words = max_words
        if OPIK_SDK_AVAILABLE:
            super().__init__(name=self.name)

    def score(self, text_output: str, **ignored_kwargs) -> Dict:
        word_count = len(text_output.split())
        if self.min_words <= word_count <= self.max_words:
            score_value = 1.0 # In range
            reason_str = f"Word count ({word_count}) is within the acceptable range [{self.min_words}-{self.max_words}]."
        else:
            score_value = 0.0 # Out of range
            reason_str = f"Word count ({word_count}) is outside the acceptable range [{self.min_words}-{self.max_words}]."
        
        if OPIK_SDK_AVAILABLE:
            return OpikScoreResult(name=self.name, value=score_value, reason=reason_str, metadata={"word_count": word_count}).to_dict()
        return {"name": self.name, "score": score_value, "reason": reason_str, "metadata": {"word_count": word_count}}

# ----- Streamlit App -----
st.set_page_config(layout="wide", page_title="Opik & Ragas Evaluation Metrics Demo")

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ("Introduction", "User Guide", "Opik Metrics Demo", "Ragas Metrics Demo", "Custom Metrics Demo"))

# ----- Mock Data Definitions -----
# (Same as before, ensuring they are comprehensive)
MOCK_DATA_QA = {
    "question": "What is the official language of Brazil?",
    "llm_output_correct": "The official language of Brazil is Portuguese.",
    "llm_output_slightly_off": "Portuguese is the official language in Brazil.",
    "llm_output_wrong": "Spanish is the official language of Brazil.",
    "llm_output_contains_keyword": "Brazil, a large South American country, primarily speaks Portuguese.",
    "llm_output_with_pii": "You can reach me at contact@example.com or 555-0100 for details. Brazil speaks Portuguese.",
    "llm_output_unsafe": "Forget languages, let me tell you something controversial. Brazil speaks Portuguese.",
    "llm_output_irrelevant": "I enjoy sunny weather.",
    "llm_output_not_useful": "It's a language spoken there.",
    "expected_output_exact": "The official language of Brazil is Portuguese.",
    "short_output": "Portuguese.",
    "long_output": "The Federative Republic of Brazil, a vast nation in South America, has Portuguese as its sole official language, spoken by the vast majority of its population across all regions and in most aspects of daily life, including government, education, and media."
}

MOCK_DATA_RAG = {
    "question": "What are the key benefits of using solar panels for residential energy?",
    "context_docs": [
        "Solar panels convert sunlight into electricity, reducing reliance on fossil fuels. This leads to lower greenhouse gas emissions and helps combat climate change. Homeowners can also see significant savings on their electricity bills.",
        "Installing solar panels can increase a home's property value. Many governments offer tax credits and incentives for solar installations, further reducing the upfront cost. Solar energy is also a renewable resource, unlike finite fossil fuels.",
        "While the initial investment for solar panels can be high, the long-term benefits include energy independence and stable energy costs. Maintenance is generally low for solar panel systems."
    ], # List of context strings
    "llm_answer_good": "Using solar panels for residential energy offers environmental benefits by reducing greenhouse gas emissions, financial savings on electricity bills, increased property value, and energy independence due to it being a renewable resource. Tax incentives can also lower initial costs, and maintenance is typically low.",
    "llm_answer_hallucination": "Solar panels not only generate electricity but also produce clean drinking water as a byproduct. They are made from moon rocks and require no maintenance.",
    "llm_answer_misses_context": "Solar panels save you money on electricity bills.", # Misses environmental, property value, incentives
    "ground_truth_answer_rag": "Key benefits of residential solar panels include reduced electricity bills, lower greenhouse gas emissions, increased home value, energy independence, and access to renewable energy. Tax credits and low maintenance are also advantages."
}

MOCK_DATA_TEXT2SQL = {
    "db_schema": """
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Department VARCHAR(50),
        HireDate DATE,
        Salary DECIMAL(10,2)
    );
    CREATE TABLE Departments (
        DepartmentID INT PRIMARY KEY,
        DepartmentName VARCHAR(50) UNIQUE
    );
    -- Assume Employees.Department refers to Departments.DepartmentName for simplicity in NL
    """,
    "question_simple": "List all employees in the 'Engineering' department.",
    "generated_sql_simple_correct": "SELECT EmployeeID, FirstName, LastName FROM Employees WHERE Department = 'Engineering';",
    "expected_sql_simple": "SELECT EmployeeID, FirstName, LastName FROM Employees WHERE Department = 'Engineering';",

    "question_complex": "What is the average salary of employees hired in 2023 for each department?",
    "generated_sql_complex_correct": "SELECT Department, AVG(Salary) AS AverageSalary FROM Employees WHERE STRFTIME('%Y', HireDate) = '2023' GROUP BY Department;",
    "generated_sql_complex_error": "SELECT Department, Salary FROM Employees WHERE HireDate = '2023';", # Incorrect aggregation
    "expected_sql_complex": "SELECT Department, AVG(Salary) AS AverageSalary FROM Employees WHERE STRFTIME('%Y', HireDate) = '2023' GROUP BY Department;"
}

MOCK_DATA_AGENTIC = {
    "task_input": "Find a recipe for vegan pasta and check if I have all ingredients in my pantry (milk, flour, tomatoes, pasta, spinach).",
    "llm_agent_output_json": """
    {
        "thought": "User wants a vegan pasta recipe and an inventory check. I should first find a recipe, then list its ingredients, then compare with the user's pantry.",
        "action": "find_recipe",
        "action_input": {
            "dish_name": "vegan pasta",
            "dietary_restrictions": ["vegan"]
        },
        "next_step_suggestion": "Parse recipe ingredients and then perform pantry check."
    }
    """,
    "llm_agent_output_not_json": "Okay, I will find a vegan pasta recipe first. Then I'll look at your pantry list: milk, flour, tomatoes, pasta, spinach.",
    "g_eval_agent_task_intro": "You are evaluating an AI agent's reasoning and action planning for a multi-step task.",
    "g_eval_agent_criteria": """
    1. Thought Quality: Is the agent's thought process logical, comprehensive for the user's request, and does it outline a sensible plan? (Score 0-10)
    2. Action Validity: Is the chosen 'action' appropriate and correctly formatted for the current step? (Score 0-10)
    3. Action Input Correctness: Are the 'action_input' parameters correct and sufficient for the chosen action? (Score 0-10)
    Return a JSON with 'score' (average of sub-scores, normalized 0-1) and 'reason' (your detailed CoT).
    """
}

# --- Helper to display metric results ---
def display_metric_results_streamlit(metric_name, results, llm_response_details=None):
    st.subheader(f"Results for {metric_name}:")
    if isinstance(results, dict): # Opik's ScoreResult.to_dict() or our custom metric dicts
        st.write(f"**Score:** `{results.get('score', results.get('value', 'N/A'))}`") # Opik uses 'value' in ScoreResult
        if 'reason' in results and results['reason']:
            st.markdown(f"**Reason:** {results['reason']}")
        if 'metadata' in results and results['metadata']:
            st.write("**Metadata:**")
            st.json(results['metadata'])
        elif 'details' in results and results['details']: # For custom PII
            st.write("**Details:**")
            st.json(results['details'])
        # For Ragas results, which are often dicts of metric_name: score_value
        elif not any(k in results for k in ['score', 'value', 'reason', 'metadata', 'details']) and len(results) > 0:
             st.json(results) # Likely a Ragas result dict
        elif not results:
             st.warning("No results to display (empty dictionary).")

    elif isinstance(results, OpikScoreResult): # If we get the raw Opik object
        st.write(f"**Score:** `{results.value}`")
        if results.reason:
            st.markdown(f"**Reason:** {results.reason}")
        if results.metadata:
            st.write("**Metadata:**")
            st.json(results.metadata)
    else:
        st.write(results) # Fallback for other types

    if llm_response_details:
        st.write("**LLM Response Details (Evaluated):**")
        resp_text, time_taken, i_tokens, o_tokens = llm_response_details
        st.text_area("LLM Output:", resp_text, height=100, disabled=True, key=f"disp_llm_out_{metric_name.replace(' ','_')}")
        st.caption(f"Time: {time_taken:.2f}s, Input Tokens: {i_tokens}, Output Tokens: {o_tokens}")


# ----- Shared UI Elements -----
st.sidebar.markdown("---")
st.sidebar.header("LLM Configuration")
primary_llm_model_input = st.sidebar.text_input("Primary LLM Model ID/Path:", "my-primary-llm-v1", help="Model ID for abc_response to generate answers.")
judge_llm_model_input = st.sidebar.text_input("Judge LLM Model ID/Path:", "local-judge-model-path/my-judge", help="Model ID for Opik/Ragas LLM-as-a-Judge metrics, passed to abc_response.")
st.sidebar.markdown("_(Ensure your `abc_response` function can handle these model IDs/paths.)_")


# ----- App Mode Logic -----
if app_mode == "Introduction":
    st.title("Opik & Ragas Evaluation Metrics Demo")
    st.markdown("""
    Welcome! This application demonstrates various evaluation metrics from the **Opik SDK** and **Ragas library**.
    Our goal is to show how these metrics can be used to assess LLM applications.

    **Key Features of this Demo:**
    - **Real SDK Usage:** We use actual metric classes from Opik and Ragas (if installed).
    - **Mocked Inputs:** The contexts, questions, and reference answers are mocked for demonstration.
    - **`abc_response` Placeholder:** All LLM calls (for generating responses to evaluate AND for LLM-as-a-Judge metrics) are routed through a placeholder function `abc_response(model, prompt)`. **You must implement this function to connect to your desired LLMs (local Hugging Face models, APIs, etc.).**
    - **Local Calculation:** Metric scores are calculated locally by the SDKs. No Opik backend platform is assumed to be running for *this demo app*.

    **Navigate using the sidebar to:**
    - **User Guide:** Detailed explanations of each metric.
    - **Opik Metrics Demo:** Interactive demos for Opik's built-in metrics.
    - **Ragas Metrics Demo:** Interactive demos for key Ragas metrics (requires `ragas` installation).
    - **Custom Metrics Demo:** Examples of user-defined evaluation metrics.
    """)
    if not OPIK_SDK_AVAILABLE:
        st.error("Opik SDK is not available. Opik metric demos will be non-functional.")
    if not RAGAS_SDK_AVAILABLE:
        st.warning("Ragas SDK is not available or has import issues. Ragas metric demos will be limited or non-functional.")

# --- Section: User Guide ---
elif app_mode == "User Guide":
    st.title("User Guide: Opik & Ragas Evaluation Metrics")
    # (Content from previous response, updated for Opik + Ragas and real SDK usage)
    st.markdown("""
    This guide provides detailed information about the evaluation metrics available through Opik and its integration with Ragas.
    It covers how they work, their typical inputs/outputs, and their use cases.
    All metrics demonstrated in this app use their respective SDKs for calculation.
    LLM-as-a-Judge metrics (from both Opik and Ragas) rely on the `abc_response` function to provide judge LLM capabilities.
    """)

    st.header("1. How to Use This Demo")
    st.markdown("""
    1.  **Implement `abc_response`:** Before diving deep, ensure the `abc_response(model, prompt)` function in the script is correctly implemented to call your LLMs. This is crucial for LLM-as-a-Judge metrics to function meaningfully.
    2.  **Select a Demo Section:** Use the sidebar to navigate to "Opik Metrics Demo", "Ragas Metrics Demo", or "Custom Metrics Demo".
    3.  **Configure LLM Models (Sidebar):** Specify the model identifiers/paths for your primary LLM and your judge LLM. Your `abc_response` function should be able to use these.
    4.  **Choose a Specific Metric:** Select the metric you want to explore.
    5.  **Review/Edit Input Data:** The demo provides mock data relevant to the selected metric. You can modify these inputs.
    6.  **Generate LLM Output (If Applicable):** For many scenarios, you'll first provide a prompt/context and click "Generate LLM Output". This uses `abc_response` with your "Primary LLM Model" to get text for evaluation.
    7.  **Evaluate:** Click the "Evaluate with [Metric Name]" button.
    8.  **View Results:** The metric's score and reasoning will be displayed. For LLM-as-a-Judge metrics, this involves a call to `abc_response` using your "Judge LLM Model".
    """)

    st.header("2. Opik Built-in Metrics")
    # ... (Detailed explanations for each Opik Heuristic and LLM-as-a-Judge metric as before) ...
    # Make sure to emphasize that Opik's LLM-as-a-Judge metrics can take a `model` parameter.
    # Opik Heuristic Metrics
    opik_heuristic_metrics_info = [
        {"name": "Equals", "description": "Checks if the LLM's output string *exactly matches* an expected target string.", "inputs": "`output` (str), `expected_output` (str)", "outputs": "Score (1.0 if equal, 0.0 if not), reason.", "use_case": "Verifying precise outputs, known correct answers."},
        {"name": "Contains", "description": "Checks if the LLM's output string *includes a specific substring*. Can be case-sensitive or insensitive.", "inputs": "`output` (str), `substring` (str), `case_sensitive` (bool, optional)", "outputs": "Score (1.0 if contains, 0.0 if not), reason.", "use_case": "Ensuring keywords or required phrases are present."},
        {"name": "RegexMatch", "description": "Checks if the LLM's output *matches a specified regular expression pattern*.", "inputs": "`output` (str), `regex_pattern` (str)", "outputs": "Score (1.0 if matches, 0.0 if not), reason.", "use_case": "Validating output formats (SQL syntax elements, emails, etc.)."},
        {"name": "IsJson", "description": "Checks if the LLM's output is a *syntactically valid JSON object*.", "inputs": "`output` (str)", "outputs": "Score (1.0 if valid JSON, 0.0 if not), reason.", "use_case": "Validating outputs for agentic function arguments or structured data generation."},
        {"name": "Levenshtein Distance", "description": "Calculates the *Levenshtein distance* (edit distance). Opik's metric usually returns a normalized similarity score (0-1, higher is better).", "inputs": "`output` (str), `expected_output` (str)", "outputs": "Normalized similarity score, `distance` (int), reason.", "use_case": "Assessing similarity where minor variations/typos are acceptable."},
    ]
    st.subheader("2.1 Opik Heuristic Metrics")
    for metric in opik_heuristic_metrics_info:
        with st.expander(f"Opik: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

    # Opik LLM-as-a-Judge Metrics
    opik_llm_judge_metrics_info = [
        {"name": "Hallucination", "description": "Assesses if the LLM's output contains fabricated information or statements not supported by the provided context. Uses a judge LLM specified by the `model` parameter during instantiation (defaults to gpt-4o via LiteLLM).", "inputs": "`output` (str), `context` (str, optional), `input` (str, optional - user query)", "outputs": "Score (e.g., 0.0 for no hallucination, 1.0 for hallucination), reason.", "use_case": "Fact-checking in RAG or informational content."},
        {"name": "G-Eval", "description": "A *task-agnostic* LLM-as-a-Judge metric. Users define `task_introduction` and `evaluation_criteria`. The judge LLM (specified by `model`) uses these for evaluation, often via Chain of Thought.", "inputs": "`output` (str - can be complex), `task_introduction` (str - at init), `evaluation_criteria` (str - at init)", "outputs": "Score (often normalized 0-1), reason (may include judge's CoT).", "use_case": "Flexible evaluation for custom aspects, agentic reasoning, Text2SQL semantic correctness."},
        {"name": "Moderation", "description": "Evaluates output appropriateness against safety policies (harmful content, etc.). Uses a judge LLM.", "inputs": "`output` (str)", "outputs": "Safety score (e.g., 0.0 safe, 1.0 unsafe), reason.", "use_case": "Ensuring responsible AI practices."},
        {"name": "AnswerRelevance", "description": "Checks if the LLM's output is relevant to the input question. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - user query)", "outputs": "Score (0-1, higher is more relevant), reason.", "use_case": "Ensuring LLM stays on topic."},
        {"name": "Usefulness", "description": "Assesses if the LLM's output is helpful and valuable. Uses a judge LLM.", "inputs": "`output` (str), `input` (str - user query)", "outputs": "Score (0-1, higher is more useful), reason.", "use_case": "Evaluating if the response is practical and actionable."},
        {"name": "ContextRecall", "description": "Measures if the LLM's output incorporates all relevant information from the provided context. Uses a judge LLM. (Standard definition; Opik's doc had a placeholder).", "inputs": "`output` (str), `context` (str), `input` (str, optional)", "outputs": "Score (0-1, higher is better recall), reason.", "use_case": "Crucial for RAG to ensure context is fully utilized; summarization."},
        {"name": "ContextPrecision", "description": "Evaluates if information in the output claimed from context is accurate and relevant to that context. Uses a judge LLM. (Standard definition; Opik's doc had a placeholder).", "inputs": "`output` (str), `context` (str), `input` (str, optional), `expected_output` (str, optional)", "outputs": "Score (0-1, higher is better precision), reason.", "use_case": "Ensuring RAG outputs faithfully represent context."},
    ]
    st.subheader("2.2 Opik LLM-as-a-Judge Metrics")
    st.markdown("These metrics use a 'judge' LLM, configured via the `model` parameter at instantiation (e.g., `Hallucination(model=your_judge_llm_id)`). This demo routes these calls through `abc_response(model=your_judge_llm_id, prompt=...)`.")
    for metric in opik_llm_judge_metrics_info:
        with st.expander(f"Opik: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")


    st.header("3. Ragas Metrics (via Integration)")
    st.markdown("""
    Ragas is a specialized framework for evaluating RAG (Retrieval Augmented Generation) systems.
    Opik can be used alongside Ragas, where Ragas calculates its specific metrics, and these results can be tracked.
    This demo shows how to use Ragas metrics directly.
    - **Installation:** You need `pip install ragas datasets langchain langchain-openai` (or other LLM providers for Langchain).
    - **Data Format:** Ragas typically expects data as a Hugging Face `Dataset` object with columns like `question`, `answer`, `contexts` (list of strings), and `ground_truth` (list of strings for reference answer).
    - **LLM for Judging:** Ragas metrics that are LLM-based (like Faithfulness, AnswerRelevancy) require an LLM. This can be configured (e.g., using OpenAI models by default, or by providing a Langchain LLM wrapper). In this demo, we'll wrap your `abc_response` into a Langchain-compatible LLM for Ragas.
    """)
    ragas_metrics_info = [
        {"name": "Faithfulness", "description": "Measures if the generated answer is factually consistent with the provided contexts. It identifies claims in the answer and verifies them against the contexts.", "inputs_ragas": "Dataset row with `answer` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher is more faithful).", "use_case": "Core RAG metric to penalize hallucinations based on context."},
        {"name": "AnswerRelevancy", "description": "Assesses how pertinent the generated answer is to the given question. It uses the question and answer, and sometimes context. It often generates variants of the question from the answer and measures similarity.", "inputs_ragas": "Dataset row with `question` (str), `answer` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher is more relevant).", "use_case": "Ensures the RAG answer actually addresses the question."},
        {"name": "ContextPrecision (Ragas)", "description": "Measures the signal-to-noise ratio of the retrieved contexts. It identifies relevant sentences in the contexts concerning the question. (Ragas has its own specific implementation).", "inputs_ragas": "Dataset row with `question` (str), `contexts` (list of str).", "outputs": "Score (0-1, higher means more precise/relevant contexts).", "use_case": "Evaluates the quality of the retrieval step in RAG."},
        {"name": "ContextRecall (Ragas)", "description": "Measures the extent to which the retrieved contexts cover the information in the `ground_truth` answer. (Ragas has its own specific implementation).", "inputs_ragas": "Dataset row with `question` (str), `contexts` (list of str), `ground_truth` (str).", "outputs": "Score (0-1, higher means more of the ground truth is covered by context).", "use_case": "Evaluates if the retriever fetched all necessary information to formulate the ground truth answer."},
        # {"name": "AnswerCorrectness", "description": "Measures semantic similarity and factual correctness of the answer against a ground_truth answer.", "inputs_ragas": "Dataset row with `answer` (str), `ground_truth` (str).", "outputs": "Score (0-1).", "use_case": "End-to-end RAG quality when reference answers exist."},
        # {"name": "AnswerSimilarity", "description": "Measures the semantic similarity between the generated answer and the ground_truth answer.", "inputs_ragas": "Dataset row with `answer` (str), `ground_truth` (str).", "outputs": "Score (0-1).", "use_case": "Compares generated answer to a reference, useful for paraphrasing or style."}
    ]
    for metric in ragas_metrics_info:
        with st.expander(f"Ragas: {metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Ragas Inputs:** {metric['inputs_ragas']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")


    st.header("4. Custom Metrics")
    # ... (Explanation of Custom Metrics as before, referencing the PII and Length Check examples) ...
    custom_metrics_info = [
            {"name": "User-Defined Logic (via OpikBaseMetric)", "description": "Create custom evaluation logic by subclassing `opik.evaluation.metrics.base_metric.BaseMetric` and implementing the `score` method.", "inputs": "User-defined.", "outputs": "`OpikScoreResult` object (name, value, reason, metadata).", "use_case": "Evaluating specific business logic, proprietary aspects, or metrics not in Opik/Ragas."},
            {"name": "PII Detection (Custom Demo)", "description": "A custom metric using regex to detect potential Personally Identifiable Information (PII) in text.", "inputs": "`text_output` (str)", "outputs": "Score (1.0 if PII detected), PII count, details.", "use_case": "Basic PII scanning. Real PII detection is more complex."},
            {"name": "Response Word Count Check (Custom Demo)", "description": "A custom heuristic metric to check if response word count is within a defined range.", "inputs": "`text_output` (str), `min_words` (int), `max_words` (int)", "outputs": "Score (1.0 if in range), actual word count.", "use_case": "Ensuring response brevity or sufficient detail."}
        ]
    for metric in custom_metrics_info:
        with st.expander(f"{metric['name']}"):
            st.markdown(f"**Description:** {metric['description']}\n\n**Inputs:** {metric['inputs']}\n\n**Outputs:** {metric['outputs']}\n\n**Use Case:** {metric['use_case']}")

# --- Section: Opik Metrics Demo ---
elif app_mode == "Opik Metrics Demo" and OPIK_SDK_AVAILABLE:
    st.title("Opik Built-in Metrics Demonstration")

    opik_metric_category = st.selectbox(
        "Select Opik Metric Type:",
        ("Heuristic", "LLM-as-a-Judge")
    )

    if opik_metric_category == "Heuristic":
        heuristic_metric_name = st.selectbox(
            "Choose a Heuristic Metric:",
            ("Equals", "Contains", "RegexMatch", "IsJson", "Levenshtein Distance")
        )
        metric_info = next(m for m in opik_heuristic_metrics_info if m['name'] == heuristic_metric_name)
        st.markdown(f"**Selected Opik Metric:** `{metric_info['name']}` - {metric_info['description']}")
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Inputs:")
            default_output_key = "llm_output_correct" if heuristic_metric_name != "IsJson" else "llm_agent_output_json"
            default_output_data_source = MOCK_DATA_QA if heuristic_metric_name != "IsJson" else MOCK_DATA_AGENTIC
            
            llm_output_val = st.text_area("LLM Output:", default_output_data_source[default_output_key], height=100, key=f"opik_hm_out_{heuristic_metric_name}")

            expected_input_val = ""
            if heuristic_metric_name in ["Equals", "Levenshtein Distance"]:
                expected_input_val = st.text_input("Expected Output:", MOCK_DATA_QA["expected_output_exact"], key=f"opik_hm_exp_{heuristic_metric_name}")
            elif heuristic_metric_name == "Contains":
                expected_input_val = st.text_input("Substring to Check:", "Portuguese", key=f"opik_hm_subs_{heuristic_metric_name}")
                case_sensitive_val = st.checkbox("Case Sensitive?", True, key="opik_hm_case")
            elif heuristic_metric_name == "RegexMatch":
                expected_input_val = st.text_input("Regex Pattern:", MOCK_DATA_TEXT2SQL["sql_syntax_regex"], key="opik_hm_regex")
            
            if st.button(f"Evaluate with Opik {heuristic_metric_name}", key=f"btn_opik_hm_{heuristic_metric_name}"):
                results = None
                metric_instance = None
                try:
                    if heuristic_metric_name == "Equals":
                        metric_instance = Equals()
                        results = metric_instance.score(output=llm_output_val, expected_output=expected_input_val)
                    elif heuristic_metric_name == "Contains":
                        metric_instance = Contains()
                        results = metric_instance.score(output=llm_output_val, substring=expected_input_val, case_sensitive=case_sensitive_val)
                    elif heuristic_metric_name == "RegexMatch":
                        metric_instance = RegexMatch()
                        results = metric_instance.score(output=llm_output_val, regex_pattern=expected_input_val)
                    elif heuristic_metric_name == "IsJson":
                        metric_instance = IsJson()
                        results = metric_instance.score(output=llm_output_val)
                    elif heuristic_metric_name == "Levenshtein Distance":
                        metric_instance = Levenshtein()
                        results = metric_instance.score(output=llm_output_val, expected_output=expected_input_val)
                    st.session_state[f"opik_hm_res_{heuristic_metric_name}"] = results.to_dict() if hasattr(results, 'to_dict') else results
                except Exception as e:
                    st.error(f"Error evaluating Opik {heuristic_metric_name}: {e}")
        with cols[1]:
            if f"opik_hm_res_{heuristic_metric_name}" in st.session_state:
                display_metric_results_streamlit(f"Opik {heuristic_metric_name}", st.session_state[f"opik_hm_res_{heuristic_metric_name}"])


    elif opik_metric_category == "LLM-as-a-Judge":
        llm_judge_metric_name = st.selectbox(
            "Choose an Opik LLM-as-a-Judge Metric:",
            ("Hallucination", "G-Eval", "Moderation", "AnswerRelevance", "Usefulness", "ContextRecall", "ContextPrecision")
        )
        metric_info = next(m for m in opik_llm_judge_metrics_info if m['name'] == llm_judge_metric_name)
        st.markdown(f"**Selected Opik Metric:** `{metric_info['name']}` - {metric_info['description']}")
        st.caption(f"This metric will use the Judge LLM: '{judge_llm_model_input}' (via `abc_response`).")

        cols_judge = st.columns(2)
        generated_llm_output_for_eval = ""
        llm_response_details_for_eval = None

        with cols_judge[0]:
            st.subheader("Inputs & LLM Output Generation:")
            user_query_val = ""
            context_text_val = ""
            expected_answer_val = "" # For ContextPrecision

            if llm_judge_metric_name == "Hallucination":
                user_query_val = st.text_input("User Question (Optional for Hallucination):", MOCK_DATA_RAG["question"], key="opik_hall_q")
                context_text_val = st.text_area("Context:", MOCK_DATA_RAG["context_docs"][0], height=100, key="opik_hall_ctx")
                st.markdown("Now, let's generate an answer based on this context, or input one manually.")
                if st.button(f"Generate Answer with '{primary_llm_model_input}' for Hallucination Check", key="opik_hall_gen"):
                    prompt = f"Context: {context_text_val}\n\nQuestion: {user_query_val}\n\nAnswer the question based ONLY on the context."
                    st.session_state.opik_hall_llm_details = abc_response(primary_llm_model_input, prompt)
                if "opik_hall_llm_details" in st.session_state:
                    llm_response_details_for_eval = st.session_state.opik_hall_llm_details
                    generated_llm_output_for_eval = llm_response_details_for_eval[0]
                generated_llm_output_for_eval = st.text_area("LLM Output to Evaluate:", generated_llm_output_for_eval or MOCK_DATA_RAG["llm_answer_hallucination"], height=100, key="opik_hall_llmout")


            elif llm_judge_metric_name == "G-Eval":
                st.markdown("Define the G-Eval task. The 'Output to Evaluate' can be complex (e.g., agent thought process, SQL query + schema + question).")
                task_intro_val = st.text_area("G-Eval: Task Introduction", MOCK_DATA_TEXT2SQL["question_complex"] + "\nSchema:\n" + MOCK_DATA_TEXT2SQL["db_schema"], height=100, key="opik_geval_intro") # Example: using NLQ + Schema as part of task context
                eval_criteria_val = st.text_area("G-Eval: Evaluation Criteria",
                                                 "1. SQL Syntactic Correctness (0-1)\n2. Semantic Alignment with Question & Schema (0-1)\n3. Efficiency/Conciseness (0-1)\nReturn JSON: {'score': <avg_0_to_1_score>, 'reason': <CoT>}",
                                                 height=150, key="opik_geval_crit")
                st.markdown("Now, let's generate a SQL query or provide one manually.")
                if st.button(f"Generate SQL with '{primary_llm_model_input}' for G-Eval", key="opik_geval_gensql"):
                    prompt = f"Convert to SQL. Schema:\n{MOCK_DATA_TEXT2SQL['db_schema']}\n\nQuestion: {MOCK_DATA_TEXT2SQL['question_complex']}"
                    st.session_state.opik_geval_llm_details = abc_response(primary_llm_model_input, prompt) # Assuming model='text2sql-generator'
                if "opik_geval_llm_details" in st.session_state:
                     llm_response_details_for_eval = st.session_state.opik_geval_llm_details
                     generated_llm_output_for_eval = llm_response_details_for_eval[0]
                # The 'output' for G-Eval's score method is the text to be judged.
                # For Text2SQL, we might pass the generated SQL. The task_intro and criteria already contain the schema and NLQ for the judge.
                generated_llm_output_for_eval = st.text_area("LLM Output (e.g., SQL Query) to Evaluate:", generated_llm_output_for_eval or MOCK_DATA_TEXT2SQL["generated_sql_complex_correct"], height=100, key="opik_geval_llmout")

            elif llm_judge_metric_name == "Moderation":
                generated_llm_output_for_eval = st.text_area("Text to Evaluate for Moderation:", MOCK_DATA_QA["llm_output_unsafe"], height=100, key="opik_mod_llmout")
                llm_response_details_for_eval = (generated_llm_output_for_eval, 0,0,0) # Direct input

            elif llm_judge_metric_name in ["AnswerRelevance", "Usefulness"]:
                user_query_val = st.text_input("User Question/Input:", MOCK_DATA_QA["question"], key=f"opik_ansrel_q_{llm_judge_metric_name}")
                st.markdown(f"Now, let's generate an answer to this question with '{primary_llm_model_input}'.")
                if st.button(f"Generate Answer for {llm_judge_metric_name} Check", key=f"opik_ansrel_gen_{llm_judge_metric_name}"):
                    st.session_state[f"opik_ansrel_llm_details_{llm_judge_metric_name}"] = abc_response(primary_llm_model_input, user_query_val)
                if f"opik_ansrel_llm_details_{llm_judge_metric_name}" in st.session_state:
                    llm_response_details_for_eval = st.session_state[f"opik_ansrel_llm_details_{llm_judge_metric_name}"]
                    generated_llm_output_for_eval = llm_response_details_for_eval[0]
                generated_llm_output_for_eval = st.text_area("LLM Output to Evaluate:", generated_llm_output_for_eval or (MOCK_DATA_QA["llm_output_correct"] if llm_judge_metric_name=="AnswerRelevance" else MOCK_DATA_QA["llm_output_correct"]), height=100, key=f"opik_ansrel_llmout_{llm_judge_metric_name}")


            elif llm_judge_metric_name in ["ContextRecall", "ContextPrecision"]:
                user_query_val = st.text_input("User Question (RAG):", MOCK_DATA_RAG["question"], key=f"opik_ctx_q_{llm_judge_metric_name}")
                context_text_val = st.text_area("Context (RAG):", "\n---\n".join(MOCK_DATA_RAG["context_docs"]), height=150, key=f"opik_ctx_c_{llm_judge_metric_name}")
                if llm_judge_metric_name == "ContextPrecision":
                    expected_answer_val = st.text_input("Expected Answer (Optional for ContextPrecision):", MOCK_DATA_RAG["ground_truth_answer_rag"], key=f"opik_ctx_exp_{llm_judge_metric_name}")

                st.markdown(f"Now, let's generate an answer based on this context with '{primary_llm_model_input}'.")
                if st.button(f"Generate Answer for {llm_judge_metric_name} Check", key=f"opik_ctx_gen_{llm_judge_metric_name}"):
                    prompt = f"Context: {context_text_val}\n\nQuestion: {user_query_val}\n\nAnswer the question based ONLY on the context."
                    st.session_state[f"opik_ctx_llm_details_{llm_judge_metric_name}"] = abc_response(primary_llm_model_input, prompt)
                if f"opik_ctx_llm_details_{llm_judge_metric_name}" in st.session_state:
                    llm_response_details_for_eval = st.session_state[f"opik_ctx_llm_details_{llm_judge_metric_name}"]
                    generated_llm_output_for_eval = llm_response_details_for_eval[0]
                generated_llm_output_for_eval = st.text_area("LLM Output to Evaluate:", generated_llm_output_for_eval or MOCK_DATA_RAG["llm_answer_good"], height=100, key=f"opik_ctx_llmout_{llm_judge_metric_name}")


            if st.button(f"Evaluate with Opik {llm_judge_metric_name}", key=f"btn_opik_ljm_{llm_judge_metric_name}"):
                results = None
                if not generated_llm_output_for_eval:
                    st.warning("Please generate or provide LLM output to evaluate.")
                else:
                    try:
                        metric_instance = None
                        if llm_judge_metric_name == "Hallucination":
                            metric_instance = Hallucination(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval, context=context_text_val, input=user_query_val)
                        elif llm_judge_metric_name == "G-Eval":
                            metric_instance = GEval(task_introduction=task_intro_val, evaluation_criteria=eval_criteria_val, model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval) # The output here is what the judge evaluates based on task_intro and criteria
                        elif llm_judge_metric_name == "Moderation":
                            metric_instance = Moderation(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval)
                        elif llm_judge_metric_name == "AnswerRelevance":
                            metric_instance = AnswerRelevance(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval, input=user_query_val)
                        elif llm_judge_metric_name == "Usefulness":
                            metric_instance = Usefulness(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval, input=user_query_val)
                        elif llm_judge_metric_name == "ContextRecall":
                            metric_instance = ContextRecall(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval, context=context_text_val, input=user_query_val)
                        elif llm_judge_metric_name == "ContextPrecision":
                            metric_instance = ContextPrecision(model=judge_llm_model_input)
                            results = metric_instance.score(output=generated_llm_output_for_eval, context=context_text_val, input=user_query_val, expected_output=expected_answer_val)
                        
                        st.session_state[f"opik_ljm_res_{llm_judge_metric_name}"] = results.to_dict() if hasattr(results, 'to_dict') else results
                    except Exception as e:
                         st.error(f"Error evaluating Opik {llm_judge_metric_name}: {e}")
        
        with cols_judge[1]:
            if f"opik_ljm_res_{llm_judge_metric_name}" in st.session_state:
                display_metric_results_streamlit(f"Opik {llm_judge_metric_name}", st.session_state[f"opik_ljm_res_{llm_judge_metric_name}"], llm_response_details_for_eval)


# --- Section: Ragas Metrics Demo ---
elif app_mode == "Ragas Metrics Demo" and RAGAS_SDK_AVAILABLE and OPIK_SDK_AVAILABLE: # Check OPIK for abc_response consistency
    st.title("Ragas Metrics Demonstration (for RAG Evaluation)")
    st.markdown("""
    This section demonstrates key Ragas metrics. Ragas expects data typically as a list of dictionaries
    or a Hugging Face `Dataset` object. For LLM-based Ragas metrics, we'll use a Langchain
    wrapper around your `abc_response` function with the specified 'Judge LLM Model ID/Path'.
    """)

    # Prepare Ragas LLM
    try:
        ragas_judge_llm_instance = RagasCustomLLM(model_name=judge_llm_model_input)
        ragas_llm_wrapper = LangchainLLMWrapper(ragas_judge_llm_instance)
        st.info(f"Ragas will use your `abc_response` via Langchain wrapper with judge model: '{judge_llm_model_input}'")
    except Exception as e:
        st.error(f"Could not initialize Ragas LLM wrapper: {e}. Ragas LLM-based metrics may not work.")
        ragas_llm_wrapper = None


    ragas_metric_name = st.selectbox(
        "Choose a Ragas Metric:",
        ("Faithfulness", "AnswerRelevancy", "ContextPrecision (Ragas)", "ContextRecall (Ragas)")
        # Add "AnswerCorrectness" if you want to demo with ground_truth input
    )
    metric_info = next(m for m in ragas_metrics_info if m['name'] == ragas_metric_name)
    st.markdown(f"**Selected Ragas Metric:** `{metric_info['name']}` - {metric_info['description']}")

    cols_ragas = st.columns(2)
    with cols_ragas[0]:
        st.subheader("RAG Inputs:")
        rag_question = st.text_input("RAG Question:", MOCK_DATA_RAG["question"], key="ragas_q")
        
        num_context_docs = st.number_input("Number of Context Documents:", min_value=1, max_value=len(MOCK_DATA_RAG["context_docs"]), value=2, key="ragas_num_ctx")
        rag_contexts_list = []
        for i in range(num_context_docs):
            rag_contexts_list.append(st.text_area(f"Context Document {i+1}:", MOCK_DATA_RAG["context_docs"][i], height=75, key=f"ragas_ctx_{i}"))
        
        rag_ground_truth_answer = ""
        if ragas_metric_name in ["ContextRecall (Ragas)", "AnswerCorrectness"]: # Metrics needing ground truth
            rag_ground_truth_answer = st.text_area("Ground Truth Answer (for Ragas ContextRecall/AnswerCorrectness):", MOCK_DATA_RAG["ground_truth_answer_rag"], height=75, key="ragas_gt")

        st.markdown(f"**1. Generate LLM Answer using '{primary_llm_model_input}':**")
        if st.button("Generate RAG Answer", key="btn_ragas_gen_ans"):
            prompt_for_rag = f"Contexts:\n{'---'.join(rag_contexts_list)}\n\nQuestion: {rag_question}\n\nAnswer the question based on the provided contexts."
            st.session_state.ragas_llm_answer_details = abc_response(primary_llm_model_input, prompt_for_rag) # Use a specific model for RAG answer generation if needed

        generated_rag_answer = ""
        llm_rag_answer_details = None
        if "ragas_llm_answer_details" in st.session_state:
            llm_rag_answer_details = st.session_state.ragas_llm_answer_details
            generated_rag_answer = llm_rag_answer_details[0]
        
        generated_rag_answer = st.text_area("Generated RAG Answer (to be evaluated):", generated_rag_answer or MOCK_DATA_RAG["llm_answer_good"], height=100, key="ragas_ans_eval")
        if llm_rag_answer_details:
             st.caption(f"Time: {llm_rag_answer_details[1]:.2f}s, Input Tokens: {llm_rag_answer_details[2]}, Output Tokens: {llm_rag_answer_details[3]}")


        if st.button(f"Evaluate with Ragas {ragas_metric_name}", key=f"btn_eval_ragas_{ragas_metric_name}"):
            if not generated_rag_answer:
                st.warning("Please generate or provide a RAG answer to evaluate.")
            elif not ragas_llm_wrapper and ragas_metric_name in ["Faithfulness", "AnswerRelevancy", "ContextRecall (Ragas)"]: # Check if LLM needed
                 st.error("Ragas LLM wrapper not available. Cannot run LLM-based Ragas metric.")
            else:
                # Prepare dataset for Ragas
                data_sample = {
                    "question": [rag_question],
                    "answer": [generated_rag_answer],
                    "contexts": [rag_contexts_list],
                }
                if rag_ground_truth_answer and ragas_metric_name in ["ContextRecall (Ragas)", "AnswerCorrectness"]: # Only add if available AND needed
                    data_sample["ground_truth"] = [rag_ground_truth_answer]
                
                # Ensure all required keys are present for the specific metric
                required_keys_map = {
                    "Faithfulness": ["question", "answer", "contexts"],
                    "AnswerRelevancy": ["question", "answer", "contexts"],
                    "ContextPrecision (Ragas)": ["question", "contexts"], # Ragas context precision might not need 'answer' if it evaluates retrieval only
                    "ContextRecall (Ragas)": ["question", "contexts", "ground_truth"]
                }
                # Filter data_sample to only include keys required by the current metric to avoid Ragas errors
                current_required_keys = required_keys_map.get(ragas_metric_name, [])
                filtered_data_sample = {k: v for k, v in data_sample.items() if k in current_required_keys or k in ["question", "answer", "contexts"]} # always include base ones
                
                # Ragas expects a HuggingFace Dataset
                try:
                    ragas_dataset = HuggingFaceDataset.from_dict(filtered_data_sample)
                    
                    metric_instance_ragas = None
                    if ragas_metric_name == "Faithfulness":
                        metric_instance_ragas = ragas_faithfulness(llm=ragas_llm_wrapper)
                    elif ragas_metric_name == "AnswerRelevancy":
                        metric_instance_ragas = ragas_answer_relevancy(llm=ragas_llm_wrapper)
                    elif ragas_metric_name == "ContextPrecision (Ragas)":
                         # This Ragas metric typically doesn't require an LLM for its basic version but can use one for advanced.
                         # Let's assume we are using the version that might require an LLM if available, or it has a non-LLM fallback.
                         # Ragas context_precision evaluates if context is relevant to the question.
                        metric_instance_ragas = ragas_context_precision(llm=ragas_llm_wrapper if ragas_llm_wrapper else None)
                    elif ragas_metric_name == "ContextRecall (Ragas)":
                        if "ground_truth" not in filtered_data_sample:
                            st.error("Ground Truth answer is required for Ragas ContextRecall.")
                            st.stop()
                        metric_instance_ragas = ragas_context_recall(llm=ragas_llm_wrapper)
                    
                    if metric_instance_ragas:
                        st.write(f"Evaluating with Ragas {ragas_metric_name} using judge: {judge_llm_model_input}...")
                        with st.spinner("Ragas evaluation in progress..."):
                            results = ragas_evaluate(ragas_dataset, metrics=[metric_instance_ragas], llm=ragas_llm_wrapper)
                        st.session_state[f"ragas_res_{ragas_metric_name}"] = results.to_pandas().to_dict(orient='records')[0] if results else {"error": "Ragas evaluation returned no results."}
                    else:
                        st.error(f"Could not instantiate Ragas metric: {ragas_metric_name}")

                except Exception as e:
                    st.error(f"Error during Ragas {ragas_metric_name} evaluation: {e}")
                    import traceback
                    st.code(traceback.format_exc())


    with cols_ragas[1]:
        if f"ragas_res_{ragas_metric_name}" in st.session_state:
            display_metric_results_streamlit(f"Ragas {ragas_metric_name}", st.session_state[f"ragas_res_{ragas_metric_name}"], llm_rag_answer_details)


# --- Section: Custom Metrics Demo ---
elif app_mode == "Custom Metrics Demo" and OPIK_SDK_AVAILABLE:
    st.title("Custom Opik Metrics Demonstration")
    custom_metric_choice = st.selectbox(
        "Choose a Custom Metric Example:",
        ("PII Detection (Demo)", "Response Word Count Check (Demo)")
    )

    cols_custom_demo = st.columns(2)
    with cols_custom_demo[0]:
        st.subheader("Inputs:")
        if custom_metric_choice == "PII Detection (Demo)":
            metric_info = next(m for m in custom_metrics_info if m['name'] == "PII Detection (Custom Demo)")
            st.markdown(f"**Selected Custom Metric:** `{metric_info['name']}` - {metric_info['description']}")
            text_to_scan_pii = st.text_area("Text Output to Scan for PII:", MOCK_DATA_QA["llm_output_with_pii"], height=100, key="custom_pii_text")
            if st.button("Evaluate with Custom PII Detection", key="btn_custom_pii"):
                custom_metric_instance = CustomPIIDetectionMetric()
                results = custom_metric_instance.score(text_output=text_to_scan_pii)
                st.session_state.custom_pii_results = results
        
        elif custom_metric_choice == "Response Word Count Check (Demo)":
            metric_info = next(m for m in custom_metrics_info if m['name'] == "Response Word Count Check (Custom Demo)")
            st.markdown(f"**Selected Custom Metric:** `{metric_info['name']}` - {metric_info['description']}")
            text_to_check_len = st.text_area("Text Output for Length Check:", MOCK_DATA_QA["long_output"], height=100, key="custom_len_text")
            min_w = st.number_input("Min Words:", 10, key="cust_minw")
            max_w = st.number_input("Max Words:", 100, key="cust_maxw")
            if st.button("Evaluate with Custom Length Check", key="btn_custom_len"):
                custom_metric_instance = CustomResponseLengthCheck(min_words=min_w, max_words=max_w)
                results = custom_metric_instance.score(text_output=text_to_check_len)
                st.session_state.custom_len_results = results

    with cols_custom_demo[1]:
        if custom_metric_choice == "PII Detection (Demo)" and 'custom_pii_results' in st.session_state:
            display_metric_results_streamlit("Custom PII Detection", st.session_state.custom_pii_results)
        elif custom_metric_choice == "Response Word Count Check (Demo)" and 'custom_len_results' in st.session_state:
            display_metric_results_streamlit("Custom Word Count Check", st.session_state.custom_len_results)


elif (app_mode == "Opik Metrics Demo" and not OPIK_SDK_AVAILABLE) or \
     (app_mode == "Ragas Metrics Demo" and not RAGAS_SDK_AVAILABLE) or \
     (app_mode == "Custom Metrics Demo" and not OPIK_SDK_AVAILABLE):
    st.error(f"{app_mode.split(' ')[0]} SDK is not available. Please install it to use this section.")
