# benchmark_logic.py
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm # tqdm might not display well in Streamlit logs, remove direct use
import time
import sqlite3
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Import necessary functions from existing modules
# Ensure these paths are correct relative to where this script is run from (usually the root)
from utils.questions import prepare_questions_df # Use the version adapted for local CSVs
from utils.llm_abc import call_abc_generator_with_retry, LLMABCResponse # Your generator wrapper
from eval.local_eval import create_in_memory_db, compare_results_local # Local eval helper
from utils.gen_prompt import generate_prompt # Rich prompt generator

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model_identifier: str
    eval_definition_file: str
    data_dir: str
    prompt_template_file: str
    num_questions: Optional[int] = None
    parallel_threads: int = 1
    decimal_points: Optional[int] = None

@dataclass
class BenchmarkResultRow:
    """Represents the results for a single evaluation row."""
    # Input/Config Data (copied from input df)
    question: str = ""
    query: str = "" # Gold query
    data_csv_files: str = ""
    query_category: str = "default"
    instructions: Optional[str] = ""
    glossary: Optional[str] = ""
    k_shot_prompt: Optional[str] = ""
    # Prompt Generation
    schema_string: str = ""
    table_aliases_str: str = ""
    prompt_generated: str = ""
    # SQL Generation Results
    generated_query: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    generation_error_msg: str = ""
    # Evaluation Results
    exact_match: int = 0
    correct: int = 0 # Subset match included
    exec_error_msg: str = ""
    error_db_exec: int = 0 # Flag for DB/Exec error

    # Allow extra fields from input df
    extra_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        init_data = {k: v for k, v in data.items() if k in known_keys}
        extra_data = {k: v for k, v in data.items() if k not in known_keys}
        instance = cls(**init_data)
        instance.extra_data = extra_data
        return instance

    def to_dict(self) -> Dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values() if f.name != 'extra_data'}
        data.update(self.extra_data)
        return data

@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    total_processed: int = 0
    total_correct: int = 0
    total_exact_match: int = 0
    total_gen_errors: int = 0
    total_exec_errors: int = 0
    avg_latency_s: float = 0.0

    @property
    def accuracy(self) -> float:
        return (self.total_correct / self.total_processed * 100) if self.total_processed > 0 else 0.0

    @property
    def exact_match_rate(self) -> float:
        return (self.total_exact_match / self.total_processed * 100) if self.total_processed > 0 else 0.0

    @property
    def gen_error_rate(self) -> float:
        return (self.total_gen_errors / self.total_processed * 100) if self.total_processed > 0 else 0.0

    @property
    def exec_error_rate(self) -> float:
        return (self.total_exec_errors / self.total_processed * 100) if self.total_processed > 0 else 0.0


def _process_single_row(row_tuple: tuple, config: BenchmarkConfig) -> BenchmarkResultRow:
    """
    Internal function to process a single evaluation case.
    Takes a tuple (index, row_series) and BenchmarkConfig.
    Returns a BenchmarkResultRow object.
    """
    index, row = row_tuple
    result = BenchmarkResultRow.from_dict(row.to_dict())

    # --- Step 1: Generate Detailed Prompt ---
    try:
        schema_str = row.get('schema_string', '')
        aliases_str = row.get('table_aliases_str', '')
        result.schema_string = schema_str
        result.table_aliases_str = aliases_str

        prompt = generate_prompt(
            prompt_file=config.prompt_template_file,
            question=result.question,
            schema_string=schema_str,
            db_type='sqlite',
            instructions=result.instructions or "",
            k_shot_prompt=result.k_shot_prompt or "",
            glossary=result.glossary or "",
            table_aliases_str=aliases_str
        )
        result.prompt_generated = prompt
    except Exception as prompt_err:
        print(f"Error generating prompt for index {index}: {prompt_err}")
        result.generation_error_msg = f"PROMPT GENERATION FAILED: {prompt_err}"
        return result

    # --- Step 2: Generate SQL ---
    generation_api_result: LLMABCResponse = call_abc_generator_with_retry(
        model=config.model_identifier,
        prompt=prompt
    )
    result.generated_query = generation_api_result.generated_sql
    result.latency_seconds = generation_api_result.latency_seconds
    result.input_tokens = generation_api_result.input_tokens
    result.output_tokens = generation_api_result.output_tokens
    result.generation_error_msg = generation_api_result.error_msg if generation_api_result.error_msg else ""

    # --- Step 3: Evaluate if Generation Succeeded ---
    if not result.generation_error_msg and result.generated_query:
        query_gen = result.generated_query
        query_gold = result.query
        data_csv_paths = row.get('full_data_paths', [])

        if not data_csv_paths:
             result.exec_error_msg = "EVALUATION FAILED: No data CSV paths found for this case."
             result.error_db_exec = 1
             return result

        conn = None
        try:
            with create_in_memory_db(data_csv_paths) as conn:
                exact_match, correct_subset = compare_results_local(
                    conn=conn,
                    query_gold=query_gold,
                    query_gen=query_gen,
                    question=result.question,
                    query_category=result.query_category,
                    decimal_points=config.decimal_points
                )
                result.exact_match = int(exact_match)
                result.correct = int(correct_subset)
        except (sqlite3.Error, pd.errors.DatabaseError, ValueError, FileNotFoundError) as db_err:
            print(f"Evaluation Error (DB/Exec/Compare) for index {index}: {db_err}")
            result.exec_error_msg = f"EVALUATION FAILED: {str(db_err)}"
            result.error_db_exec = 1
        except Exception as e:
            print(f"Unexpected Evaluation Error for index {index}: {e}")
            result.exec_error_msg = f"UNEXPECTED EVAL ERROR: {str(e)}"
            result.error_db_exec = 1
    else:
        result.exec_error_msg = "Skipped due to generation error"

    return result

def run_benchmark_logic(config: BenchmarkConfig, progress_callback=None) -> tuple[pd.DataFrame, BenchmarkSummary]:
    """
    Runs the benchmark evaluation based on the provided configuration.

    Args:
        config: A BenchmarkConfig object with run parameters.
        progress_callback: An optional function to call with progress updates.

    Returns:
        A tuple containing:
        - pd.DataFrame: DataFrame with detailed results for each case.
        - BenchmarkSummary: Summary statistics for the run.
    """
    print(f"--- Running Benchmark: Model={config.model_identifier}, EvalFile={config.eval_definition_file} ---")

    try:
        questions_df = prepare_questions_df(
            questions_file=config.eval_definition_file,
            data_dir=config.data_dir,
            num_questions=config.num_questions
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error preparing questions: {e}")
        raise  # Re-raise for the caller (Streamlit UI) to handle

    input_rows = list(questions_df.iterrows())
    output_rows_list = []
    summary = BenchmarkSummary()
    all_latencies = []

    with ThreadPoolExecutor(max_workers=config.parallel_threads) as executor:
        futures = [
            executor.submit(_process_single_row, row_tuple, config)
            for row_tuple in input_rows
        ]
        total_tasks = len(futures)
        print(f"Submitting {total_tasks} tasks to {config.parallel_threads} workers...")

        for i, future in enumerate(as_completed(futures)):
            try:
                result_row_obj = future.result()
                output_rows_list.append(result_row_obj.to_dict())

                summary.total_processed += 1
                if result_row_obj.generation_error_msg: summary.total_gen_errors += 1
                if result_row_obj.error_db_exec: summary.total_exec_errors += 1
                if result_row_obj.correct: summary.total_correct += 1
                if result_row_obj.exact_match: summary.total_exact_match += 1
                if result_row_obj.latency_seconds > 0: all_latencies.append(result_row_obj.latency_seconds)

            except Exception as exc:
                print(f'Critical error processing future result: {exc}')
                summary.total_processed += 1
                summary.total_gen_errors += 1

            if progress_callback:
                progress_callback((i + 1) / total_tasks)

    print("Benchmark processing complete.")

    if not output_rows_list:
        print("Warning: No results were generated.")
        return pd.DataFrame(), summary

    output_df = pd.DataFrame(output_rows_list)

    summary.avg_latency_s = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    # Define expected columns order
    base_cols = [f.name for f in BenchmarkResultRow.__dataclass_fields__.values() if f.name != 'extra_data']
    extra_cols_present = list(output_df.columns.difference(base_cols))
    final_columns = base_cols + extra_cols_present
    output_df = output_df.reindex(columns=final_columns, fill_value="")

    print(f"Benchmark Summary: Processed={summary.total_processed}, Correct={summary.total_correct}, GenErrors={summary.total_gen_errors}, ExecErrors={summary.total_exec_errors}")

    return output_df, summary
