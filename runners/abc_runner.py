# runners/abc_runner.py
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import sqlite3

# Use the updated questions module
from utils.questions import prepare_questions_df
# Use the wrapper for your custom SQL generator
from utils.llm_abc import call_abc_generator_with_retry, LLMABCResponse
# Use the NEW local evaluation helper
from eval.local_eval import create_in_memory_db, compare_results_local
# Import the restored prompt generator
from utils.gen_prompt import generate_prompt

def process_row(row_tuple: tuple, model_name: str, prompt_template_file: str, decimal_points: int) -> dict:
    """
    Processes a single row:
    1. Generates detailed prompt.
    2. Generates SQL using custom generator.
    3. Creates in-memory DB from specified CSVs.
    4. Evaluates generated SQL against gold standard using the in-memory DB.
    5. Returns results.

    Args:
        row_tuple: A tuple containing (index, row_series).
        model_name: Identifier for the custom SQL generator model/method.
        prompt_template_file: Path to the prompt template file (.md, .txt).
        decimal_points: Precision for comparing numeric results (optional).

    Returns:
        A dictionary containing the original row data updated with generation
        and evaluation results.
    """
    index, row = row_tuple
    row_dict = row.to_dict() # Work with a dictionary copy

    print(f"\n--- Processing Row Index: {index} ---")
    print(f"Question: {row_dict['question'][:100]}...")

    # Initialize results
    row_dict["prompt_generated"] = ""
    row_dict["generated_query"] = ""
    row_dict["latency_seconds"] = 0.0
    row_dict["input_tokens"] = 0
    row_dict["output_tokens"] = 0
    row_dict["generation_error_msg"] = ""
    row_dict["exact_match"] = 0
    row_dict["correct"] = 0
    row_dict["exec_error_msg"] = ""
    row_dict["error_db_exec"] = 0

    # --- Step 1: Generate Detailed Prompt ---
    try:
        prompt = generate_prompt(
            prompt_file=prompt_template_file,
            question=row_dict['question'],
            schema_string=row_dict['schema_string'],
            db_type='sqlite', # Assuming in-memory evaluation uses SQLite syntax
            instructions=row_dict.get('instructions', ''),
            k_shot_prompt=row_dict.get('k_shot_prompt', ''),
            glossary=row_dict.get('glossary', ''),
            prev_invalid_sql=row_dict.get('prev_invalid_sql', ''),
            prev_error_msg=row_dict.get('prev_error_msg', ''),
            table_aliases_str=row_dict.get('table_aliases_str', '')
            # Add other args if needed by your template
        )
        row_dict["prompt_generated"] = prompt # Store the generated prompt
        print(f"Generated Prompt (first 100 chars): {prompt[:100]}...")
    except Exception as prompt_err:
        print(f"Error generating prompt for index {index}: {prompt_err}")
        row_dict["generation_error_msg"] = f"PROMPT GENERATION FAILED: {prompt_err}"
        # Cannot proceed without a prompt
        return row_dict

    # --- Step 2: Generate SQL ---
    generation_result: LLMABCResponse = call_abc_generator_with_retry(
        model=model_name,
        prompt=prompt # Use the detailed prompt
    )
    row_dict["generated_query"] = generation_result.generated_sql
    row_dict["latency_seconds"] = generation_result.latency_seconds
    row_dict["input_tokens"] = generation_result.input_tokens # Or maybe len(prompt.split())?
    row_dict["output_tokens"] = generation_result.output_tokens
    row_dict["generation_error_msg"] = generation_result.error_msg if generation_result.error_msg else ""

    # --- Step 3: Evaluate if Generation Succeeded ---
    if not row_dict["generation_error_msg"] and row_dict["generated_query"]:
        query_gen = row_dict["generated_query"]
        query_gold = row_dict["query"]
        data_csv_paths = row_dict['full_data_paths']
        question = row_dict["question"]
        query_category = row_dict.get("query_category", "default")

        conn = None
        try:
            with create_in_memory_db(data_csv_paths) as conn:
                exact_match, correct_subset = compare_results_local(
                    conn=conn,
                    query_gold=query_gold,
                    query_gen=query_gen,
                    question=question,
                    query_category=query_category,
                    decimal_points=decimal_points
                )
                row_dict["exact_match"] = int(exact_match)
                row_dict["correct"] = int(correct_subset)
                row_dict["exec_error_msg"] = ""
                row_dict["error_db_exec"] = 0
        except (sqlite3.Error, pd.errors.DatabaseError, ValueError, FileNotFoundError) as db_err:
            print(f"Evaluation Error (DB/Exec/Compare) for index {index}: {db_err}")
            row_dict["exact_match"] = 0; row_dict["correct"] = 0
            row_dict["exec_error_msg"] = f"EVALUATION FAILED: {str(db_err)}"
            row_dict["error_db_exec"] = 1
        except Exception as e:
            print(f"Unexpected Evaluation Error for index {index}: {e}")
            row_dict["exact_match"] = 0; row_dict["correct"] = 0
            row_dict["exec_error_msg"] = f"UNEXPECTED EVAL ERROR: {str(e)}"
            row_dict["error_db_exec"] = 1
    else:
        row_dict["exec_error_msg"] = "Skipped due to generation error"
        print(f"Skipping evaluation for index {index} due to generation error.")

    print(f"--- Finished Row Index: {index} ---")
    return row_dict


def run_abc_eval(args):
    """
    Main function using local CSV data, detailed prompts, and in-memory SQLite.
    """
    print(f"--- Starting Local CSV Evaluation Runner (Rich Prompts) ---")
    print(f"Data Directory: {args.data_dir}")
    print(f"Prompt Template(s): {args.prompt_file}") # Now using prompt file arg
    print(f"Custom Generator ID: {args.model}")

    output_dir = os.path.dirname(args.output_file[0])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle prompt file mapping (similar to original main.py)
    if len(args.questions_file) > 1 and len(args.prompt_file) == 1:
        prompt_files = args.prompt_file * len(args.questions_file)
    elif len(args.questions_file) == 1 and len(args.prompt_file) > 1:
         # This case might be less common now, but handle it
         # Requires reading the single questions_file multiple times or processing differently
         print("Warning: Processing one question file with multiple prompts.")
         # For simplicity, we'll just use the first prompt file for all questions if Q=1, P>1
         # A more complex setup could iterate through prompts for the single question set
         if len(args.output_file) == len(args.prompt_file):
              prompt_files = args.prompt_file
              questions_files = args.questions_file * len(args.prompt_file)
              output_files = args.output_file
         else:
              print("Error: Mismatched output files for multiple prompts and single question file.")
              return # Or raise error
    elif len(args.questions_file) == len(args.prompt_file):
         prompt_files = args.prompt_file
         questions_files = args.questions_file
         output_files = args.output_file
    else: # Should have been caught by main.py validation, but double-check
         print("Error: Mismatch between number of question, prompt, and output files.")
         return


    for questions_file, prompt_file, output_file in zip(questions_files, prompt_files, output_files):
        print(f"\nProcessing: {questions_file} + {prompt_file} -> {output_file}")

        try:
            questions_df = prepare_questions_df(
                questions_file,
                args.data_dir,
                args.num_questions
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error preparing questions from {questions_file}: {e}")
            continue

        input_rows = list(questions_df.iterrows())
        output_rows = []
        total_processed, total_gen_errors, total_exec_errors, total_correct = 0, 0, 0, 0

        with ThreadPoolExecutor(max_workers=args.parallel_threads) as executor:
            futures = [
                executor.submit(
                    process_row,
                    row_tuple,
                    args.model,
                    prompt_file, # Pass the specific prompt file for this run
                    args.decimal_points
                )
                for row_tuple in input_rows
            ]

            print(f"Submitting {len(futures)} tasks to {args.parallel_threads} workers...")
            with tqdm(total=len(futures), desc="Evaluating SQL (Local CSV)") as pbar:
                for future in as_completed(futures):
                    try:
                        result_row = future.result()
                        output_rows.append(result_row)
                        if result_row.get("generation_error_msg"): total_gen_errors += 1
                        if result_row.get("error_db_exec"): total_exec_errors += 1
                        if result_row.get("correct"): total_correct += 1
                    except Exception as exc:
                        print(f'Critical error processing future result: {exc}')
                        total_gen_errors += 1
                    finally:
                        total_processed += 1
                        pbar.update(1)
                        accuracy = (total_correct / total_processed * 100) if total_processed > 0 else 0
                        pbar.set_description(
                            f"Evaluating (Acc: {accuracy:.2f}%, GenErrs: {total_gen_errors}, ExecErrs: {total_exec_errors})"
                        )

        if not output_rows:
            print("No results were processed for this file pair.")
            continue

        output_df = pd.DataFrame(output_rows)

        # Define expected columns order
        final_columns = [
            'question', 'query', 'data_csv_files', # Input/Gold
            'prompt_generated', # Added prompt
            'generated_query', # Generated SQL
            'correct', 'exact_match', # Evaluation Results
            'latency_seconds', 'input_tokens', 'output_tokens', # Generation Perf
            'generation_error_msg', 'exec_error_msg', 'error_db_exec', # Errors
            'query_category', # Optional original column
            # Add other original columns if needed
        ]
        existing_cols = [col for col in final_columns if col in output_df.columns]
        extra_cols = [col for col in output_df.columns if col not in existing_cols]
        output_df = output_df[existing_cols + extra_cols]

        try:
            output_df.to_csv(output_file, index=False, float_format="%.3f")
            print(f"\nResults saved to {output_file}")
            print(f"Summary: Processed={total_processed}, Correct={total_correct}, GenErrors={total_gen_errors}, ExecErrors={total_exec_errors}")
        except Exception as e:
            print(f"\nError saving results to {output_file}: {e}")

    print("\n--- Local CSV Evaluation Runner (Rich Prompts) Finished ---")

