# main.py
import argparse
import os

# Import the specific runner function for the local CSV setup
from runners.abc_runner import run_abc_eval # Assuming the runner is named abc_runner.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQL evaluation using local CSV data, detailed prompts, and in-memory SQLite.")

    # --- Data Arguments ---
    parser.add_argument(
        "-q", "--questions_file",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to the input CSV file(s) defining evaluation cases. Required columns: question, query (gold SQL), data_csv_files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the actual data CSV files listed in 'data_csv_files' column."
    )
    parser.add_argument(
        "-o", "--output_file",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to the output CSV file(s) for saving results."
    )
    parser.add_argument(
        "-n", "--num_questions",
        type=int,
        default=None,
        help="Maximum number of evaluation cases to process from the input file(s)."
    )

    # --- Prompt Template Argument ---
    parser.add_argument(
        "-f", "--prompt_file",
        nargs="+", # Allow multiple prompt files
        type=str,
        required=True,
        help="Path(s) to the prompt template file(s) (.md or .txt)."
    )

    # --- Custom Generator Argument ---
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="custom_generator", # Default identifier
        help="Identifier for your custom SQL generator model/method."
    )

    # --- Evaluation Arguments ---
    parser.add_argument(
        "-dp", "--decimal_points",
        type=int,
        default=None,
        help="Number of decimal points to round to for numeric comparisons (optional)."
    )

    # --- Execution Arguments ---
    parser.add_argument(
        "-p", "--parallel_threads",
        type=int,
        default=1, # Start with 1 for SQLite in-memory, increase carefully
        help="Number of parallel threads for processing evaluation cases."
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    # Check matching lengths or single prompt/question file cases
    q_len, p_len, o_len = len(args.questions_file), len(args.prompt_file), len(args.output_file)
    if not (q_len == p_len == o_len or
            (q_len > 1 and p_len == 1 and o_len == q_len) or
            (q_len == 1 and p_len > 1 and o_len == p_len)):
         raise ValueError(
             "Number of question, prompt, and output files must match, "
             "or prompt/question file must be single if the others are multiple. "
             f"Got Q:{q_len}, P:{p_len}, O:{o_len}"
         )

    if not os.path.isdir(args.data_dir):
         raise FileNotFoundError(f"Data directory not found: {args.data_dir}")


    print("--- Local CSV Evaluation Configuration ---")
    print(f"Evaluation Definition File(s): {args.questions_file}")
    print(f"Prompt Template File(s): {args.prompt_file}")
    print(f"Data CSV Directory: {args.data_dir}")
    print(f"Output File(s): {args.output_file}")
    print(f"Custom Generator ID: {args.model}")
    print(f"Max Cases: {'All' if args.num_questions is None else args.num_questions}")
    print(f"Parallel Threads: {args.parallel_threads}")
    print(f"Decimal Points for Compare: {args.decimal_points if args.decimal_points is not None else 'Default'}")
    print("---------------------------------------")

    # --- Run Evaluation ---
    try:
        run_abc_eval(args)
    except FileNotFoundError as e:
         print(f"\nError: A required file or directory was not found: {e}")
    except ValueError as e:
         print(f"\nError: Invalid configuration or data: {e}")
    except ImportError as e:
         print(f"\nError: A required library is missing: {e}. Please check dependencies (pandas, tqdm, numpy).") # Added numpy
    except Exception as e:
        import traceback
        print(f"\nAn unexpected critical error occurred during evaluation: {e}")
        traceback.print_exc()

