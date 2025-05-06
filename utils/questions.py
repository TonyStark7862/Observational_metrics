# utils/questions.py
import pandas as pd
from typing import Optional
import os
import sqlite3 # For reading schema from data CSVs
from .aliases import *

# Import utilities needed for prompt generation inputs
from .gen_prompt import to_prompt_schema # Keep for potential schema generation if needed
from .aliases import generate_aliases, get_table_names, mk_alias_str # Import alias functions

def get_schema_from_csvs(csv_paths: list[str]) -> str:
    """
    Attempts to infer a basic schema DDL string by reading headers from CSV files.
    Loads data into a temporary in-memory SQLite DB to infer types.

    Args:
        csv_paths: List of paths to data CSV files.

    Returns:
        A string representing the inferred schema in CREATE TABLE format,
        or an empty string if inference fails.
    """
    schema_ddl = ""
    if not csv_paths:
        return ""

    conn = None
    try:
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: Schema inference skipped for missing file: {csv_path}")
                continue

            table_name = os.path.splitext(os.path.basename(csv_path))[0]
            table_name = "".join(c if c.isalnum() else "_" for c in table_name)
            if not table_name:
                continue

            try:
                df = pd.read_csv(csv_path, nrows=5) # Read a few rows for type inference
                df.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in df.columns]

                # Load into temp table for type inference
                df.to_sql(f"temp_{table_name}", conn, index=False, if_exists='replace')

                # Get inferred schema from SQLite
                cursor.execute(f"PRAGMA table_info(temp_{table_name});")
                columns_info = cursor.fetchall()
                # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk

                if not columns_info:
                    print(f"Warning: Could not get column info for table '{table_name}' from {csv_path}")
                    continue

                schema_ddl += f"CREATE TABLE {table_name} (\n"
                col_defs = []
                for i, col_info in enumerate(columns_info):
                    col_name = col_info[1]
                    col_type = col_info[2] if col_info[2] else "TEXT" # Default to TEXT if type is missing
                    # Basic type mapping (can be expanded)
                    if 'int' in col_type.lower():
                        col_type = 'INTEGER'
                    elif 'real' in col_type.lower() or 'float' in col_type.lower() or 'double' in col_type.lower():
                         col_type = 'REAL'
                    else:
                         col_type = 'TEXT' # Default for strings, dates etc. in basic inference

                    col_defs.append(f"  {col_name} {col_type}")

                schema_ddl += ",\n".join(col_defs)
                schema_ddl += "\n);\n"

            except Exception as e:
                print(f"Error inferring schema for {csv_path} -> {table_name}: {e}")

    except Exception as db_e:
         print(f"Error creating in-memory DB for schema inference: {db_e}")
    finally:
        if conn:
            conn.close()

    return schema_ddl


def prepare_questions_df(
    questions_file: str,
    data_dir: str,
    num_questions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads evaluation definition, prepares inputs for detailed prompt generation.

    Args:
        questions_file: Path to the input CSV defining evaluation cases.
                        Expected columns: 'question', 'query' (gold SQL),
                        'data_csv_files'. Optional: 'instructions', 'glossary',
                        'k_shot_prompt', 'query_category'.
        data_dir: The base directory containing the data CSV files.
        num_questions: Maximum number of questions to load.

    Returns:
        A Pandas DataFrame ready for processing.
    """
    try:
        question_query_df = pd.read_csv(questions_file, nrows=num_questions)
        print(f"Loaded {len(question_query_df)} evaluation cases from {questions_file}")
    except FileNotFoundError:
        print(f"Error: Evaluation definition file not found at {questions_file}")
        raise
    except Exception as e:
        print(f"Error reading CSV file {questions_file}: {e}")
        raise

    # --- Validate required columns ---
    required_cols = ['question', 'query', 'data_csv_files']
    missing_cols = [col for col in required_cols if col not in question_query_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {questions_file}: {', '.join(missing_cols)}")

    # --- Prepare DataFrame & Optional Columns ---
    question_query_df.fillna("", inplace=True)

    # Handle optional columns used in prompt generation
    prompt_input_cols = ['instructions', 'glossary', 'k_shot_prompt', 'prev_invalid_sql', 'prev_error_msg']
    for col in prompt_input_cols:
        if col not in question_query_df.columns:
            question_query_df[col] = ""
        else:
            # Basic formatting/cleaning can be done here if needed
            if col == 'instructions' and question_query_df[col].dtype == 'object':
                 question_query_df[col] = question_query_df[col].apply(lambda x: x.replace(". ", ".\n") if isinstance(x, str) else x)
                 question_query_df[col] = question_query_df[col].apply(
                     lambda x: f"\nFollow the instructions below to generate the query:\n{x}\n" if x else ""
                 )
            elif col == 'glossary' and question_query_df[col].dtype == 'object':
                 question_query_df[col] = question_query_df[col].apply(
                     lambda x: f"\nUse the following glossary terms if relevant:\n{x}\n" if x else ""
                 )
            elif col == 'k_shot_prompt' and question_query_df[col].dtype == 'object':
                  question_query_df[col] = question_query_df[col].apply(lambda x: x.replace("\\n", "\n") if isinstance(x, str) else x)
                  question_query_df[col] = question_query_df[col].apply(
                      lambda x: f"\nRefer to these examples:\n{x}\n" if x else ""
                  )


    # --- Get Full Data Paths ---
    def get_full_paths(filenames_str):
        if not filenames_str: return []
        relative_paths = [fname.strip() for fname in filenames_str.split(',')]
        full_paths = [os.path.join(data_dir, fname) for fname in relative_paths]
        for fpath in full_paths:
            if not os.path.exists(fpath): print(f"Warning: Data CSV file not found: {fpath}")
        return full_paths
    question_query_df['full_data_paths'] = question_query_df['data_csv_files'].apply(get_full_paths)

    # --- Infer Schema & Generate Aliases ---
    print("Inferring schema from data CSVs...")
    question_query_df['schema_string'] = question_query_df['full_data_paths'].apply(get_schema_from_csvs)

    print("Generating table aliases...")
    def get_aliases_for_row(schema_str):
        if not schema_str: return ""
        table_names = get_table_names(schema_str) # Use utility from aliases.py
        aliases_dict = generate_aliases_dict(table_names) # Use utility from aliases.py
        return mk_alias_str(aliases_dict) # Use utility from aliases.py
    question_query_df['table_aliases_str'] = question_query_df['schema_string'].apply(get_aliases_for_row)


    # --- Initialize Result Columns ---
    # (These are filled by the runner)
    question_query_df["prompt_generated"] = "" # Store the generated prompt
    question_query_df["generated_query"] = ""
    question_query_df["latency_seconds"] = 0.0
    question_query_df["input_tokens"] = 0
    question_query_df["output_tokens"] = 0
    question_query_df["generation_error_msg"] = ""
    question_query_df["exact_match"] = 0
    question_query_df["correct"] = 0
    question_query_df["exec_error_msg"] = ""
    question_query_df["error_db_exec"] = 0

    # Add query_category if missing
    if 'query_category' not in question_query_df.columns:
        question_query_df['query_category'] = 'default'

    print(f"Prepared DataFrame with {len(question_query_df)} evaluation cases.")
    return question_query_df
