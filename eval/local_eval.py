# eval/local_eval.py
import pandas as pd
import sqlite3
import os
from contextlib import contextmanager

# Import comparison functions from the original eval module
# Make sure eval/eval.py exists and these functions are defined there
try:
    from .eval import normalize_table, compare_df, subset_df
except ImportError:
    # Fallback if running script directly or structure issue
    try:
        from eval import normalize_table, compare_df, subset_df
    except ImportError:
         raise ImportError("Could not import normalize_table, compare_df, subset_df from eval.eval")


@contextmanager
def create_in_memory_db(csv_file_paths: list[str]) -> sqlite3.Connection:
    """
    Context manager to create an in-memory SQLite database and load data from CSVs.

    Args:
        csv_file_paths: A list of paths to the CSV files to load.

    Yields:
        A sqlite3.Connection object to the populated in-memory database.
        The database is automatically closed upon exiting the context.
    """
    conn = None
    try:
        print("Creating in-memory SQLite database...")
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        for csv_path in csv_file_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found at {csv_path}, skipping.")
                continue

            # Derive table name from filename (e.g., 'data/customers.csv' -> 'customers')
            table_name = os.path.splitext(os.path.basename(csv_path))[0]
            # Sanitize table name (basic example, might need more robust logic)
            table_name = "".join(c if c.isalnum() else "_" for c in table_name)
            if not table_name:
                print(f"Warning: Could not derive valid table name from {csv_path}, skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
                # Ensure column names are valid SQL identifiers
                df.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in df.columns]
                # Load data into SQLite table
                df.to_sql(table_name, conn, index=False, if_exists='replace')
                print(f"Loaded data from {csv_path} into table '{table_name}'.")
            except Exception as e:
                print(f"Error loading {csv_path} into table '{table_name}': {e}")
                # Decide if this error should stop the process or just be warned
                # raise # Uncomment to make loading errors fatal

        yield conn # Provide the connection to the 'with' block

    finally:
        if conn:
            print("Closing in-memory SQLite database.")
            conn.close()


def execute_sql_on_memory_db(conn: sqlite3.Connection, sql_query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the provided in-memory SQLite connection.

    Args:
        conn: The active sqlite3 connection.
        sql_query: The SQL query string to execute.

    Returns:
        A Pandas DataFrame containing the query results.

    Raises:
        Exception: If the query execution fails.
    """
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        # Optional: Convert data types if needed, pandas might handle basic types well
        return result_df
    except Exception as e:
        print(f"Error executing SQL: {sql_query}\nError: {e}")
        raise # Re-raise the exception to be caught by the caller


def compare_results_local(
    conn: sqlite3.Connection,
    query_gold: str,
    query_gen: str,
    question: str, # Keep for context in normalize_table
    query_category: str, # Keep for context in normalize_table
    decimal_points: int = None # Keep for potential rounding
) -> tuple[bool, bool]:
    """
    Executes gold and generated queries on the in-memory DB and compares results.

    Args:
        conn: Active connection to the in-memory SQLite database.
        query_gold: The gold standard SQL query.
        query_gen: The generated SQL query.
        question: The original natural language question (for context).
        query_category: The category of the query (for context).
        decimal_points: Number of decimal points for rounding comparison.

    Returns:
        A tuple (exact_match: bool, subset_match: bool).
    """
    try:
        results_gold = execute_sql_on_memory_db(conn, query_gold)
        print(f"Gold Query OK. Result shape: {results_gold.shape}")
    except Exception as e_gold:
        print(f"Error executing GOLD query: {e_gold}")
        # Gold query failed, generated query cannot match or be a subset
        raise ValueError(f"Gold query failed to execute: {e_gold}") from e_gold

    try:
        results_gen = execute_sql_on_memory_db(conn, query_gen)
        print(f"Generated Query OK. Result shape: {results_gen.shape}")
    except Exception as e_gen:
        print(f"Error executing GENERATED query: {e_gen}")
        # Generated query failed, cannot match or be a subset
        raise ValueError(f"Generated query failed to execute: {e_gen}") from e_gen

    # Perform comparison using imported functions
    try:
        # Use normalize_table and compare_df for robust comparison
        # Pass original SQL for potential ORDER BY handling in normalize_table
        is_exact_match = compare_df(
            results_gold.copy(), results_gen.copy(), query_category, question, query_gold, query_gen
        )

        # Check for subset if not an exact match
        is_subset_match = False
        if is_exact_match:
            is_subset_match = True
        else:
            # Check if gold is a subset of generated
             is_subset_match = subset_df(
                 results_gold.copy(), results_gen.copy(), query_category, question, query_gold, query_gen
             )

        print(f"Comparison Results: Exact={is_exact_match}, Subset Correct={is_subset_match}")
        return (is_exact_match, is_subset_match)

    except Exception as e_cmp:
        print(f"Error during DataFrame comparison: {e_cmp}")
        # Treat comparison errors as mismatch
        return (False, False)

