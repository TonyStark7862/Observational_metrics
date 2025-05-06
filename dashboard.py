# dashboard.py
import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import time
import glob
from datetime import datetime

# --- Configuration ---

# Define the models available under the 'abc' generator type
# These should correspond to the identifiers your abc_sql_generator understands
AVAILABLE_MODELS = ["default_model_v1", "experimental_model_v2", "legacy_model"]

# Define available datasets
# Maps display name to a dictionary containing the evaluation definition file
# and the directory holding the corresponding data CSVs.
DATASETS = {
    "E-commerce (Customers, Orders, Products)": {
        "eval_file": "data/eval_cases_rich.csv", # Your input definition CSV
        "data_dir": "data",
        "description": "Sample e-commerce data including customer info, order history, and product catalog."
    },
    # Add more datasets here as needed
    # "Sample Dataset 2": {
    #     "eval_file": "data/eval_cases_set2.csv",
    #     "data_dir": "data/set2_data",
    #     "description": "Description for dataset 2."
    # },
}

RESULTS_DIR = "results" # Directory where main.py saves results
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists

# --- Helper Functions ---

@st.cache_data # Cache schema reading
def get_schema_display(data_dir, eval_file):
    """Reads data CSVs listed in eval file to display headers as schema."""
    schema_str = "Schema Information:\n\n"
    try:
        eval_df = pd.read_csv(eval_file, nrows=5) # Read a few rows to get file lists
        eval_df.fillna("", inplace=True)
        all_csvs = set()
        for csv_list_str in eval_df['data_csv_files']:
            if csv_list_str:
                all_csvs.update(fname.strip() for fname in csv_list_str.split(','))

        if not all_csvs:
            return "No data CSV files listed in evaluation definition."

        for csv_filename in sorted(list(all_csvs)):
            csv_path = os.path.join(data_dir, csv_filename)
            table_name = os.path.splitext(csv_filename)[0]
            schema_str += f"Table: {table_name} (from {csv_filename})\n"
            if os.path.exists(csv_path):
                try:
                    # Read only header to get columns
                    df_sample = pd.read_csv(csv_path, nrows=0)
                    schema_str += f"  Columns: {', '.join(df_sample.columns)}\n\n"
                except Exception as e:
                    schema_str += f"  Error reading columns: {e}\n\n"
            else:
                schema_str += f"  Error: File not found at {csv_path}\n\n"
        return schema_str
    except Exception as e:
        return f"Error reading evaluation file {eval_file} to determine schema: {e}"

def calculate_metrics(df):
    """Calculates summary metrics from the results dataframe."""
    if df.empty:
        return {"Accuracy": 0, "Exact Match": 0, "Generation Errors": 0, "Execution Errors": 0, "Avg Latency (s)": 0}

    total_rows = len(df)
    accuracy = df['correct'].mean() * 100 if 'correct' in df else 0
    exact_match = df['exact_match'].mean() * 100 if 'exact_match' in df else 0

    gen_errors = df['generation_error_msg'].fillna('').apply(lambda x: bool(x)).sum()
    exec_errors = df['error_db_exec'].fillna(0).sum()

    avg_latency = df['latency_seconds'].mean() if 'latency_seconds' in df else 0

    return {
        "Accuracy": f"{accuracy:.2f}%",
        "Exact Match": f"{exact_match:.2f}%",
        "Generation Errors": f"{gen_errors} ({gen_errors/total_rows:.1%})",
        "Execution Errors": f"{exec_errors} ({exec_errors/total_rows:.1%})",
        "Avg Latency (s)": f"{avg_latency:.3f}",
        "Total Cases": total_rows
    }

def find_latest_result_file(pattern):
    """Finds the most recently modified file matching a pattern."""
    try:
        list_of_files = glob.glob(os.path.join(RESULTS_DIR, pattern))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest result file: {e}")
        return None

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="LLM SQL Evaluation")

# Sidebar
with st.sidebar:
    st.title("📊 Benchmark")
    # Since 'abc' is the only generator type now, we don't need a top-level selectbox for it.
    # If you add more generator types later, you can add a selectbox here.
    st.markdown("### Configuration")

    selected_model = st.selectbox(
        "Select Model (within 'abc' type):",
        options=AVAILABLE_MODELS,
        key="selected_model"
    )

    selected_dataset_name = st.selectbox(
        "Select Dataset:",
        options=list(DATASETS.keys()),
        key="selected_dataset"
    )

    # Get dataset config
    dataset_config = DATASETS[selected_dataset_name]
    eval_file_path = dataset_config["eval_file"]
    data_dir_path = dataset_config["data_dir"]

    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info(dataset_config.get("description", "No description available."))

    # Display Schema
    schema_info = get_schema_display(data_dir_path, eval_file_path)
    st.text_area("Schema Preview (from data CSV headers):", schema_info, height=250)


# Main Page
st.title("Custom LLM SQL Generation Benchmark")
st.markdown("Use the sidebar to configure and run evaluations using your custom SQL generator.")
st.divider()

# --- Run Benchmark Section ---
st.header("🚀 Run New Benchmark")

# Use columns for better layout
col1, col2 = st.columns([1, 3])

with col1:
    st.write("**Selected Configuration:**")
    st.write(f"- **Model:** `{selected_model}`")
    st.write(f"- **Dataset:** `{selected_dataset_name}`")
    # Add other relevant config display if needed (e.g., prompt template)
    prompt_template_file = "prompts/prompt.md" # Assuming a default template
    st.write(f"- **Prompt Template:** `{prompt_template_file}`")

    run_button = st.button("Run Benchmark", type="primary", use_container_width=True)

with col2:
    st.write("**Instructions:**")
    st.write("1. Select the desired model and dataset from the sidebar.")
    st.write("2. Verify the schema preview.")
    st.write(f"3. Ensure your custom SQL generator (`utils/llm_abc.py`) is correctly implemented.")
    st.write(f"4. Ensure the necessary data files exist in the '{data_dir_path}' directory.")
    st.write(f"5. Click 'Run Benchmark'. Results will appear below.")


# --- Benchmark Execution Logic ---
if run_button:
    st.divider()
    st.subheader("📈 Benchmark Results (Current Run)")

    # Generate a unique output filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_short_name = selected_dataset_name.split(" ")[0].lower() # e.g., "ecommerce"
    output_filename = f"{selected_model}_{dataset_short_name}_{timestamp}.csv"
    output_filepath = os.path.join(RESULTS_DIR, output_filename)

    # Prepare arguments for main.py
    # Use sys.executable to ensure using the same Python environment
    command = [
        sys.executable,
        "main.py",
        "-q", eval_file_path,
        "--data_dir", data_dir_path,
        "-f", prompt_template_file, # Assuming one prompt file for now
        "-o", output_filepath,
        "-m", selected_model,
        "-p", "1", # Start with 1 thread for local SQLite, adjust if needed
        # Add other necessary args like -dp if needed
    ]

    st.write(f"**Running evaluation for:** `{selected_model}` on `{selected_dataset_name}`")
    st.write(f"Output will be saved to: `{output_filepath}`")
    st.write(f"Command: `{' '.join(command)}`") # Show the command being run

    progress_bar = st.progress(0, text="Starting benchmark...")
    start_time = time.time()
    results_placeholder = st.empty() # Placeholder for results table/metrics
    log_placeholder = st.expander("Show Run Logs", expanded=False)
    log_content = ""

    try:
        # Run main.py as a subprocess
        # Use Popen for potentially streaming output (though complex to capture well in Streamlit)
        # For simplicity, using run and capturing output at the end.
        # Consider redirecting stderr to stdout for combined logs
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit code, handle manually
            encoding='utf-8',
            errors='replace' # Handle potential encoding errors in output
        )
        run_time = time.time() - start_time
        progress_bar.progress(100, text=f"Benchmark finished in {run_time:.2f}s")

        log_content += f"--- STDOUT ---\n{process.stdout}\n"
        log_content += f"\n--- STDERR ---\n{process.stderr}\n"
        log_placeholder.code(log_content, language='bash')

        if process.returncode != 0:
            st.error(f"Benchmark script failed with exit code {process.returncode}. Check logs above.")
        elif not os.path.exists(output_filepath):
             st.error(f"Benchmark script seemed to run okay, but the expected output file was not found: {output_filepath}")
        else:
            st.success(f"Benchmark completed successfully in {run_time:.2f} seconds!")
            # Load and display results
            try:
                results_df = pd.read_csv(output_filepath)
                results_placeholder.dataframe(results_df)

                st.subheader("📊 Summary Metrics (Current Run)")
                metrics = calculate_metrics(results_df)
                cols = st.columns(len(metrics))
                for i, (key, value) in enumerate(metrics.items()):
                    cols[i].metric(label=key, value=value)

                # Update session state for potential later use
                st.session_state['last_run_df'] = results_df
                st.session_state['last_run_metrics'] = metrics
                st.session_state['last_run_config'] = {
                    "model": selected_model, "dataset": selected_dataset_name, "file": output_filepath
                }

            except Exception as e:
                st.error(f"Error reading or processing results file {output_filepath}: {e}")

    except FileNotFoundError:
        st.error(f"Error: Could not find 'main.py'. Make sure the dashboard is run from the project root directory.")
        progress_bar.progress(100, text="Error")
    except Exception as e:
        st.error(f"An unexpected error occurred while running the benchmark: {e}")
        progress_bar.progress(100, text="Error")
        log_placeholder.code(log_content + f"\n\nPYTHON EXCEPTION:\n{e}", language='bash')


# --- Past Results Section ---
st.divider()
st.header(" vergangene Benchmarks") # German for "Past Benchmarks" - just for fun, change if needed!

past_results_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")), key=os.path.getmtime, reverse=True)
past_results_options = {os.path.basename(f): f for f in past_results_files}

if not past_results_options:
    st.info("No past benchmark results found in the 'results' directory.")
else:
    selected_past_file_name = st.selectbox(
        "Select a past benchmark run to view:",
        options=list(past_results_options.keys()),
        index=0, # Default to the latest
        key="selected_past_run"
    )

    if selected_past_file_name:
        selected_past_filepath = past_results_options[selected_past_file_name]
        st.write(f"Displaying results from: `{selected_past_filepath}`")
        try:
            past_df = pd.read_csv(selected_past_filepath)
            st.subheader("📊 Summary Metrics (Past Run)")
            past_metrics = calculate_metrics(past_df)
            cols_past = st.columns(len(past_metrics))
            for i, (key, value) in enumerate(past_metrics.items()):
                cols_past[i].metric(label=key, value=value)

            st.subheader("📄 Detailed Results (Past Run)")
            st.dataframe(past_df)
        except FileNotFoundError:
             st.error(f"Error: Selected past result file not found: {selected_past_filepath}")
        except Exception as e:
            st.error(f"Error loading or displaying past results from {selected_past_filepath}: {e}")

