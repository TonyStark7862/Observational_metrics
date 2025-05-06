# benchmark_ui.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import glob
import time

# Import the refactored benchmark logic and config classes
from benchmark_logic import run_benchmark_logic, BenchmarkConfig, BenchmarkSummary

# --- Configuration ---
AVAILABLE_MODELS = ["default_model_v1", "experimental_model_v2", "legacy_model"]
DATASETS = {
    "E-commerce (Customers, Orders, Products)": {
        "eval_file": "data/eval_cases_rich.csv",
        "data_dir": "data",
        "description": "Sample e-commerce data."
    },
}
RESULTS_DIR = "results"
DEFAULT_PROMPT_TEMPLATE = "prompts/prompt.md" # Make sure this file exists

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helper Functions ---
@st.cache_data
def get_schema_display_benchmark(data_dir, eval_file):
    """Reads data CSVs listed in eval file to display headers as schema."""
    schema_str = "Schema Information (from data CSV headers):\n\n"
    try:
        if not os.path.exists(eval_file):
             return f"Error: Evaluation definition file not found at {eval_file}"
        eval_df = pd.read_csv(eval_file, nrows=5)
        eval_df.fillna("", inplace=True)
        if 'data_csv_files' not in eval_df.columns:
            return f"Error: Column 'data_csv_files' not found in {eval_file}"

        all_csvs = set()
        for csv_list_str in eval_df['data_csv_files']:
            if isinstance(csv_list_str, str) and csv_list_str:
                all_csvs.update(fname.strip() for fname in csv_list_str.split(','))
        if not all_csvs: return "No data CSV files listed in evaluation definition."

        for csv_filename in sorted(list(all_csvs)):
            csv_path = os.path.join(data_dir, csv_filename)
            table_name = os.path.splitext(csv_filename)[0]
            schema_str += f"Table: {table_name} ({csv_filename})\n"
            if os.path.exists(csv_path):
                try:
                    df_sample = pd.read_csv(csv_path, nrows=0)
                    schema_str += f"  Columns: {', '.join(df_sample.columns)}\n\n"
                except Exception as e: schema_str += f"  Error reading columns: {e}\n\n"
            else: schema_str += f"  Error: File not found at {csv_path}\n\n"
        return schema_str
    except Exception as e: return f"Error reading evaluation file {eval_file}: {e}"

def display_summary_metrics(summary: BenchmarkSummary):
    """Displays summary metrics in Streamlit columns."""
    st.markdown("#### Run Summary")
    cols = st.columns(6)
    cols[0].metric("Total Cases", summary.total_processed)
    cols[1].metric("Accuracy", f"{summary.accuracy:.2f}%")
    cols[2].metric("Exact Match", f"{summary.exact_match_rate:.2f}%")
    cols[3].metric("Gen Errors", f"{summary.total_gen_errors} ({summary.gen_error_rate:.1f}%)")
    cols[4].metric("Exec Errors", f"{summary.total_exec_errors} ({summary.exec_error_rate:.1f}%)")
    cols[5].metric("Avg Latency", f"{summary.avg_latency_s:.3f}s")

def find_benchmark_results_files():
    """Finds CSV files in the results directory, sorted by modification time."""
    try:
        list_of_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        if not list_of_files: return {}
        list_of_files.sort(key=os.path.getmtime, reverse=True)
        return {os.path.basename(f): f for f in list_of_files}
    except Exception as e:
        print(f"Error finding result files: {e}")
        return {}

# --- Main Function to Display the Benchmark Tab ---
def display_benchmark_section():
    """Renders the Streamlit UI for the benchmark tab."""

    st.header("🚀 SQL Generation Benchmark")
    st.caption("Evaluate custom SQL generation models using local CSV data.")

    # --- Configuration Section ---
    st.subheader("Configuration")
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_model = st.selectbox(
            "Select Model:", options=AVAILABLE_MODELS, key="bm_model"
        )
        selected_dataset_name = st.selectbox(
            "Select Dataset:", options=list(DATASETS.keys()), key="bm_dataset"
        )
        prompt_template = st.text_input(
            "Prompt Template File:", value=DEFAULT_PROMPT_TEMPLATE, key="bm_prompt_file"
        )
        parallel_threads = st.number_input("Parallel Threads", min_value=1, max_value=10, value=1, key="bm_threads")

    with col2:
        dataset_config = DATASETS[selected_dataset_name]
        st.info(f"**Dataset:** {dataset_config.get('description', 'N/A')}")
        schema_info = get_schema_display_benchmark(dataset_config["data_dir"], dataset_config["eval_file"])
        st.text_area("Schema Preview:", schema_info, height=150, key="bm_schema_preview")

    run_button = st.button("Run Benchmark", type="primary", key="bm_run_button", use_container_width=True)

    st.divider()

    # --- Run Logic & Results Display ---
    if 'benchmark_running' not in st.session_state: st.session_state.benchmark_running = False
    if 'benchmark_results_df' not in st.session_state: st.session_state.benchmark_results_df = None
    if 'benchmark_summary' not in st.session_state: st.session_state.benchmark_summary = None
    if 'benchmark_output_file' not in st.session_state: st.session_state.benchmark_output_file = None

    if run_button and not st.session_state.benchmark_running:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_results_df = None
        st.session_state.benchmark_summary = None
        st.session_state.benchmark_output_file = None
        st.info("Benchmark run initiated...")
        st.rerun()

    if st.session_state.benchmark_running or st.session_state.benchmark_results_df is not None:
         st.subheader("📈 Benchmark Run Status & Results")

    if st.session_state.benchmark_running:
        st.button("Run Benchmark", key="bm_run_button_disabled", use_container_width=True, disabled=True)
        config = BenchmarkConfig(
            model_identifier=selected_model,
            eval_definition_file=dataset_config["eval_file"],
            data_dir=dataset_config["data_dir"],
            prompt_template_file=prompt_template,
            parallel_threads=parallel_threads
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_short_name = selected_dataset_name.split(" ")[0].lower()
        output_filename = f"{config.model_identifier}_{dataset_short_name}_{timestamp}.csv"
        output_filepath = os.path.join(RESULTS_DIR, output_filename)

        status_text = st.empty()
        progress_bar = st.progress(0.0, text="Initializing benchmark...")
        log_expander = st.expander("Show Run Logs (Updates may be limited)", expanded=False)
        log_placeholder = log_expander.empty()

        def update_progress(value):
            value = max(0.0, min(1.0, value))
            progress_text = f"Running benchmark... {int(value * 100)}% complete"
            try:
                progress_bar.progress(value, text=progress_text)
                status_text.text(progress_text)
            except Exception as e: print(f"Streamlit progress update error: {e}")

        start_time = time.time()
        try:
            results_df, summary = run_benchmark_logic(config, progress_callback=update_progress)
            st.session_state.benchmark_results_df = results_df
            st.session_state.benchmark_summary = summary
            st.session_state.benchmark_output_file = output_filepath

            if not results_df.empty:
                 results_df.to_csv(output_filepath, index=False, float_format="%.3f")
                 status_text.success(f"Benchmark finished in {time.time() - start_time:.2f}s. Results saved.")
                 log_placeholder.info("Run completed. Check summary and results below.")
            else:
                 status_text.warning("Benchmark run completed, but no results were generated.")
                 log_placeholder.warning("Run completed, but no results were generated.")

        except Exception as e:
            status_text.error(f"Benchmark run failed: {e}")
            import traceback
            log_placeholder.code(traceback.format_exc())
        finally:
            st.session_state.benchmark_running = False
            try: progress_bar.empty()
            except: pass
            st.rerun()

    if st.session_state.benchmark_results_df is not None:
        if st.session_state.benchmark_summary:
            display_summary_metrics(st.session_state.benchmark_summary)
        st.dataframe(st.session_state.benchmark_results_df)
        if st.session_state.benchmark_output_file and os.path.exists(st.session_state.benchmark_output_file):
            try:
                with open(st.session_state.benchmark_output_file, "rb") as fp:
                    st.download_button(
                        label="Download Results as CSV",
                        data=fp,
                        file_name=os.path.basename(st.session_state.benchmark_output_file),
                        mime='text/csv',
                    )
            except Exception as e:
                st.warning(f"Could not prepare results for download: {e}")

    # --- Past Results Section ---
    st.divider()
    st.subheader(" vergangene Benchmarks")

    past_results_dict = find_benchmark_results_files()

    if not past_results_dict:
        st.info("No past benchmark results found.")
    else:
        selected_past_file_name = st.selectbox(
            "Select Past Run:",
            options=list(past_results_dict.keys()),
            key="bm_past_run_select"
        )
        if selected_past_file_name:
            selected_past_filepath = past_results_dict[selected_past_file_name]
            st.caption(f"Displaying results from: `{selected_past_filepath}`")
            try:
                past_df = pd.read_csv(selected_past_filepath)
                temp_summary = BenchmarkSummary(total_processed=len(past_df))
                if not past_df.empty:
                    if 'correct' in past_df: temp_summary.total_correct = past_df['correct'].sum()
                    if 'exact_match' in past_df: temp_summary.total_exact_match = past_df['exact_match'].sum()
                    if 'generation_error_msg' in past_df: temp_summary.total_gen_errors = past_df['generation_error_msg'].fillna('').apply(bool).sum()
                    if 'error_db_exec' in past_df: temp_summary.total_exec_errors = past_df['error_db_exec'].sum()
                    if 'latency_seconds' in past_df: temp_summary.avg_latency_s = past_df['latency_seconds'].mean()

                display_summary_metrics(temp_summary)
                st.dataframe(past_df)
            except Exception as e:
                st.error(f"Error loading past results from {selected_past_file_name}: {e}")
