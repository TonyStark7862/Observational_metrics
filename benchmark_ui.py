# benchmark_ui.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import glob
import time
import re # For parsing filenames

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
PROMPTS_DIR = "prompts" # Directory for prompt templates

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True) # Ensure prompts dir exists

# --- Helper Functions ---
@st.cache_data
def list_prompt_files(prompts_dir=PROMPTS_DIR):
    """Lists .md or .txt files in the prompts directory."""
    try:
        files = [f for f in os.listdir(prompts_dir) if os.path.isfile(os.path.join(prompts_dir, f)) and (f.endswith(".md") or f.endswith(".txt"))]
        return sorted(files) if files else ["prompt.md"] # Default if empty
    except FileNotFoundError:
        st.error(f"Prompts directory not found: {prompts_dir}")
        return ["prompt.md"] # Default fallback

@st.cache_data
def get_schema_display_benchmark(data_dir, eval_file):
    """Reads data CSVs listed in eval file to display headers as schema."""
    schema_str = "Schema Information (from data CSV headers):\n\n"
    # ...(same implementation as before)...
    try:
        if not os.path.exists(eval_file): return f"Error: Eval file not found: {eval_file}"
        eval_df = pd.read_csv(eval_file, nrows=5); eval_df.fillna("", inplace=True)
        if 'data_csv_files' not in eval_df.columns: return f"Error: Column 'data_csv_files' missing."
        all_csvs = set()
        for csv_list_str in eval_df['data_csv_files']:
            if isinstance(csv_list_str, str) and csv_list_str: all_csvs.update(fname.strip() for fname in csv_list_str.split(','))
        if not all_csvs: return "No data CSV files listed."
        for csv_filename in sorted(list(all_csvs)):
            csv_path = os.path.join(data_dir, csv_filename); table_name = os.path.splitext(csv_filename)[0]
            schema_str += f"Table: {table_name} ({csv_filename})\n"
            if os.path.exists(csv_path):
                try: df_sample = pd.read_csv(csv_path, nrows=0); schema_str += f"  Columns: {', '.join(df_sample.columns)}\n\n"
                except Exception as e: schema_str += f"  Error reading columns: {e}\n\n"
            else: schema_str += f"  Error: File not found: {csv_path}\n\n"
        return schema_str
    except Exception as e: return f"Error reading evaluation file {eval_file}: {e}"


def display_summary_metrics(summary: BenchmarkSummary, title_prefix=""):
    """Displays summary metrics in Streamlit columns."""
    st.markdown(f"#### {title_prefix} Run Summary")
    cols = st.columns(6)
    cols[0].metric("Total Cases", summary.total_processed)
    cols[1].metric("Accuracy", f"{summary.accuracy:.2f}%")
    cols[2].metric("Exact Match", f"{summary.exact_match_rate:.2f}%")
    cols[3].metric("Gen Errors", f"{summary.total_gen_errors} ({summary.gen_error_rate:.1f}%)")
    cols[4].metric("Exec Errors", f"{summary.total_exec_errors} ({summary.exec_error_rate:.1f}%)")
    cols[5].metric("Avg Latency", f"{summary.avg_latency_s:.3f}s")

def parse_filename(filename):
    """Parses the benchmark result filename to extract config details."""
    # Filename format: {model_id}_{dataset_shortname}_{prompt_filename_base}_{timestamp}.csv
    pattern = r"^(.*?)_(.*?)_(.*?)_(\d{8}_\d{6})\.csv$"
    match = re.match(pattern, filename)
    if match:
        model, dataset, prompt_base, timestamp_str = match.groups()
        try:
            # Attempt to parse timestamp for sorting/display
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = None # Or handle error
        return {"model": model, "dataset": dataset, "prompt_base": prompt_base, "timestamp_str": timestamp_str, "timestamp": timestamp, "filename": filename}
    else:
        # Fallback for potentially older filenames or different formats
        return {"model": "Unknown", "dataset": "Unknown", "prompt_base": "Unknown", "timestamp_str": "Unknown", "timestamp": None, "filename": filename}


def find_and_parse_results_files(results_dir=RESULTS_DIR):
    """Finds and parses all benchmark result CSV filenames."""
    parsed_files = []
    try:
        list_of_files = glob.glob(os.path.join(results_dir, "*.csv"))
        for f in list_of_files:
            parsed_info = parse_filename(os.path.basename(f))
            parsed_info['full_path'] = f
            parsed_files.append(parsed_info)
        # Sort by timestamp descending (newest first)
        parsed_files.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        return parsed_files
    except Exception as e:
        print(f"Error finding/parsing result files: {e}")
        return []

def generate_comparison_summary(current_summary: BenchmarkSummary, past_runs_data: List[Dict]) -> pd.DataFrame:
    """Creates a DataFrame comparing the current run to relevant past runs."""
    comparison_data = []

    # Add current run first
    if current_summary:
        current_display = current_summary.to_display_dict()
        current_display["Run Type"] = "Current Run"
        comparison_data.append(current_display)

    # Add relevant past runs
    for past_run in past_runs_data:
         # Ensure it's a comparable run (same model, same dataset)
        if (current_summary and
            past_run.get('model') == current_summary.model_identifier and
            past_run.get('dataset_shortname') == current_summary.dataset_name.split(" ")[0].lower()):

            try:
                past_df = pd.read_csv(past_run['full_path'])
                # Recalculate summary (could be stored/cached later)
                past_summary = BenchmarkSummary(
                    model_identifier=past_run.get('model', 'Unknown'),
                    dataset_name=current_summary.dataset_name, # Assume same dataset if shortname matches
                    prompt_template_file=past_run.get('prompt_base', 'Unknown') + ".md", # Reconstruct approx prompt file
                    run_timestamp=past_run.get('timestamp_str', 'Unknown'),
                    output_filename=past_run.get('filename', 'Unknown'),
                    total_processed=len(past_df)
                )
                if not past_df.empty:
                    if 'correct' in past_df: past_summary.total_correct = past_df['correct'].sum()
                    if 'exact_match' in past_df: past_summary.total_exact_match = past_df['exact_match'].sum()
                    if 'generation_error_msg' in past_df: past_summary.total_gen_errors = past_df['generation_error_msg'].fillna('').apply(bool).sum()
                    if 'error_db_exec' in past_df: past_summary.total_exec_errors = past_df['error_db_exec'].sum()
                    if 'latency_seconds' in past_df: past_summary.avg_latency_s = past_df['latency_seconds'].mean()

                past_display = past_summary.to_display_dict()
                past_display["Run Type"] = "Past Run"
                comparison_data.append(past_display)

            except Exception as e:
                print(f"Error processing past run file {past_run.get('filename', 'Unknown')}: {e}")


    if not comparison_data:
        return pd.DataFrame()

    comp_df = pd.DataFrame(comparison_data)
    # Reorder columns for better readability
    ordered_cols = [
        "Run Type", "Timestamp", "Model", "Dataset", "Prompt File",
        "Accuracy (%)", "Exact Match (%)", "Gen Errors (%)", "Exec Errors (%)",
        "Avg Latency (s)", "Total Cases", "Run File"
    ]
    comp_df = comp_df[[col for col in ordered_cols if col in comp_df.columns]]
    return comp_df


# --- Main Function to Display the Benchmark Section ---
def display_benchmark_section():
    """Renders the Streamlit UI for the benchmark section."""

    st.header("🚀 SQL Generation Benchmark")
    st.caption("Evaluate custom SQL generation models using local CSV data.")

    # --- Configuration Section ---
    st.subheader("Configuration")
    col1, col2 = st.columns([1, 2])

    prompt_files = list_prompt_files() # Get available prompt templates

    with col1:
        selected_model = st.selectbox("Select Model:", options=AVAILABLE_MODELS, key="bm_model")
        selected_dataset_name = st.selectbox("Select Dataset:", options=list(DATASETS.keys()), key="bm_dataset")
        selected_prompt_file = st.selectbox("Select Prompt Template:", options=prompt_files, key="bm_prompt_select")
        prompt_template_path = os.path.join(PROMPTS_DIR, selected_prompt_file)

        parallel_threads = st.number_input("Parallel Threads", min_value=1, max_value=10, value=1, key="bm_threads", help="Number of concurrent evaluation cases.")

    with col2:
        dataset_config = DATASETS[selected_dataset_name]
        st.info(f"**Dataset:** {dataset_config.get('description', 'N/A')}")
        schema_info = get_schema_display_benchmark(dataset_config["data_dir"], dataset_config["eval_file"])
        st.text_area("Schema Preview:", schema_info, height=150, key="bm_schema_preview")

        # Display selected prompt content
        st.markdown("**Selected Prompt Template Preview:**")
        try:
            with open(prompt_template_path, 'r') as f:
                prompt_content = f.read()
            st.text_area("Prompt Content:", prompt_content, height=100, key="bm_prompt_preview", disabled=True)
        except Exception as e:
            st.warning(f"Could not read prompt file {prompt_template_path}: {e}")


    run_button = st.button("Run Benchmark", type="primary", key="bm_run_button", use_container_width=True)

    st.divider()

    # --- Run Logic & Results Display ---
    if 'benchmark_running' not in st.session_state: st.session_state.benchmark_running = False
    if 'benchmark_results_df' not in st.session_state: st.session_state.benchmark_results_df = None
    if 'benchmark_summary' not in st.session_state: st.session_state.benchmark_summary = None
    if 'benchmark_output_file' not in st.session_state: st.session_state.benchmark_output_file = None
    if 'past_runs_for_comparison' not in st.session_state: st.session_state.past_runs_for_comparison = []

    if run_button and not st.session_state.benchmark_running:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_results_df = None; st.session_state.benchmark_summary = None
        st.session_state.benchmark_output_file = None; st.session_state.past_runs_for_comparison = []
        st.info("Benchmark run initiated...")
        st.rerun()

    if st.session_state.benchmark_running or st.session_state.benchmark_results_df is not None:
         st.subheader("📈 Benchmark Run Status & Results")

    if st.session_state.benchmark_running:
        st.button("Run Benchmark", key="bm_run_button_disabled", use_container_width=True, disabled=True)
        # --- Prepare and Run ---
        config = BenchmarkConfig(
            model_identifier=selected_model, # Use state from current selection
            eval_definition_file=dataset_config["eval_file"],
            data_dir=dataset_config["data_dir"],
            prompt_template_file=prompt_template_path, # Use full path
            parallel_threads=parallel_threads,
            dataset_name=selected_dataset_name # Store dataset name
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_short_name = selected_dataset_name.split(" ")[0].lower().replace("(","").replace(")","").replace(",","")
        prompt_filename_base = os.path.splitext(selected_prompt_file)[0]
        output_filename = f"{config.model_identifier}_{dataset_short_name}_{prompt_filename_base}_{timestamp}.csv"
        output_filepath = os.path.join(RESULTS_DIR, output_filename)

        status_text = st.empty(); progress_bar = st.progress(0.0, text="Initializing...")
        log_expander = st.expander("Show Run Logs", expanded=False); log_placeholder = log_expander.empty()

        def update_progress(value):
            value = max(0.0, min(1.0, value)); progress_text = f"Running benchmark... {int(value * 100)}% complete"
            try: progress_bar.progress(value, text=progress_text); status_text.text(progress_text)
            except Exception as e: print(f"Streamlit progress update error: {e}")

        start_time = time.time()
        try:
            results_df, summary = run_benchmark_logic(config, progress_callback=update_progress)
            summary.run_timestamp = timestamp # Store timestamp string in summary
            summary.output_filename = output_filepath # Store filename in summary

            st.session_state.benchmark_results_df = results_df
            st.session_state.benchmark_summary = summary
            st.session_state.benchmark_output_file = output_filepath

            # Find relevant past runs *after* the current run finishes
            all_past_runs = find_and_parse_results_files()
            st.session_state.past_runs_for_comparison = [
                run for run in all_past_runs
                if run['model'] == config.model_identifier and run['dataset'] == dataset_short_name and run['filename'] != os.path.basename(output_filepath)
            ]


            if not results_df.empty:
                 results_df.to_csv(output_filepath, index=False, float_format="%.3f")
                 status_text.success(f"Benchmark finished in {time.time() - start_time:.2f}s. Results saved.")
                 log_placeholder.info("Run completed. Check summary and results below.")
            else:
                 status_text.warning("Benchmark run completed, but no results were generated.")
                 log_placeholder.warning("Run completed, but no results were generated.")

        except Exception as e:
            status_text.error(f"Benchmark run failed: {e}")
            import traceback; log_placeholder.code(traceback.format_exc())
        finally:
            st.session_state.benchmark_running = False
            try: progress_bar.empty()
            except: pass
            st.rerun() # Rerun to display results

    # --- Display Current Run Results (if available) ---
    if st.session_state.benchmark_results_df is not None:
        st.markdown(f"**Run File:** `{os.path.basename(st.session_state.benchmark_output_file or 'Unknown')}`")
        if st.session_state.benchmark_summary:
            display_summary_metrics(st.session_state.benchmark_summary, title_prefix="Current")

        # --- Comparison Summary Table ---
        if st.session_state.past_runs_for_comparison:
             st.markdown("#### Comparison with Past Runs (Same Model & Dataset)")
             comparison_df = generate_comparison_summary(st.session_state.benchmark_summary, st.session_state.past_runs_for_comparison)
             if not comparison_df.empty:
                 st.dataframe(comparison_df, hide_index=True)
             else:
                 st.caption("Could not generate comparison data.")
        else:
             st.caption("No relevant past runs found for comparison.")


        st.markdown("#### Detailed Results (Current Run)")
        # Configure dataframe display for side-by-side feel
        st.dataframe(
            st.session_state.benchmark_results_df,
            column_config={
                "query": st.column_config.TextColumn("Gold Query", width="medium"),
                "generated_query": st.column_config.TextColumn("Generated Query", width="medium"),
                "question": st.column_config.TextColumn("Question", width="medium"),
                "prompt_generated": st.column_config.TextColumn("Full Prompt", width="small", help="The full prompt sent to the generator"),
                "correct": st.column_config.CheckboxColumn("Correct (Subset?)", width="small"),
                "exact_match": st.column_config.CheckboxColumn("Exact Match", width="small"),
                "latency_seconds": st.column_config.NumberColumn("Latency (s)", format="%.3f", width="small"),
                "generation_error_msg": st.column_config.TextColumn("Gen Error", width="small"),
                "exec_error_msg": st.column_config.TextColumn("Exec/Eval Error", width="small"),
            },
             use_container_width=True
        )

        # Download Button
        if st.session_state.benchmark_output_file and os.path.exists(st.session_state.benchmark_output_file):
            try:
                with open(st.session_state.benchmark_output_file, "rb") as fp:
                    st.download_button(label="Download Results as CSV", data=fp,
                                       file_name=os.path.basename(st.session_state.benchmark_output_file), mime='text/csv')
            except Exception as e: st.warning(f"Could not prepare results for download: {e}")

    # --- Past Results Section ---
    st.divider()
    st.subheader(" vergangene Benchmarks") # Past Benchmarks

    all_parsed_files = find_and_parse_results_files()

    if not all_parsed_files:
        st.info("No past benchmark results found.")
    else:
        # Create a display name including model, dataset, prompt, timestamp
        display_options = { f"{p['model']} | {p['dataset']} | {p['prompt_base']} | {p['timestamp_str']}": p['full_path'] for p in all_parsed_files }

        selected_past_display_name = st.selectbox(
            "Select Past Run to View:",
            options=list(display_options.keys()),
            key="bm_past_run_select"
        )
        if selected_past_display_name:
            selected_past_filepath = display_options[selected_past_display_name]
            st.caption(f"Displaying results from: `{os.path.basename(selected_past_filepath)}`")
            try:
                past_df = pd.read_csv(selected_past_filepath)
                # Recalculate summary
                parsed_info = parse_filename(os.path.basename(selected_past_filepath)) # Reparse for consistency
                past_summary = BenchmarkSummary(
                    model_identifier=parsed_info.get('model','?'), dataset_name=parsed_info.get('dataset','?'),
                    prompt_template_file=parsed_info.get('prompt_base','?')+".md", run_timestamp=parsed_info.get('timestamp_str','?'),
                    output_filename=selected_past_filepath, total_processed=len(past_df)
                )
                if not past_df.empty:
                    if 'correct' in past_df: past_summary.total_correct = past_df['correct'].sum()
                    if 'exact_match' in past_df: past_summary.total_exact_match = past_df['exact_match'].sum()
                    if 'generation_error_msg' in past_df: past_summary.total_gen_errors = past_df['generation_error_msg'].fillna('').apply(bool).sum()
                    if 'error_db_exec' in past_df: past_summary.total_exec_errors = past_df['error_db_exec'].sum()
                    if 'latency_seconds' in past_df: past_summary.avg_latency_s = past_df['latency_seconds'].mean()

                display_summary_metrics(past_summary, title_prefix="Selected Past")
                st.dataframe(past_df,
                             column_config={ # Apply similar config for consistency
                                "query": st.column_config.TextColumn("Gold Query", width="medium"),
                                "generated_query": st.column_config.TextColumn("Generated Query", width="medium"),
                                "question": st.column_config.TextColumn("Question", width="medium"),
                             },
                             use_container_width=True)
            except Exception as e:
                st.error(f"Error loading past results from {selected_past_display_name}: {e}")
