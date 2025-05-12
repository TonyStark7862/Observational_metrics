import streamlit as st

# --- Import the benchmark UI function ---
try:
    from benchmark_ui import display_benchmark_section
    benchmark_available = True
except ImportError as e:
    print(f"Warning: Could not import benchmark_ui: {e}. Benchmark functionality will be disabled.")
    benchmark_available = False
    # Define a placeholder function
    def display_benchmark_section():
        st.error("Benchmark module ('benchmark_ui.py') could not be loaded. Please ensure it exists and dependencies are met.")

# --- Import the health check UI function ---
try:
    from alert import display_health_check_section # Assuming alert.py is in the same directory
    health_check_available = True
except ImportError as e:
    print(f"Warning: Could not import alert: {e}. Health Check functionality will be disabled.")
    # st.error(f"DEBUG: Alert import error: {e}") # For debugging during development
    health_check_available = False
    # Define a placeholder function
    def display_health_check_section():
        st.error("Health Check module ('alert.py') could not be loaded. Please ensure it exists and dependencies are met.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="LLM Observability Dashboard")

# --- Sidebar Navigation ---
st.sidebar.title("📋 Dashboard Menu")
st.sidebar.divider()

# Define available sections
sections = ["🏠 Home"]
if benchmark_available:
    sections.append("📊 Benchmark")
if health_check_available:
    sections.append("🩺 Health Check")

# Store current selection in session_state to persist across reruns/refreshes
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "🏠 Home"

# Use st.session_state.app_mode to set the index for the radio button
try:
    current_selection_index = sections.index(st.session_state.app_mode)
except ValueError: # If current app_mode is not in sections (e.g. after code change)
    current_selection_index = 0
    st.session_state.app_mode = sections[0]


app_mode_selection = st.sidebar.radio(
    "Navigate to:",
    sections,
    index=current_selection_index,
    key="app_mode_selector"
)

# Update session_state and rerun if selection changes
if app_mode_selection != st.session_state.app_mode:
    st.session_state.app_mode = app_mode_selection
    st.rerun()


st.sidebar.divider()
st.sidebar.info(f"Version: 1.1 ({datetime.date.today().strftime('%Y-%m-%d')})")
st.sidebar.caption(f"Current Tab: {st.session_state.app_mode}")


# --- Main Content Area ---

if st.session_state.app_mode == "🏠 Home":
    st.title("🏠 Welcome to the LLM Observability Dashboard!")
    st.markdown("""
    This dashboard provides tools to benchmark your Text-to-SQL Language Learning Models (LLMs)
    and monitor the health of critical system components.

    **👈 Select a section from the sidebar to get started:**

    * **📊 Benchmark:** Evaluate the performance of different SQL generation models against your datasets.
    * **🩺 Health Check:** Monitor the status of your LLM API and other dependent services.

    This system is designed to help you ensure reliability and quality in your LLM applications.
    """)

elif st.session_state.app_mode == "📊 Benchmark":
    if benchmark_available:
        display_benchmark_section()
    else:
        st.error("The Benchmark module is currently unavailable. Please check the logs.")

elif st.session_state.app_mode == "🩺 Health Check":
    if health_check_available:
        display_health_check_section()
    else:
        st.error("The Health Check module is currently unavailable. Please check the logs.")
