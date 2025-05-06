# dashboard_main.py
import streamlit as st

# --- Import the benchmark UI function ---
try:
    # Assuming benchmark_ui.py is in the same directory
    from benchmark_ui import display_benchmark_section # Use the correct function name
    benchmark_available = True
except ImportError as e:
    print(f"Warning: Could not import benchmark_ui: {e}. Benchmark functionality will be disabled.")
    benchmark_available = False
    # Define a placeholder function
    def display_benchmark_section():
        st.error("Benchmark module ('benchmark_ui.py') could not be loaded. Please ensure it exists and dependencies are met.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="My LLM Application")

# --- Sidebar Navigation ---
st.sidebar.title("📋 Table of Contents")
st.sidebar.divider()

# Define available sections
sections = ["🏠 Home"]
if benchmark_available:
    sections.append("📊 Benchmark")
# Add more sections here if needed later
# sections.append("⚙️ Settings")

app_mode = st.sidebar.radio(
    "Go to:",
    sections,
    key="app_mode"
)
st.sidebar.divider()
st.sidebar.info("Application Status: Running") # Example sidebar info

# --- Main Content Area ---

if app_mode == "🏠 Home":
    st.title("🏠 Welcome!")
    st.header("Work in Progress")
    st.write("This is the main home page of the application.")
    st.info("Select 'Benchmark' from the sidebar to evaluate SQL generation models.")
    # Add any other introductory content here

elif app_mode == "📊 Benchmark":
    # Call the function from benchmark_ui.py to render its content
    display_benchmark_section() # Call the imported function

# Add elif blocks for other sections if you add them later
# elif app_mode == "⚙️ Settings":
#     st.title("⚙️ Settings")
#     st.write("Application settings will go here.")

