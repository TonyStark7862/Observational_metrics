import streamlit as st
import time
import datetime
import requests  # For token URL check
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# --- For Auto-Refresh ---
from streamlit_autorefresh import st_autorefresh

# --- Mock External Functions ---
# TODO: Replace these with your actual implementations
def abc_response(model: str, prompt: str) -> Dict[str, Any]:
    """
    Mock for your LLM API call.
    Returns a dict like: {'response_text': '...', 'latency_seconds': 0.5, 'input_tokens': 10, 'output_tokens': 5, 'error_msg': None}
    """
    # st.sidebar.write(f"MOCK abc_response: model='{model}', prompt='{prompt[:30]}...'") # Debugging
    # Simulate different scenarios
    if "error_test_prompt" in prompt:
        time.sleep(0.2)
        return {'response_text': '', 'latency_seconds': 0.2, 'input_tokens': 5, 'output_tokens': 0, 'error_msg': "Simulated LLM error from 'error_test_prompt'"}
    if "timeout_test_prompt" in prompt:
        time.sleep(2)
        return {'response_text': '', 'latency_seconds': 2.0, 'input_tokens': 5, 'output_tokens': 0, 'error_msg': "Simulated LLM timeout"}
    if "rate_limit_test_prompt" in prompt:
        time.sleep(0.1)
        return {'response_text': '', 'latency_seconds': 0.1, 'input_tokens': 5, 'output_tokens': 0, 'error_msg': "Simulated rate limit exceeded"}

    # Successful response
    time.sleep(0.3 + (len(prompt) % 3) * 0.05) # Vary latency a bit
    response_text = f"Mock response to: {prompt}"
    if prompt == "Hi, health check. Respond with only 'Hi' and nothing else.":
        response_text = "Hi"

    return {
        'response_text': response_text,
        'latency_seconds': 0.5,
        'input_tokens': len(prompt.split()),
        'output_tokens': len(response_text.split()),
        'error_msg': None
    }

def send_mail(subject: str, body: str):
    """Mock for your email sending function."""
    print(f"--- SENDING EMAIL (MOCK) ---")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print(f"--- EMAIL SENT (MOCK) ---")
    # Use st.toast for UI feedback in the mock
    st.toast(f"Mock Email Sent: {subject}", icon="💌")

# --- Health Check Configuration and State ---
@dataclass
class CheckStatus:
    status: str = "UNKNOWN"  # OK, ERROR, DISABLED, RATE_LIMITED
    last_checked_at: float = 0.0
    last_error: Optional[str] = None
    details: str = "Not yet checked."

@dataclass
class ABCConfig:
    enabled: bool = True
    interval_seconds: int = 60
    model_name: str = "default_health_check_model"
    prompt: str = "Hi, health check. Respond with only 'Hi' and nothing else."
    expected_response: str = "Hi"
    rate_limit_cooldown_seconds: int = 300  # 5 minutes
    rate_limited_until: float = 0.0
    current_status: CheckStatus = field(default_factory=CheckStatus)

@dataclass
class TokenURLConfig:
    enabled: bool = True
    url: str = "https://jsonplaceholder.typicode.com/todos/1" # Placeholder, replace
    interval_seconds: int = 60
    current_status: CheckStatus = field(default_factory=CheckStatus)

# --- Health Check Logic ---

def _format_email_body(check_name: str, error_details: str, config_details: Optional[dict] = None) -> str:
    body = f"ALERT! Health Check Failure for {check_name}.\n\n"
    body += f"Timestamp: {datetime.datetime.utcnow().isoformat()} Z\n"
    body += f"Error Details: {error_details}\n\n"
    if config_details:
        body += "Configuration at time of error:\n"
        # Avoid serializing complex objects like CheckStatus directly if it's part of config_details
        safe_config_details = {k: v for k, v in config_details.items() if not isinstance(v, CheckStatus)}
        for k, v in safe_config_details.items():
            body += f"  {k}: {v}\n"
    body += "\nPlease investigate."
    return body

def perform_abc_check(config: ABCConfig):
    if not config.enabled:
        if config.current_status.status != "DISABLED":
            config.current_status.status = "DISABLED"
            config.current_status.details = "Monitoring is disabled."
            config.current_status.last_error = None
        return

    current_time = time.time()
    if current_time < config.rate_limited_until:
        if config.current_status.status != "RATE_LIMITED": # Update status only if it changed
            config.current_status.status = "RATE_LIMITED"
            remaining_cooldown = config.rate_limited_until - current_time
            config.current_status.details = f"Rate limited. Next check possible in {remaining_cooldown:.0f} seconds."
            config.current_status.last_error = "Currently under rate limit cooldown."
        return

    if current_time - config.current_status.last_checked_at < config.interval_seconds:
        return # Not time yet

    # st.sidebar.write(f"Performing ABC Check at {datetime.datetime.now()}") # Debugging
    config.current_status.last_checked_at = current_time
    error_occurred = False
    error_msg_for_email = ""
    latency = 0
    response_text_received = ""

    try:
        result = abc_response(config.model_name, config.prompt)
        latency = result.get('latency_seconds', 0)
        response_text_received = result.get('response_text', '')

        if result.get('error_msg'):
            error_occurred = True
            error_msg_for_email = result['error_msg']
            if "rate limit" in error_msg_for_email.lower(): # Simple check for simulated rate limit
                config.rate_limited_until = current_time + config.rate_limit_cooldown_seconds
                config.current_status.status = "RATE_LIMITED"
                details = f"Rate limit detected. Cooldown for {config.rate_limit_cooldown_seconds}s. Last error: {error_msg_for_email}"
                config.current_status.details = details
                config.current_status.last_error = error_msg_for_email
                subject = f"CRITICAL: LLM API ('abc_response') Rate Limited - Health Check"
                body = _format_email_body("ABC Response (Rate Limit)", error_msg_for_email, config.__dict__)
                send_mail(subject, body)
                return

        elif response_text_received != config.expected_response:
            error_occurred = True
            error_msg_for_email = f"Unexpected response. Expected '{config.expected_response}', Got: '{response_text_received}'. Latency: {latency:.2f}s"
        else: # Success
            config.current_status.status = "OK"
            config.current_status.last_error = None
            details = f"OK. Response: '{response_text_received}'. Latency: {latency:.2f}s. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
            config.current_status.details = details
            if config.rate_limited_until > 0: # Was previously rate limited
                 config.rate_limited_until = 0 # Clear rate limit
                 st.toast("ABC Response rate limit cooldown lifted after successful check.", icon="✅")


    except Exception as e:
        error_occurred = True
        error_msg_for_email = f"Exception during abc_response check: {str(e)}"

    if error_occurred:
        config.current_status.status = "ERROR"
        config.current_status.last_error = error_msg_for_email
        details = f"Error: {error_msg_for_email}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
        config.current_status.details = details
        subject = f"CRITICAL: LLM API ('abc_response') Health Check Failed"
        body = _format_email_body("ABC Response", error_msg_for_email, config.__dict__)
        send_mail(subject, body)

def perform_token_url_check(config: TokenURLConfig):
    if not config.enabled:
        if config.current_status.status != "DISABLED":
            config.current_status.status = "DISABLED"
            config.current_status.details = "Monitoring is disabled."
            config.current_status.last_error = None
        return

    current_time = time.time()
    if current_time - config.current_status.last_checked_at < config.interval_seconds:
        return # Not time yet

    # st.sidebar.write(f"Performing Token URL Check at {datetime.datetime.now()}") # Debugging
    config.current_status.last_checked_at = current_time
    error_occurred = False
    error_msg_for_email = ""
    status_code_str = "N/A"

    try:
        response = requests.get(config.url, timeout=10)
        status_code_str = str(response.status_code)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        config.current_status.status = "OK"
        config.current_status.last_error = None
        details = f"OK. Status Code: {status_code_str}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
        config.current_status.details = details

    except requests.exceptions.RequestException as e:
        error_occurred = True
        error_msg_for_email = f"Error connecting to token URL '{config.url}': {str(e)}. Status Code: {status_code_str}"

    if error_occurred:
        config.current_status.status = "ERROR"
        config.current_status.last_error = error_msg_for_email
        details = f"Error: {error_msg_for_email}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
        config.current_status.details = details
        subject = f"CRITICAL: Token URL ('{config.url}') Health Check Failed"
        body_config_details = {"url": config.url, "interval_seconds": config.interval_seconds}
        body = _format_email_body(f"Token URL ({config.url})", error_msg_for_email, body_config_details)
        send_mail(subject, body)

# --- Streamlit UI for Health Check Tab ---
def display_health_check_section():
    st.title("🩺 System Health Checks")
    # st.caption(f"Page last script run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Initialize configurations in session state if not present
    if 'abc_config' not in st.session_state:
        st.session_state.abc_config = ABCConfig()
    if 'token_url_config' not in st.session_state:
        # TODO: User should replace this URL
        st.session_state.token_url_config = TokenURLConfig(url="https://jsonplaceholder.typicode.com/todos/1")
    if 'health_check_refresh_interval' not in st.session_state:
         st.session_state.health_check_refresh_interval = 60 # Default auto-refresh interval in seconds

    abc_cfg = st.session_state.abc_config
    token_cfg = st.session_state.token_url_config

    # Auto-refresh controller for this specific tab
    # Only run autorefresh if this tab is active
    if st.session_state.get('app_mode') == "🩺 Health Check":
        refresh_interval_ms = st.session_state.health_check_refresh_interval * 1000
        st_autorefresh(interval=refresh_interval_ms, limit=None, key="healthcheck_autorefresher")
        # The checks below will run on each refresh if their individual interval has passed

    # Perform checks (these will run on every script rerun if conditions are met)
    perform_abc_check(abc_cfg)
    perform_token_url_check(token_cfg)

    # --- UI Layout ---
    st.subheader("ABC Response Health")
    with st.container(border=True):
        col1_abc_toggle, col1_abc_interval = st.columns(2)
        abc_cfg.enabled = col1_abc_toggle.toggle("Enable ABC Monitoring", value=abc_cfg.enabled, key="abc_enabled_toggle")

        if abc_cfg.enabled:
            new_interval = col1_abc_interval.number_input(
                "Check Interval (seconds)",
                min_value=10, # Min interval can be lower now with auto-refresh
                max_value=3600,
                value=abc_cfg.interval_seconds,
                step=10,
                key="abc_interval_input",
                help="How often to attempt a health check for ABC response."
            )
            if new_interval != abc_cfg.interval_seconds:
                abc_cfg.interval_seconds = new_interval
                # No explicit rerun needed, state change handles it

        with st.expander("ABC Health Check Configuration", expanded=False):
            abc_cfg.model_name = st.text_input("Model for Health Check", value=abc_cfg.model_name, key="abc_model_input", disabled=not abc_cfg.enabled)
            abc_cfg.prompt = st.text_area("Health Check Prompt", value=abc_cfg.prompt, key="abc_prompt_input", height=100, disabled=not abc_cfg.enabled)
            abc_cfg.expected_response = st.text_input("Expected Exact Response", value=abc_cfg.expected_response, key="abc_expected_input", disabled=not abc_cfg.enabled)
            abc_cfg.rate_limit_cooldown_seconds = st.number_input(
                "Rate Limit Cooldown (s)",
                min_value=60, max_value=3600,
                value=abc_cfg.rate_limit_cooldown_seconds,
                step=60, key="abc_cooldown_input", disabled=not abc_cfg.enabled,
                help="How long to wait after a rate limit error before retrying."
            )

        status_color = "grey"
        status_icon = "❓"
        if abc_cfg.current_status.status == "OK": status_color = "green"; status_icon = "✅"
        elif abc_cfg.current_status.status == "ERROR": status_color = "red"; status_icon = "❌"
        elif abc_cfg.current_status.status == "RATE_LIMITED": status_color = "orange"; status_icon = "⏳"
        elif abc_cfg.current_status.status == "DISABLED": status_color = "grey"; status_icon = "⏸️"

        st.markdown(f"""
        **Status:** <span style='color:{status_color}; font-weight:bold;'>{status_icon} {abc_cfg.current_status.status}</span>
        """, unsafe_allow_html=True)
        st.caption(f"**Details:** {abc_cfg.current_status.details}")
        if abc_cfg.current_status.last_error and abc_cfg.current_status.status not in ["OK", "DISABLED"]:
            st.error(f"Last Error Recorded: {abc_cfg.current_status.last_error}")
        if abc_cfg.current_status.status == "RATE_LIMITED":
             st.warning(f"Rate limited. Next attempt after: {datetime.datetime.fromtimestamp(abc_cfg.rate_limited_until).strftime('%Y-%m-%d %H:%M:%S') if abc_cfg.rate_limited_until > 0 else 'N/A'}")


    st.divider()

    st.subheader("Token URL Health")
    with st.container(border=True):
        col1_token_toggle, col1_token_interval_display = st.columns(2) # Interval display is just text here
        token_cfg.enabled = col1_token_toggle.toggle("Enable Token URL Monitoring", value=token_cfg.enabled, key="token_enabled_toggle")

        if token_cfg.enabled:
            col1_token_interval_display.info(f"Check Interval: {token_cfg.interval_seconds} seconds (fixed)")
            # To make Token URL interval configurable, add a number_input similar to ABC's

        with st.expander("Token URL Configuration", expanded=False):
            new_token_url = st.text_input("Token URL to Check", value=token_cfg.url, key="token_url_input", disabled=not token_cfg.enabled)
            if new_token_url != token_cfg.url:
                token_cfg.url = new_token_url
                # No explicit rerun needed

        status_color_token = "grey"; token_icon = "❓"
        if token_cfg.current_status.status == "OK": status_color_token = "green"; token_icon = "✅"
        elif token_cfg.current_status.status == "ERROR": status_color_token = "red"; token_icon = "❌"
        elif token_cfg.current_status.status == "DISABLED": status_color_token = "grey"; token_icon = "⏸️"

        st.markdown(f"""
        **Status:** <span style='color:{status_color_token}; font-weight:bold;'>{token_icon} {token_cfg.current_status.status}</span>
        """, unsafe_allow_html=True)
        st.caption(f"**Details:** {token_cfg.current_status.details}")
        if token_cfg.current_status.last_error and token_cfg.current_status.status not in ["OK", "DISABLED"]:
            st.error(f"Last Error Recorded: {token_cfg.current_status.last_error}")

    st.divider()
    if st.button("Force Refresh All Statuses Now", key="manual_refresh_health_all"):
        if abc_cfg.enabled:
            abc_cfg.current_status.last_checked_at = 0 # Force recheck
        if token_cfg.enabled:
            token_cfg.current_status.last_checked_at = 0 # Force recheck
        st.rerun()

    st.caption(f"This page auto-refreshes every {st.session_state.health_check_refresh_interval} seconds if active.")
    new_refresh_interval = st.slider("Set Auto-Refresh Interval (seconds for this tab)", 10, 300, st.session_state.health_check_refresh_interval, 10, key="health_refresh_slider")
    if new_refresh_interval != st.session_state.health_check_refresh_interval:
        st.session_state.health_check_refresh_interval = new_refresh_interval
        st.rerun()
