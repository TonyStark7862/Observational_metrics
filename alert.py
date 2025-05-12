import streamlit as st
import time
import datetime
import requests  # For token URL check
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

# --- For Auto-Refresh ---
from streamlit_autorefresh import st_autorefresh

# --- Mock External Functions ---
# TODO: Replace these with your actual implementations
def abc_response(model: str, prompt: str) -> Tuple[str, float, int, int]:
    """
    Mock for your LLM API call.
    Returns a tuple: (response_text: str, latency_seconds: float, input_tokens: int, output_tokens: int)
    Errors should be indicated in response_text or by raising an exception.
    """
    # st.sidebar.write(f"MOCK abc_response: model='{model}', prompt='{prompt[:30]}...'") # Debugging

    # Simulate different scenarios
    if "error_test_prompt" in prompt:
        time.sleep(0.2)
        # Error message is returned as response_text
        return "Simulated LLM error from 'error_test_prompt'", 0.2, 5, 0
    if "timeout_test_prompt" in prompt: # This would be a real timeout in your actual call
        time.sleep(2)
        # For a timeout, your actual function might raise an exception or return an error string
        return "Simulated LLM timeout", 2.0, 5, 0
    if "rate_limit_test_prompt" in prompt:
        time.sleep(0.1)
        # Rate limit error message returned as response_text
        return "Simulated rate limit exceeded", 0.1, 5, 0

    # Successful response
    time.sleep(0.3 + (len(prompt) % 3) * 0.05) # Vary latency a bit
    response_text_content = f"Mock response to: {prompt}"
    if prompt == "Hi, health check. Respond with only 'Hi' and nothing else.":
        response_text_content = "Hi"
    
    latency = 0.5
    input_tokens_count = len(prompt.split())
    output_tokens_count = len(response_text_content.split())

    return response_text_content, latency, input_tokens_count, output_tokens_count

def send_mail(subject: str, body: str):
    """Mock for your email sending function."""
    print(f"--- SENDING EMAIL (MOCK) ---")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print(f"--- EMAIL SENT (MOCK) ---")
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
    # To store last metrics for display
    last_latency: float = 0.0
    last_input_tokens: int = 0
    last_output_tokens: int = 0


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
        if config.current_status.status != "RATE_LIMITED":
            config.current_status.status = "RATE_LIMITED"
            remaining_cooldown = config.rate_limited_until - current_time
            config.current_status.details = f"Rate limited. Next check possible in {remaining_cooldown:.0f} seconds."
            # No need to update last_error here, it's set when rate limit is triggered
        return

    if current_time - config.current_status.last_checked_at < config.interval_seconds:
        return

    config.current_status.last_checked_at = current_time
    error_occurred = False
    error_msg_for_email = ""
    
    try:
        response_text, latency, input_tokens, output_tokens = abc_response(config.model_name, config.prompt)
        config.last_latency = latency
        config.last_input_tokens = input_tokens
        config.last_output_tokens = output_tokens

        # Check for specific error strings that might indicate a rate limit or other direct issues
        # Your actual abc_response should return these specific strings in response_text if such errors occur
        if "rate limit" in response_text.lower():
            error_occurred = True
            error_msg_for_email = response_text # The response_text itself is the error
            config.rate_limited_until = current_time + config.rate_limit_cooldown_seconds
            config.current_status.status = "RATE_LIMITED"
            details = f"Rate limit detected by API. Cooldown for {config.rate_limit_cooldown_seconds}s. API Message: {response_text}"
            config.current_status.details = details
            config.current_status.last_error = error_msg_for_email # Set last_error here
            subject = f"CRITICAL: LLM API ('abc_response') Rate Limited - Health Check"
            body = _format_email_body("ABC Response (Rate Limit)", error_msg_for_email, config.__dict__)
            send_mail(subject, body)
            return # Exit after setting rate limit and sending mail

        # Check if the response text matches the expected successful response
        elif response_text != config.expected_response:
            error_occurred = True
            error_msg_for_email = f"Unexpected response. Expected '{config.expected_response}', Got: '{response_text}'. Latency: {latency:.2f}s, InTokens: {input_tokens}, OutTokens: {output_tokens}"
        
        # If no specific error string and response matches expected, it's a success
        else:
            config.current_status.status = "OK"
            config.current_status.last_error = None
            details = f"OK. Response: '{response_text}'. Latency: {latency:.2f}s, InTokens: {input_tokens}, OutTokens: {output_tokens}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
            config.current_status.details = details
            if config.rate_limited_until > 0: # Was previously rate limited
                 config.rate_limited_until = 0 # Clear rate limit
                 st.toast("ABC Response rate limit cooldown lifted after successful check.", icon="✅")

    except Exception as e: # Catches exceptions raised by abc_response() itself
        error_occurred = True
        error_msg_for_email = f"Exception during abc_response call: {str(e)}"
        config.last_latency, config.last_input_tokens, config.last_output_tokens = 0,0,0 # Reset metrics on exception


    if error_occurred: # This block handles errors not covered by the specific "rate limit" path above
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
        return

    config.current_status.last_checked_at = current_time
    error_occurred = False
    error_msg_for_email = ""
    status_code_str = "N/A"

    try:
        response = requests.get(config.url, timeout=10)
        status_code_str = str(response.status_code)
        response.raise_for_status()
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

    if 'abc_config' not in st.session_state:
        st.session_state.abc_config = ABCConfig()
    if 'token_url_config' not in st.session_state:
        st.session_state.token_url_config = TokenURLConfig(url="https://jsonplaceholder.typicode.com/todos/1") # TODO: Replace with actual URL
    if 'health_check_refresh_interval' not in st.session_state:
         st.session_state.health_check_refresh_interval = 60

    abc_cfg = st.session_state.abc_config
    token_cfg = st.session_state.token_url_config

    if st.session_state.get('app_mode') == "🩺 Health Check":
        refresh_interval_ms = st.session_state.health_check_refresh_interval * 1000
        st_autorefresh(interval=refresh_interval_ms, limit=None, key="healthcheck_autorefresher")

    perform_abc_check(abc_cfg)
    perform_token_url_check(token_cfg)

    st.subheader("ABC Response Health")
    with st.container(border=True):
        col1_abc_toggle, col1_abc_interval = st.columns(2)
        abc_cfg.enabled = col1_abc_toggle.toggle("Enable ABC Monitoring", value=abc_cfg.enabled, key="abc_enabled_toggle")

        if abc_cfg.enabled:
            new_interval_abc = col1_abc_interval.number_input(
                "Check Interval (seconds)", min_value=10, max_value=3600,
                value=abc_cfg.interval_seconds, step=10, key="abc_interval_input",
                help="How often to attempt a health check for ABC response."
            )
            if new_interval_abc != abc_cfg.interval_seconds: abc_cfg.interval_seconds = new_interval_abc

        with st.expander("ABC Health Check Configuration", expanded=False):
            abc_cfg.model_name = st.text_input("Model for Health Check", value=abc_cfg.model_name, key="abc_model_input", disabled=not abc_cfg.enabled)
            abc_cfg.prompt = st.text_area("Health Check Prompt", value=abc_cfg.prompt, key="abc_prompt_input", height=100, disabled=not abc_cfg.enabled)
            abc_cfg.expected_response = st.text_input("Expected Exact Response", value=abc_cfg.expected_response, key="abc_expected_input", disabled=not abc_cfg.enabled)
            abc_cfg.rate_limit_cooldown_seconds = st.number_input(
                "Rate Limit Cooldown (s)", min_value=60, max_value=3600,
                value=abc_cfg.rate_limit_cooldown_seconds, step=60, key="abc_cooldown_input",
                disabled=not abc_cfg.enabled, help="How long to wait after a rate limit error before retrying."
            )

        status_color = "grey"; status_icon = "❓"
        if abc_cfg.current_status.status == "OK": status_color = "green"; status_icon = "✅"
        elif abc_cfg.current_status.status == "ERROR": status_color = "red"; status_icon = "❌"
        elif abc_cfg.current_status.status == "RATE_LIMITED": status_color = "orange"; status_icon = "⏳"
        elif abc_cfg.current_status.status == "DISABLED": status_color = "grey"; status_icon = "⏸️"

        st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{status_icon} {abc_cfg.current_status.status}</span>", unsafe_allow_html=True)
        st.caption(f"**Details:** {abc_cfg.current_status.details}")
        if abc_cfg.last_latency > 0 and abc_cfg.current_status.status == "OK":
             st.caption(f"Last successful call: Latency: {abc_cfg.last_latency:.2f}s, InTokens: {abc_cfg.last_input_tokens}, OutTokens: {abc_cfg.last_output_tokens}")

        if abc_cfg.current_status.last_error and abc_cfg.current_status.status not in ["OK", "DISABLED"]:
            st.error(f"Last Error Recorded: {abc_cfg.current_status.last_error}")
        if abc_cfg.current_status.status == "RATE_LIMITED":
             st.warning(f"Rate limited. Next attempt after: {datetime.datetime.fromtimestamp(abc_cfg.rate_limited_until).strftime('%Y-%m-%d %H:%M:%S') if abc_cfg.rate_limited_until > 0 else 'N/A'}")

    st.divider()

    st.subheader("Token URL Health")
    with st.container(border=True):
        col1_token_toggle, col1_token_interval_display = st.columns(2)
        token_cfg.enabled = col1_token_toggle.toggle("Enable Token URL Monitoring", value=token_cfg.enabled, key="token_enabled_toggle")

        if token_cfg.enabled:
            col1_token_interval_display.info(f"Check Interval: {token_cfg.interval_seconds} seconds (fixed for this example)")
            # If you want to make token URL interval configurable:
            # new_interval_token = col1_token_interval_display.number_input(...)
            # if new_interval_token != token_cfg.interval_seconds: token_cfg.interval_seconds = new_interval_token


        with st.expander("Token URL Configuration", expanded=False):
            new_token_url = st.text_input("Token URL to Check", value=token_cfg.url, key="token_url_input", disabled=not token_cfg.enabled)
            if new_token_url != token_cfg.url: token_cfg.url = new_token_url

        status_color_token = "grey"; token_icon = "❓"
        if token_cfg.current_status.status == "OK": status_color_token = "green"; token_icon = "✅"
        elif token_cfg.current_status.status == "ERROR": status_color_token = "red"; token_icon = "❌"
        elif token_cfg.current_status.status == "DISABLED": status_color_token = "grey"; token_icon = "⏸️"

        st.markdown(f"**Status:** <span style='color:{status_color_token}; font-weight:bold;'>{token_icon} {token_cfg.current_status.status}</span>", unsafe_allow_html=True)
        st.caption(f"**Details:** {token_cfg.current_status.details}")
        if token_cfg.current_status.last_error and token_cfg.current_status.status not in ["OK", "DISABLED"]:
            st.error(f"Last Error Recorded: {token_cfg.current_status.last_error}")

    st.divider()
    if st.button("Force Refresh All Statuses Now", key="manual_refresh_health_all"):
        if abc_cfg.enabled: abc_cfg.current_status.last_checked_at = 0
        if token_cfg.enabled: token_cfg.current_status.last_checked_at = 0
        st.rerun()

    st.caption(f"This page attempts to auto-refresh every {st.session_state.health_check_refresh_interval} seconds if active, triggering checks based on their individual intervals.")
    new_refresh_slider_val = st.slider("Set Auto-Refresh Interval (seconds for this tab)", 10, 300, st.session_state.health_check_refresh_interval, 10, key="health_refresh_slider")
    if new_refresh_slider_val != st.session_state.health_check_refresh_interval:
        st.session_state.health_check_refresh_interval = new_refresh_slider_val
        st.rerun()
