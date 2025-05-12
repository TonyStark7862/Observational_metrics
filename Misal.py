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

    if "error_test_prompt" in prompt:
        time.sleep(0.2)
        return "Simulated LLM error from 'error_test_prompt'", 0.2, 5, 0
    if "timeout_test_prompt" in prompt:
        time.sleep(2)
        return "Simulated LLM timeout", 2.0, 5, 0
    if "rate_limit_test_prompt" in prompt:
        time.sleep(0.1)
        return "Simulated rate limit exceeded", 0.1, 5, 0
    
    if "intermittent_fail_prompt" in prompt: # For testing recovery and re-alerting
        # To test 24h reminder, you'd need to manipulate system time or the last_reminder_sent_at
        # This mock will fail for 15s, then recover for 15s, then fail again with a *different* message
        current_cycle_time = time.time() % 30
        if current_cycle_time < 7:
             time.sleep(0.1)
             return "Simulated intermittent LLM error TYPE A", 0.1, 5, 0
        elif current_cycle_time > 15 and current_cycle_time < 22: # Fails again with different error
            time.sleep(0.1)
            return "Simulated intermittent LLM error TYPE B", 0.1, 5, 0


    time.sleep(0.3 + (len(prompt) % 3) * 0.05)
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
    status: str = "UNKNOWN"
    last_checked_at: float = 0.0
    last_error_message: Optional[str] = None # Specific error for the current episode
    details: str = "Not yet checked."
    initial_alert_sent_for_error: bool = False # For the very first occurrence of this error_message
    error_onset_time: float = 0.0 # When this specific last_error_message started
    last_reminder_alert_sent_at: float = 0.0 # Timestamp of the last 24h reminder for this error

@dataclass
class HealthCheckBaseConfig: # Common fields for all checks
    enabled: bool = True
    interval_seconds: int = 60
    current_status: CheckStatus = field(default_factory=CheckStatus)
    send_recovery_email: bool = True
    reminder_interval_hours: int = 24 # NEW: Configurable reminder interval

@dataclass
class ABCConfig(HealthCheckBaseConfig):
    model_name: str = "default_health_check_model"
    prompt: str = "Hi, health check. Respond with only 'Hi' and nothing else."
    # prompt: str = "intermittent_fail_prompt" # For testing recovery & re-alert
    expected_response: str = "Hi"
    rate_limit_cooldown_seconds: int = 300
    rate_limited_until: float = 0.0
    last_latency: float = 0.0
    last_input_tokens: int = 0
    last_output_tokens: int = 0

@dataclass
class TokenURLConfig(HealthCheckBaseConfig):
    url: str = "https://jsonplaceholder.typicode.com/todos/1" # Placeholder

# --- Health Check Logic ---

def _format_email_body(check_name: str, event_type: str, details: str, config_info: Optional[dict] = None, is_reminder: bool = False) -> str:
    prefix = "REMINDER: " if is_reminder else ""
    body = f"Health Check Event: {prefix}{event_type} for {check_name}.\n\n"
    body += f"Timestamp: {datetime.datetime.utcnow().isoformat()} Z\n"
    body += f"Details: {details}\n\n"
    if config_info:
        body += "Configuration at time of event:\n"
        safe_config = {k: v for k, v in config_info.items() if not isinstance(v, CheckStatus)}
        for k, v in safe_config.items():
            body += f"  {k}: {v}\n"
    if event_type == "FAILURE" and not is_reminder:
        body += "\nPlease investigate."
    elif event_type == "FAILURE" and is_reminder:
        body += "\nThis is a reminder that the issue is still ongoing. Please investigate if not already actioned."
    elif event_type == "RECOVERY":
        body += "\nThe service has recovered."
    return body

def process_check_result(
    config_obj: Any, # ABCConfig or TokenURLConfig
    check_name: str,
    new_status: str, 
    current_error_message_from_check: Optional[str], 
    success_details: Optional[str] = None
):
    cs = config_obj.current_status
    current_time = cs.last_checked_at # Use the check time for consistency
    previous_status = cs.status
    previous_error_message = cs.last_error_message
    reminder_threshold_seconds = config_obj.reminder_interval_hours * 60 * 60

    cs.status = new_status

    if new_status == "OK":
        cs.details = success_details if success_details else "OK"
        # If it was previously in an error state for which an initial alert might have been sent
        if cs.last_error_message is not None and config_obj.send_recovery_email:
            subject = f"RESOLVED: {check_name} Health Check Recovered"
            body = _format_email_body(check_name, "RECOVERY", cs.details, config_obj.__dict__)
            send_mail(subject, body)
        
        # Reset error state
        cs.last_error_message = None
        cs.initial_alert_sent_for_error = False
        cs.error_onset_time = 0.0
        cs.last_reminder_alert_sent_at = 0.0
        if hasattr(config_obj, 'rate_limited_until') and config_obj.rate_limited_until > 0:
            config_obj.rate_limited_until = 0
            st.toast(f"{check_name} rate limit cooldown lifted.", icon="✅")

    elif new_status in ["ERROR", "RATE_LIMITED"]:
        cs.details = f"{new_status}: {current_error_message_from_check}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
        
        is_new_error_type = (cs.last_error_message != current_error_message_from_check)
        
        if is_new_error_type or not cs.initial_alert_sent_for_error:
            # This is a brand new error or the first time we're seeing this specific error message
            cs.last_error_message = current_error_message_from_check
            cs.initial_alert_sent_for_error = True
            cs.error_onset_time = current_time
            cs.last_reminder_alert_sent_at = current_time # Initial alert also counts as the first reminder point

            alert_type = "Rate Limited" if new_status == "RATE_LIMITED" else "Failed"
            subject = f"CRITICAL: {check_name} Health Check {alert_type}"
            body = _format_email_body(check_name, "FAILURE", cs.details, config_obj.__dict__, is_reminder=False)
            send_mail(subject, body)
        elif cs.initial_alert_sent_for_error and (current_time - cs.last_reminder_alert_sent_at >= reminder_threshold_seconds):
            # Same error, initial alert was sent, and it's time for a reminder
            cs.last_reminder_alert_sent_at = current_time # Update reminder timestamp

            alert_type = "Rate Limited (Ongoing)" if new_status == "RATE_LIMITED" else "Failed (Ongoing)"
            subject = f"REMINDER: {check_name} Health Check Still {alert_type}"
            body = _format_email_body(check_name, "FAILURE", cs.details, config_obj.__dict__, is_reminder=True)
            send_mail(subject, body)
        # Else: same error, initial alert sent, not time for reminder -> do nothing

def perform_abc_check(config: ABCConfig):
    if not config.enabled:
        if config.current_status.status != "DISABLED":
            config.current_status.status = "DISABLED"
            config.current_status.details = "Monitoring is disabled."
            process_check_result(config, "ABC Response", "DISABLED", None) # Process to clear any prior error state
        return

    current_time = time.time()
    if current_time < config.rate_limited_until:
        if config.current_status.status != "RATE_LIMITED": # Update status only if it changed
             config.current_status.status = "RATE_LIMITED" 
             remaining_cooldown = config.rate_limited_until - current_time
             config.current_status.details = f"Rate limited. Next check possible in {remaining_cooldown:.0f} seconds."
        return

    if current_time - config.current_status.last_checked_at < config.interval_seconds:
        return

    config.current_status.last_checked_at = current_time # Set check time before the actual check
    current_error_msg: Optional[str] = None
    new_status_str: str = "UNKNOWN"
    success_details_str: Optional[str] = None
    
    try:
        response_text, latency, input_tokens, output_tokens = abc_response(config.model_name, config.prompt)
        config.last_latency = latency
        config.last_input_tokens = input_tokens
        config.last_output_tokens = output_tokens

        if "rate limit" in response_text.lower():
            new_status_str = "RATE_LIMITED"
            current_error_msg = response_text # The response_text itself is the error
            config.rate_limited_until = current_time + config.rate_limit_cooldown_seconds
        elif response_text != config.expected_response:
            new_status_str = "ERROR"
            current_error_msg = f"Unexpected response. Expected '{config.expected_response}', Got: '{response_text}'. Latency: {latency:.2f}s, InTokens: {input_tokens}, OutTokens: {output_tokens}"
        else:
            new_status_str = "OK"
            success_details_str = f"OK. Response: '{response_text}'. Latency: {latency:.2f}s, InTokens: {input_tokens}, OutTokens: {output_tokens}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"

    except Exception as e:
        new_status_str = "ERROR"
        current_error_msg = f"Exception during abc_response call: {str(e)}"
        config.last_latency, config.last_input_tokens, config.last_output_tokens = 0,0,0

    process_check_result(config, "ABC Response", new_status_str, current_error_msg, success_details_str)

def perform_token_url_check(config: TokenURLConfig):
    if not config.enabled:
        if config.current_status.status != "DISABLED":
            config.current_status.status = "DISABLED"
            config.current_status.details = "Monitoring is disabled."
            process_check_result(config, f"Token URL ({config.url})", "DISABLED", None)
        return

    current_time = time.time()
    if current_time - config.current_status.last_checked_at < config.interval_seconds:
        return

    config.current_status.last_checked_at = current_time
    current_error_msg: Optional[str] = None
    new_status_str: str = "UNKNOWN"
    success_details_str: Optional[str] = None
    status_code_str = "N/A"

    try:
        response = requests.get(config.url, timeout=10)
        status_code_str = str(response.status_code)
        response.raise_for_status()
        new_status_str = "OK"
        success_details_str = f"OK. Status Code: {status_code_str}. Last checked: {datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"

    except requests.exceptions.RequestException as e:
        new_status_str = "ERROR"
        current_error_msg = f"Error connecting to token URL '{config.url}': {str(e)}. Status Code: {status_code_str}"

    process_check_result(config, f"Token URL ({config.url})", new_status_str, current_error_msg, success_details_str)

# --- Streamlit UI for Health Check Tab ---
def display_health_check_section():
    st.title("🩺 System Health Checks")

    if 'abc_config' not in st.session_state:
        st.session_state.abc_config = ABCConfig()
    if 'token_url_config' not in st.session_state:
        st.session_state.token_url_config = TokenURLConfig(url="https://jsonplaceholder.typicode.com/todos/1") # TODO: Replace
    if 'health_check_refresh_interval' not in st.session_state:
         st.session_state.health_check_refresh_interval = 60

    abc_cfg = st.session_state.abc_config
    token_cfg = st.session_state.token_url_config

    if st.session_state.get('app_mode') == "🩺 Health Check":
        refresh_interval_ms = st.session_state.health_check_refresh_interval * 1000
        st_autorefresh(interval=refresh_interval_ms, limit=None, key="healthcheck_autorefresher")

    perform_abc_check(abc_cfg)
    perform_token_url_check(token_cfg)

    # --- UI for ABC Response Health ---
    st.subheader("ABC Response Health")
    with st.container(border=True):
        cols_abc_header = st.columns([2,1,2])
        abc_cfg.enabled = cols_abc_header[0].toggle("Enable ABC Monitoring", value=abc_cfg.enabled, key="abc_enabled_toggle")
        if abc_cfg.enabled:
            new_interval_abc = cols_abc_header[1].number_input(
                "Interval (s)", min_value=10, max_value=3600 * 24, # Allow up to 24h for interval
                value=abc_cfg.interval_seconds, step=10, key="abc_interval_input",
                help="How often to attempt an active health check."
            )
            if new_interval_abc != abc_cfg.interval_seconds: abc_cfg.interval_seconds = new_interval_abc
            abc_cfg.send_recovery_email = cols_abc_header[2].checkbox("Send Recovery Email", value=abc_cfg.send_recovery_email, key="abc_recovery_mail_toggle")

        with st.expander("ABC Health Check Configuration", expanded=False):
            if abc_cfg.enabled:
                st.caption("These settings apply when ABC Monitoring is enabled.")
            abc_cfg.model_name = st.text_input("Model for Health Check", value=abc_cfg.model_name, key="abc_model_input", disabled=not abc_cfg.enabled)
            abc_cfg.prompt = st.text_area("Health Check Prompt", value=abc_cfg.prompt, key="abc_prompt_input", height=100, disabled=not abc_cfg.enabled)
            abc_cfg.expected_response = st.text_input("Expected Exact Response", value=abc_cfg.expected_response, key="abc_expected_input", disabled=not abc_cfg.enabled)
            
            col_rate, col_remind = st.columns(2)
            abc_cfg.rate_limit_cooldown_seconds = col_rate.number_input(
                "Rate Limit Cooldown (s)", min_value=60, max_value=3600*6, # Max 6 hours for cooldown
                value=abc_cfg.rate_limit_cooldown_seconds, step=60, key="abc_cooldown_input",
                disabled=not abc_cfg.enabled, help="How long to wait after a rate limit error before retrying."
            )
            abc_cfg.reminder_interval_hours = col_remind.number_input(
                "Reminder Interval (hours)", min_value=1, max_value=168, # 1 hour to 1 week
                value=abc_cfg.reminder_interval_hours, step=1, key="abc_reminder_hours",
                disabled=not abc_cfg.enabled, help="How often to send a reminder if the same error persists."
            )


        status_color = "grey"; status_icon = "❓"
        if abc_cfg.current_status.status == "OK": status_color = "green"; status_icon = "✅"
        elif abc_cfg.current_status.status == "ERROR": status_color = "red"; status_icon = "❌"
        elif abc_cfg.current_status.status == "RATE_LIMITED": status_color = "orange"; status_icon = "⏳"
        elif abc_cfg.current_status.status == "DISABLED": status_color = "grey"; status_icon = "⏸️"

        st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{status_icon} {abc_cfg.current_status.status}</span>", unsafe_allow_html=True)
        st.caption(f"**Details:** {abc_cfg.current_status.details}")
        
        if abc_cfg.current_status.status == "OK" and (abc_cfg.last_latency > 0 or abc_cfg.last_input_tokens > 0 or abc_cfg.last_output_tokens > 0) :
             st.caption(f"Last call metrics: Latency: {abc_cfg.last_latency:.2f}s, InTokens: {abc_cfg.last_input_tokens}, OutTokens: {abc_cfg.last_output_tokens}")

        if abc_cfg.current_status.last_error_message and abc_cfg.current_status.status not in ["OK", "DISABLED"]:
            onset_time_str = datetime.datetime.fromtimestamp(abc_cfg.current_status.error_onset_time).strftime('%Y-%m-%d %H:%M:%S %Z') if abc_cfg.current_status.error_onset_time > 0 else 'N/A'
            st.error(f"Current Error: {abc_cfg.current_status.last_error_message} (Since: {onset_time_str})")
            if abc_cfg.current_status.initial_alert_sent_for_error:
                 last_reminder_str = datetime.datetime.fromtimestamp(abc_cfg.current_status.last_reminder_alert_sent_at).strftime('%Y-%m-%d %H:%M:%S %Z') if abc_cfg.current_status.last_reminder_alert_sent_at > 0 else 'N/A'
                 st.info(f"Initial alert sent. Last reminder for this error sent at: {last_reminder_str}. Next reminder (if issue persists) in approx. {abc_cfg.reminder_interval_hours} hours from then.")
        if abc_cfg.current_status.status == "RATE_LIMITED":
             st.warning(f"Rate limited. Next active check attempt after: {datetime.datetime.fromtimestamp(abc_cfg.rate_limited_until).strftime('%Y-%m-%d %H:%M:%S %Z') if abc_cfg.rate_limited_until > 0 else 'N/A'}")

    st.divider()

    # --- UI for Token URL Health ---
    st.subheader("Token URL Health")
    with st.container(border=True):
        cols_token_header = st.columns([2,1,2])
        token_cfg.enabled = cols_token_header[0].toggle("Enable Token URL Monitoring", value=token_cfg.enabled, key="token_enabled_toggle")

        if token_cfg.enabled:
            # Token URL interval can also be made configurable like ABC's if desired
            cols_token_header[1].info(f"Interval: {token_cfg.interval_seconds}s (fixed)")
            token_cfg.send_recovery_email = cols_token_header[2].checkbox("Send Recovery Email", value=token_cfg.send_recovery_email, key="token_recovery_mail_toggle")


        with st.expander("Token URL Configuration", expanded=False):
            if token_cfg.enabled:
                st.caption("These settings apply when Token URL Monitoring is enabled.")
            new_token_url = st.text_input("Token URL to Check", value=token_cfg.url, key="token_url_input", disabled=not token_cfg.enabled)
            if new_token_url != token_cfg.url: token_cfg.url = new_token_url
            
            token_cfg.reminder_interval_hours = st.number_input(
                "Reminder Interval (hours)", min_value=1, max_value=168, # 1 hour to 1 week
                value=token_cfg.reminder_interval_hours, step=1, key="token_reminder_hours",
                disabled=not token_cfg.enabled, help="How often to send a reminder if the same error persists."
            )


        status_color_token = "grey"; token_icon = "❓"
        if token_cfg.current_status.status == "OK": status_color_token = "green"; token_icon = "✅"
        elif token_cfg.current_status.status == "ERROR": status_color_token = "red"; token_icon = "❌"
        elif token_cfg.current_status.status == "DISABLED": status_color_token = "grey"; token_icon = "⏸️"

        st.markdown(f"**Status:** <span style='color:{status_color_token}; font-weight:bold;'>{token_icon} {token_cfg.current_status.status}</span>", unsafe_allow_html=True)
        st.caption(f"**Details:** {token_cfg.current_status.details}")
        if token_cfg.current_status.last_error_message and token_cfg.current_status.status not in ["OK", "DISABLED"]:
            onset_time_str_token = datetime.datetime.fromtimestamp(token_cfg.current_status.error_onset_time).strftime('%Y-%m-%d %H:%M:%S %Z') if token_cfg.current_status.error_onset_time > 0 else 'N/A'
            st.error(f"Current Error: {token_cfg.current_status.last_error_message} (Since: {onset_time_str_token})")
            if token_cfg.current_status.initial_alert_sent_for_error:
                last_reminder_str_token = datetime.datetime.fromtimestamp(token_cfg.current_status.last_reminder_alert_sent_at).strftime('%Y-%m-%d %H:%M:%S %Z') if token_cfg.current_status.last_reminder_alert_sent_at > 0 else 'N/A'
                st.info(f"Initial alert sent. Last reminder for this error sent at: {last_reminder_str_token}. Next reminder (if issue persists) in approx. {token_cfg.reminder_interval_hours} hours from then.")

    st.divider()
    if st.button("Force Refresh & Recheck All Enabled Monitors", key="manual_refresh_health_all"):
        if abc_cfg.enabled: abc_cfg.current_status.last_checked_at = 0
        if token_cfg.enabled: token_cfg.current_status.last_checked_at = 0
        st.rerun()

    st.caption(f"This page auto-refreshes every {st.session_state.health_check_refresh_interval} seconds if this tab is active. Individual checks run based on their own configured intervals.")
    new_refresh_slider_val = st.slider("Set Page Auto-Refresh Interval (seconds)", 10, 300, st.session_state.health_check_refresh_interval, 10, key="health_refresh_slider_main")
    if new_refresh_slider_val != st.session_state.health_check_refresh_interval:
        st.session_state.health_check_refresh_interval = new_refresh_slider_val
        st.rerun()
