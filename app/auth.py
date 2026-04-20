from __future__ import annotations

import re
from urllib.parse import urlparse

import streamlit as st


def _get_stripe_api_key() -> str | None:
    try:
        testing_mode = bool(st.secrets.get("testing_mode", False))
    except Exception:
        testing_mode = False
    if testing_mode:
        return st.secrets.get("stripe_api_key_test") or st.secrets.get("stripe_api_key")
    return st.secrets.get("stripe_api_key")


def _infer_return_url() -> str | None:
    return_url = st.secrets.get("billing_portal_return_url") or st.secrets.get(
        "app_url"
    )
    if return_url:
        return return_url
    auth = st.secrets.get("auth", {})
    redirect_uri = auth.get("redirect_uri")
    if not redirect_uri:
        return None
    parsed = urlparse(redirect_uri)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_user_email() -> str | None:
    try:
        return st.user.email
    except Exception:
        return None


def _get_subscription_exempt_emails() -> set[str]:
    try:
        raw = st.secrets.get("subscription_exempt_emails", [])
    except Exception:
        raw = []
    if isinstance(raw, str):
        candidates = re.split(r"[,\n;]", raw)
    elif isinstance(raw, (list, tuple, set)):
        candidates = raw
    else:
        candidates = []
    return {
        str(value).strip().lower()
        for value in candidates
        if str(value).strip()
    }


def _is_subscription_exempt_user() -> bool:
    email = _get_user_email()
    if not email:
        return False
    return email.strip().lower() in _get_subscription_exempt_emails()


@st.cache_resource(ttl=86400, max_entries=5)
def _create_billing_portal_url(email: str) -> str | None:
    api_key = _get_stripe_api_key()
    if not api_key:
        return None
    try:
        import stripe
    except Exception:
        return None
    return_url = _infer_return_url()
    if not return_url:
        return None
    stripe.api_key = api_key
    try:
        customers = stripe.Customer.list(email=email, limit=1)
        if not customers.data:
            return None
        session = stripe.billing_portal.Session.create(
            customer=customers.data[0].id,
            return_url=return_url,
        )
        return session.url
    except Exception:
        return None


def _resolve_subscription_status(result: object | None = None) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        for key in ("subscribed", "is_subscribed", "subscription_active", "active"):
            if key in result:
                return bool(result.get(key))
    for key in (
        "user_subscribed",
        "is_subscribed",
        "subscription_active",
        "subscribed",
    ):
        if key in st.session_state:
            return bool(st.session_state.get(key))
    return False


def _is_user_subscribed(result: object | None = None) -> bool:
    if _is_subscription_exempt_user():
        return True
    return _resolve_subscription_status(result)
