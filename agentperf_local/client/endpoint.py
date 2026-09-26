"""Validate the measured completion endpoint."""

from urllib.parse import urlsplit, urlunsplit

SUPPORTED_ENDPOINT_SCHEMES = frozenset(("http", "https"))
LOOPBACK_HOSTNAMES = frozenset(("127.0.0.1", "::1", "localhost", "0.0.0.0"))
MIN_VISIBLE_ASCII_CODE_POINT = 33
DELETE_CODE_POINT = 127
MAX_TCP_PORT = 65_535
COMPLETION_PATH = "/chat/completions"
MEASURED_TRANSPORT_POLICY_ID = "direct-sse-no-retry-v1"
# Every client backend must present the same identity, so a server cannot serve
# one backend differently from another. The Rust engine pins the same value.
MEASURED_USER_AGENT = "agentperf-local/0.1.0"


def url_names_loopback_host(url: str) -> bool:
    """Return whether the URL names a loopback host, without resolving it."""
    # 0.0.0.0 counts as loopback: servers print it as their listen address,
    # users paste it, and connecting to it reaches the local machine.
    hostname = urlsplit(url).hostname
    return hostname is not None and hostname.lower() in LOOPBACK_HOSTNAMES


def url_is_cleartext_remote(url: str) -> bool:
    """Return whether a secret sent to this URL would travel in cleartext to another host."""
    return urlsplit(url).scheme == "http" and not url_names_loopback_host(url)


def normalize_base_url(base_url: str) -> str:
    """Return one safe base URL without a trailing slash."""
    if not base_url:
        raise ValueError("base_url must not be empty")
    if any(
        ord(character) < MIN_VISIBLE_ASCII_CODE_POINT or ord(character) == DELETE_CODE_POINT for character in base_url
    ):
        raise ValueError("base_url must not contain spaces or control characters")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("base_url must be a valid HTTP or HTTPS URL") from error
    if parsed.scheme not in SUPPORTED_ENDPOINT_SCHEMES:
        raise ValueError("base_url scheme must be http or https")
    if parsed.hostname is None:
        raise ValueError("base_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not include user information")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment")
    if port is not None and not 0 < port <= MAX_TCP_PORT:
        raise ValueError("base_url port must be between 1 and 65535")
    path = parsed.path.rstrip("/")
    if path.endswith(COMPLETION_PATH):
        raise ValueError("base_url must not include the chat completions path")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
