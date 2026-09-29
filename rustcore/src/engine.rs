//! Run HTTP requests and collect timestamped response bytes.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::header::{
    HeaderMap, HeaderName, HeaderValue, ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_TYPE,
    USER_AGENT,
};
use reqwest::{redirect, retry, Client, Url};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::ClockBase;

const ERROR_BODY_LIMIT_BYTES: usize = 512;
/// Identity shared with the Python client so a server cannot tell them apart.
const MEASURED_USER_AGENT: &str = "agentperf-local/0.3.1";

#[derive(Debug, Clone)]
pub struct CoreError(String);

impl CoreError {
    fn new(message: String) -> Self {
        Self(message)
    }
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CoreError {}

pub type ReadResult = Result<(Vec<(f64, Bytes)>, bool), CoreError>;

pub struct CoreClient {
    http: Client,
    endpoint: Url,
    base_headers: HeaderMap,
    clock: ClockBase,
}

impl CoreClient {
    pub fn new(
        base_url: &str,
        api_key: Option<&str>,
        timeout_seconds: f64,
        max_connections: usize,
        clock: ClockBase,
    ) -> Result<Self, CoreError> {
        if timeout_seconds <= 0.0 {
            return Err(CoreError::new(
                "timeout_seconds must be greater than zero".to_string(),
            ));
        }
        if max_connections == 0 {
            return Err(CoreError::new(
                "max_connections must be greater than zero".to_string(),
            ));
        }
        let timeout = Duration::from_secs_f64(timeout_seconds);
        let http = Client::builder()
            .http1_only()
            .pool_idle_timeout(None)
            .pool_max_idle_per_host(max_connections)
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .redirect(redirect::Policy::none())
            .retry(retry::never())
            .no_proxy()
            .build()
            .map_err(|error| CoreError::new(format!("failed to build HTTP client: {error}")))?;
        let endpoint_text = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let endpoint = Url::parse(&endpoint_text)
            .map_err(|error| CoreError::new(format!("invalid base URL {base_url:?}: {error}")))?;
        Ok(Self {
            http,
            endpoint,
            base_headers: base_headers(api_key)?,
            clock,
        })
    }

    pub fn start(
        &self,
        body: Vec<u8>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> CoreHandle {
        let cancel = CancellationToken::new();
        let join = tokio::spawn(run_request(
            self.http.clone(),
            self.endpoint.clone(),
            self.base_headers.clone(),
            extra_headers,
            body,
            self.clock,
            cancel.clone(),
        ));
        CoreHandle::new(cancel, join)
    }
}

fn base_headers(api_key: Option<&str>) -> Result<HeaderMap, CoreError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    headers.insert(USER_AGENT, HeaderValue::from_static(MEASURED_USER_AGENT));
    if let Some(key) = api_key {
        let value = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|error| CoreError::new(format!("invalid authorization header: {error}")))?;
        headers.insert(AUTHORIZATION, value);
    }
    Ok(headers)
}

fn request_headers(
    base: HeaderMap,
    extra_headers: Option<HashMap<String, String>>,
) -> Result<HeaderMap, CoreError> {
    let mut headers = base;
    // Insert, never append: a caller-supplied header replaces the default of the
    // same name, exactly as the Python client behaves.
    for (name, value) in extra_headers.unwrap_or_default() {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| CoreError::new(format!("invalid header name {name:?}: {error}")))?;
        let header_value = HeaderValue::from_str(&value)
            .map_err(|error| CoreError::new(format!("invalid header value for {name:?}: {error}")))?;
        headers.insert(header_name, header_value);
    }
    Ok(headers)
}

async fn run_request(
    http: Client,
    endpoint: Url,
    base: HeaderMap,
    extra_headers: Option<HashMap<String, String>>,
    body: Vec<u8>,
    clock: ClockBase,
    cancel: CancellationToken,
) -> ReadResult {
    let headers = request_headers(base, extra_headers)?;
    let response = http
        .post(endpoint)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|error| CoreError::new(format!("request failed: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        let bytes = response.bytes().await.unwrap_or_default();
        let prefix = &bytes[..bytes.len().min(ERROR_BODY_LIMIT_BYTES)];
        return Err(CoreError::new(format!(
            "status {}: {}",
            status.as_u16(),
            String::from_utf8_lossy(prefix)
        )));
    }

    let mut stream = response.bytes_stream();
    let mut reads = Vec::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok((reads, true)),
            next = stream.next() => match next {
                Some(Ok(bytes)) => reads.push((clock.now(), bytes)),
                Some(Err(error)) => return Err(CoreError::new(format!("stream failed: {error}"))),
                None => return Ok((reads, false)),
            },
        }
    }
}

pub struct CoreHandle {
    cancel: CancellationToken,
    join: Mutex<Option<JoinHandle<ReadResult>>>,
}

impl CoreHandle {
    fn new(cancel: CancellationToken, join: JoinHandle<ReadResult>) -> Self {
        Self {
            cancel,
            join: Mutex::new(Some(join)),
        }
    }

    pub fn abort(&self) {
        self.cancel.cancel();
    }

    pub async fn wait(&self) -> ReadResult {
        let join = self
            .join
            .lock()
            .expect("request handle mutex poisoned")
            .take()
            .ok_or_else(|| CoreError::new("wait was already called".to_string()))?;
        join.await
            .map_err(|error| CoreError::new(format!("request task failed: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_clock() -> ClockBase {
        ClockBase::new(0.0)
    }

    #[test]
    fn endpoint_joins_one_path_separator() {
        let client = CoreClient::new("http://127.0.0.1:9/v1/", None, 1.0, 1, test_clock()).unwrap();
        assert_eq!(
            client.endpoint.as_str(),
            "http://127.0.0.1:9/v1/chat/completions"
        );
    }

    #[test]
    fn configuration_rejects_non_positive_limits() {
        assert!(CoreClient::new("http://127.0.0.1:9/v1", None, 0.0, 1, test_clock()).is_err());
        assert!(CoreClient::new("http://127.0.0.1:9/v1", None, 1.0, 0, test_clock()).is_err());
    }

    #[test]
    fn extra_headers_replace_defaults_instead_of_appending() {
        let extra = HashMap::from([
            ("accept".to_string(), "application/json".to_string()),
            ("x-replay".to_string(), "yes".to_string()),
        ]);
        let headers = request_headers(base_headers(None).unwrap(), Some(extra)).expect("build request headers");

        assert_eq!(headers.get_all(ACCEPT).iter().count(), 1);
        assert_eq!(headers[ACCEPT], "application/json");
        assert_eq!(headers[USER_AGENT], MEASURED_USER_AGENT);
        assert_eq!(headers["x-replay"], "yes");
    }

    #[test]
    fn invalid_extra_header_names_are_rejected() {
        let extra = HashMap::from([("bad header".to_string(), "value".to_string())]);
        assert!(request_headers(base_headers(None).unwrap(), Some(extra)).is_err());
    }
}
