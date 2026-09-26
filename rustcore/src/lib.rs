//! Expose the raw streaming engine to Python.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub mod engine;

pyo3::create_exception!(
    agentperf_local_rustcore,
    RustCoreError,
    PyException,
    "Raised when a Rust streaming request fails."
);

static RUNTIME_REGISTERED: OnceLock<()> = OnceLock::new();

fn ensure_runtime_registered() {
    RUNTIME_REGISTERED.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build Tokio runtime");
        let runtime = Box::leak(Box::new(runtime));
        pyo3_async_runtimes::tokio::init_with_runtime(runtime)
            .expect("Tokio runtime was already registered");
    });
}

#[derive(Clone, Copy)]
pub struct ClockBase {
    perf_counter_at_epoch: f64,
    instant_at_epoch: Instant,
}

impl ClockBase {
    pub fn new(perf_counter_now: f64) -> Self {
        Self {
            perf_counter_at_epoch: perf_counter_now,
            instant_at_epoch: Instant::now(),
        }
    }

    pub fn now(&self) -> f64 {
        self.perf_counter_at_epoch + self.instant_at_epoch.elapsed().as_secs_f64()
    }
}

#[pyclass]
struct RustCoreClient {
    inner: Mutex<Option<Arc<engine::CoreClient>>>,
}

#[pymethods]
impl RustCoreClient {
    #[new]
    fn new(
        base_url: &str,
        api_key: Option<&str>,
        timeout_seconds: f64,
        max_connections: usize,
        perf_counter_now: f64,
    ) -> PyResult<Self> {
        // Sample the clock epoch before any other setup so Rust read timestamps
        // stay aligned with the caller's perf_counter() sample.
        let clock = ClockBase::new(perf_counter_now);
        ensure_runtime_registered();
        let core =
            engine::CoreClient::new(base_url, api_key, timeout_seconds, max_connections, clock)
                .map_err(|error| RustCoreError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Mutex::new(Some(Arc::new(core))),
        })
    }

    #[pyo3(signature = (body, headers=None))]
    fn start(
        &self,
        body: Vec<u8>,
        headers: Option<HashMap<String, String>>,
    ) -> PyResult<RequestHandle> {
        let core = self
            .inner
            .lock()
            .expect("Rust client mutex poisoned")
            .clone()
            .ok_or_else(|| RustCoreError::new_err("Rust client is closed"))?;
        let _runtime_guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        Ok(RequestHandle {
            inner: Arc::new(core.start(body, headers)),
        })
    }

    fn close(&self) {
        self.inner
            .lock()
            .expect("Rust client mutex poisoned")
            .take();
    }
}

#[pyclass]
struct RequestHandle {
    inner: Arc<engine::CoreHandle>,
}

#[pymethods]
impl RequestHandle {
    fn wait<'python>(&self, python: Python<'python>) -> PyResult<Bound<'python, PyAny>> {
        let handle = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(python, async move {
            match handle.wait().await {
                Ok((reads, aborted)) => Python::attach(|python| {
                    let reads: Vec<(f64, Py<PyBytes>)> = reads
                        .into_iter()
                        .map(|(timestamp, data)| (timestamp, PyBytes::new(python, &data).unbind()))
                        .collect();
                    Ok((reads, aborted))
                }),
                Err(error) => Err(RustCoreError::new_err(error.to_string())),
            }
        })
    }

    fn abort(&self) {
        self.inner.abort();
    }
}

#[pymodule]
fn agentperf_local_rustcore(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustCoreClient>()?;
    module.add_class::<RequestHandle>()?;
    module.add("RustCoreError", module.py().get_type::<RustCoreError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
