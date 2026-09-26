use std::time::Duration;

use agentperf_local_rustcore::engine::CoreClient;
use agentperf_local_rustcore::ClockBase;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const CLOCK_BASE: f64 = 1_000_000.0;

async fn serve(parts: Vec<(Duration, &'static [u8])>, status: u16) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("read test server address");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = vec![0_u8; 4096];
        let _ = socket.read(&mut request).await;
        let status_text = if status == 200 { "OK" } else { "Bad Request" };
        let headers = format!(
            "HTTP/1.1 {status} {status_text}\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
        );
        socket
            .write_all(headers.as_bytes())
            .await
            .expect("write headers");
        for (delay, part) in parts {
            tokio::time::sleep(delay).await;
            let frame = format!("{:x}\r\n", part.len());
            if socket.write_all(frame.as_bytes()).await.is_err() {
                return;
            }
            if socket.write_all(part).await.is_err() || socket.write_all(b"\r\n").await.is_err() {
                return;
            }
        }
        let _ = socket.write_all(b"0\r\n\r\n").await;
    });
    format!("http://{address}/v1")
}

#[tokio::test]
async fn streams_raw_bytes_with_monotonic_timestamps() {
    let first = b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n";
    let done = b"data: [DONE]\n\n";
    let base_url = serve(
        vec![
            (Duration::from_millis(10), first),
            (Duration::from_millis(10), done),
        ],
        200,
    )
    .await;
    let client = CoreClient::new(&base_url, None, 1.0, 2, ClockBase::new(CLOCK_BASE))
        .expect("create client");
    let (reads, aborted) = client
        .start(b"{}".to_vec(), None)
        .wait()
        .await
        .expect("stream response");

    assert!(!aborted);
    assert_eq!(
        reads.iter().map(|(_, data)| data.len()).sum::<usize>(),
        first.len() + done.len()
    );
    assert!(reads.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    assert!(reads.iter().all(|(timestamp, _)| *timestamp >= CLOCK_BASE));
}

#[tokio::test]
async fn abort_returns_partial_reads() {
    let event = b"data: {\"choices\":[]}\n\n";
    let base_url = serve(
        vec![
            (Duration::from_millis(5), event),
            (Duration::from_secs(1), event),
        ],
        200,
    )
    .await;
    let client = CoreClient::new(&base_url, None, 2.0, 2, ClockBase::new(CLOCK_BASE))
        .expect("create client");
    let handle = client.start(b"{}".to_vec(), None);
    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.abort();
    let (reads, aborted) = handle.wait().await.expect("finish aborted request");

    assert!(aborted);
    assert!(!reads.is_empty());
}

#[tokio::test]
async fn non_success_status_is_an_error() {
    let base_url = serve(vec![(Duration::ZERO, b"invalid request")], 400).await;
    let client = CoreClient::new(&base_url, None, 1.0, 2, ClockBase::new(CLOCK_BASE))
        .expect("create client");
    let error = client
        .start(b"{}".to_vec(), None)
        .wait()
        .await
        .expect_err("reject non-success response");
    assert!(error.to_string().contains("400"));
}
