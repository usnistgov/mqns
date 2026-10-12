use super::messages::*;
use anyhow::{Result, anyhow};
use async_nats::{self, HeaderMap, jetstream};
use bytes::Bytes;
use serde_json;
use std::{collections::BTreeSet, fmt, sync::Arc};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;

#[derive(Clone, Copy)]
pub enum PathAction {
    Insert,
    Delete,
}

impl fmt::Display for PathAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PathAction::Insert => "PATH_INSERT",
            PathAction::Delete => "PATH_DELETE",
        };
        write!(f, "{}", s)
    }
}

/// Southbound interface to interact with simulated quantum nodes.
#[derive(Clone)]
pub struct Southbound {
    js: jetstream::Context,
    nats_prefix: String,
    gate_subject: String,
    gate_stream: Arc<Mutex<jetstream::consumer::pull::Stream>>,
}

impl Southbound {
    async fn create_pull_stream(
        js: &jetstream::Context,
        subject: String,
    ) -> Result<jetstream::consumer::pull::Stream> {
        let stream_name = js.stream_by_subject(&subject).await?;
        let stream = js.get_stream(stream_name).await?;
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::Config {
                filter_subject: subject,
                ..Default::default()
            })
            .await?;

        let messages = consumer.messages().await?;
        Ok(messages)
    }

    fn extract_header_t(message: &jetstream::message::Message) -> u64 {
        message
            .headers
            .as_ref()
            .and_then(|h| h.get("t"))
            .and_then(|s| s.as_str().parse::<u64>().ok())
            .unwrap_or(0)
    }

    /// Construct southbound interface.
    pub async fn new(nc: async_nats::Client, nats_prefix: &str) -> Result<Self> {
        let js = jetstream::new(nc);
        let gate_stream =
            Self::create_pull_stream(&js, format!("{}.O._.gate", nats_prefix)).await?;
        Ok(Self {
            js,
            nats_prefix: nats_prefix.into(),
            gate_subject: format!("{nats_prefix}.I._.gate"),
            gate_stream: Arc::new(Mutex::new(gate_stream)),
        })
    }

    /// Send update_gate command and wait for data plane to reach the clock gate.
    ///
    /// * `t`: Clock gate in time slots.
    pub async fn update_gate(&self, t: u64) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert("t", t.to_string());
        self.js
            .publish_with_headers(self.gate_subject.clone(), headers, "".into())
            .await?
            .await?;

        let mut messages = self.gate_stream.lock().await;
        while let Some(Ok(message)) = messages.next().await {
            let now = Self::extract_header_t(&message);
            message.ack().await.map_err(|e| anyhow::anyhow!(e))?;
            if now >= t {
                return Ok(());
            }
        }
        Err(anyhow::anyhow!(
            "{}.O._.gate stream ended unexpectedly",
            self.nats_prefix
        ))
    }

    /// Schedule simulation stop.
    ///
    /// * `t`: Simulation stop time in time slots.
    pub async fn stop(&self, t: u64) -> Result<()> {
        let subject = format!("{}.I._.stop", self.nats_prefix);
        let mut headers = HeaderMap::new();
        headers.insert("t", t.to_string());
        self.js
            .publish_with_headers(subject, headers, "".into())
            .await?
            .await?;
        Ok(())
    }

    /// Send PATH_INSERT or PATH_DELETE command.
    pub async fn path_do(
        &self,
        act: PathAction,
        t: u64,
        req_id: u32,
        paths: &[PathInstructions],
    ) -> Result<()> {
        match act {
            PathAction::Insert => self.path_insert(t, req_id, paths).await,
            PathAction::Delete => self.path_delete(t, req_id, paths).await,
        }
    }

    /// Send PATH_INSERT command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `req_id`: Request identifier.
    /// * `paths`: Slice of routing path instructions belonging to this request.
    pub async fn path_insert(&self, t: u64, req_id: u32, paths: &[PathInstructions]) -> Result<()> {
        let msg = PathInsertMsg {
            cmd: CMD_PATH_INSERT,
            req_id,
            paths,
        };
        let payload = Bytes::from(serde_json::to_vec(&msg)?);
        self.send_instructions(t, payload, paths).await
    }

    /// Send PATH_DELETE command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `req_id`: Request identifier.
    /// * `paths`: Slice of routing path instructions belonging to this request.
    pub async fn path_delete(&self, t: u64, req_id: u32, paths: &[PathInstructions]) -> Result<()> {
        let msg = PathDeleteMsg {
            cmd: CMD_PATH_DELETE,
            req_id,
        };
        let payload = Bytes::from(serde_json::to_vec(&msg)?);
        self.send_instructions(t, payload, paths).await
    }

    async fn send_instructions(
        &self,
        t: u64,
        payload: Bytes,
        paths: &[PathInstructions],
    ) -> Result<()> {
        // Collect unique node names in deterministic order across all paths
        let nodes: BTreeSet<&str> = paths
            .iter()
            .flat_map(|p| p.route.iter().map(|s| s.as_str()))
            .collect();

        for dest in nodes {
            let subject = format!("{}.I.{dest}.ctrl", self.nats_prefix);
            let mut headers = HeaderMap::new();
            headers.insert("t", t.to_string());
            headers.insert("fmt", "json");
            if let Err(e) = self
                .js
                .publish_with_headers(subject, headers, payload.clone())
                .await?
                .await
            {
                return Err(anyhow!("Failed to deliver instructions to {}: {}", dest, e));
            }
        }
        Ok(())
    }

    pub async fn recv_link_states(&self, ch: mpsc::Sender<LinkStateMsg>) -> Result<()> {
        let subject = format!("{}.O.ctrl.*", self.nats_prefix);
        let mut messages = Self::create_pull_stream(&self.js, subject).await?;
        while let Some(result) = messages.next().await {
            match result {
                Ok(message) => {
                    let t = Self::extract_header_t(&message);
                    message.ack().await.map_err(|e| anyhow::anyhow!(e))?;

                    if let Ok(mut msg) = serde_json::from_slice::<LinkStateMsg>(&message.payload) {
                        if msg.cmd == CMD_LS {
                            msg.t = t;
                            if let Err(_) = ch.send(msg).await {
                                break; // channel receiver closed
                            }
                        }
                    }
                }
                Err(e) => return Err(anyhow!(e)),
            }
        }

        Ok(())
    }
}
