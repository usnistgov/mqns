use anyhow::{Result, anyhow};
use async_nats::{self, HeaderMap, jetstream};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;

pub const CTRL_DELAY: f64 = 5e-6;

/// Convert seconds to time slots at given accuracy.
pub fn sec_to_time_slot(sec: f64, accuracy: u64) -> u64 {
    (sec * accuracy as f64).round() as u64
}

const CMD_INSTALL_PATH: &str = "INSTALL_PATH";
const CMD_UNINSTALL_PATH: &str = "UNINSTALL_PATH";
const CMD_LS: &str = "LS";

#[derive(Debug, Clone, Serialize)]
struct InstallPathCmd<'a> {
    cmd: String, // CMD_INSTALL_PATH
    path_id: u32,
    instructions: &'a PathInstructions,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct UninstallPathCmd {
    cmd: String, // CMD_UNINSTALL_PATH
    path_id: u32,
}

/// Instructions from the controller to forwarders regarding a routing path.
/// See mqns.network.fw.PathInstructions struct for details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathInstructions {
    pub req_id: u32,
    pub route: Vec<String>,
    pub swap: Vec<i32>,
    pub swap_cutoff: Vec<i32>,
    pub m_v: Option<Vec<MultiplexingVectorElem>>,
    pub purif: HashMap<String, String>,
}

/// Multiplexing Vector element in PathInstructions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MultiplexingVectorElem {
    Count(i32, i32),
    Key(String),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinkStateMsg {
    #[serde(skip)]
    pub t: u64,
    pub cmd: String, // CMD_LS
    pub ls: Vec<LinkStateEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinkStateEntry {
    pub node: String,
    pub neighbor: String,
    pub qubit: String,
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

    /// Send install_path command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `path_id`: Path identifier.
    /// * `instructions`: Routing path instructions. A copy of the command is sent to each node in the route.
    pub async fn install_path(
        &self,
        t: u64,
        path_id: u32,
        instructions: &PathInstructions,
    ) -> Result<()> {
        let cmd = InstallPathCmd {
            cmd: CMD_INSTALL_PATH.to_string(),
            path_id: path_id,
            instructions: instructions,
        };
        let payload = Bytes::from(serde_json::to_vec(&cmd)?);
        self.send_instructions(t, payload, instructions).await
    }

    /// Send uninstall_path command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `path_id`: Path identifier.
    /// * `instructions`: Routing path instructions. A copy of the command is sent to each node in the route.
    pub async fn uninstall_path(
        &self,
        t: u64,
        path_id: u32,
        instructions: &PathInstructions,
    ) -> Result<()> {
        let cmd = UninstallPathCmd {
            cmd: CMD_UNINSTALL_PATH.to_string(),
            path_id: path_id,
        };
        let payload = Bytes::from(serde_json::to_vec(&cmd)?);
        self.send_instructions(t, payload, instructions).await
    }

    async fn send_instructions(
        &self,
        t: u64,
        payload: Bytes,
        instructions: &PathInstructions,
    ) -> Result<()> {
        for dst in &instructions.route {
            let subject = format!("{}.I.{dst}.ctrl", self.nats_prefix);
            let mut headers = HeaderMap::new();
            headers.insert("t", t.to_string());
            headers.insert("fmt", "json");
            if let Err(e) = self
                .js
                .publish_with_headers(subject, headers, payload.clone())
                .await?
                .await
            {
                return Err(anyhow!("Failed to deliver instructions to {}: {}", dst, e));
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
