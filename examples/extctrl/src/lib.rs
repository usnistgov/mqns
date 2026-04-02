use anyhow::{Result, anyhow};
use async_nats::{self, HeaderMap, jetstream};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use tokio::sync::mpsc;
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
    pub m_v: Option<Vec<Vec<i32>>>,
    pub purif: HashMap<String, String>,
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
    pub qubit: i32,
}

/// Southbound interface to interact with simulated quantum nodes.
#[derive(Debug, Clone)]
pub struct Southbound {
    js: jetstream::Context,
    nats_prefix: String,
    gate_subject: String,
}

impl Southbound {
    /// Construct southbound interface.
    pub fn new(js: jetstream::Context, nats_prefix: &str) -> Self {
        Self {
            js,
            nats_prefix: nats_prefix.into(),
            gate_subject: format!("{nats_prefix}.I._.gate"),
        }
    }

    /// Send update_gate command.
    ///
    /// * `t`: Clock gate in time slots.
    pub async fn update_gate(&self, t: u64) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert("t", t.to_string());
        self.js
            .publish_with_headers(self.gate_subject.clone(), headers, "".into())
            .await?
            .await?;
        Ok(())
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
        let stream_name = self.js.stream_by_subject(&subject).await?;
        let stream = self.js.get_stream(stream_name).await?;
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::Config {
                filter_subject: subject,
                ..Default::default()
            })
            .await?;

        let mut messages = consumer.messages().await?;
        while let Some(result) = messages.next().await {
            match result {
                Ok(message) => {
                    let t = message
                        .headers
                        .as_ref()
                        .and_then(|h| h.get("t"))
                        .and_then(|s| s.as_str().parse::<u64>().ok())
                        .unwrap_or(0);
                    message.ack().await.ok();

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
