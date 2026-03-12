use anyhow::Result;
use async_nats::{self, HeaderMap, jetstream};
use bytes::Bytes;
use serde::Serialize;
use serde_json;
use std::collections::HashMap;

const CMD_INSTALL_PATH: &str = "INSTALL_PATH";
const CMD_UNINSTALL_PATH: &str = "UNINSTALL_PATH";

#[derive(Serialize)]
struct InstallPathCmd<'a> {
    cmd: String,
    path_id: u32,
    instructions: &'a PathInstructions,
}

#[derive(Serialize)]
struct UninstallPathCmd {
    cmd: String,
    path_id: u32,
}

/// Instructions from the controller to forwarders regarding a routing path.
/// See mqns.network.fw.PathInstructions struct for details.
#[derive(Serialize)]
pub struct PathInstructions {
    pub req_id: u32,
    pub route: Vec<String>,
    pub swap: Vec<i32>,
    pub swap_cutoff: Vec<i32>,
    pub m_v: Option<Vec<Vec<i32>>>,
    pub purif: HashMap<String, String>,
}

/// Southbound interface to interact with simulated quantum nodes.
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
            gate_subject: format!("{nats_prefix}._.gate"),
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
        let subject = format!("{}._.stop", self.nats_prefix);
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
            let subject = format!("{}.{dst}.ctrl", self.nats_prefix);
            let mut headers = HeaderMap::new();
            headers.insert("t", t.to_string());
            headers.insert("fmt", "json");
            self.js
                .publish_with_headers(subject, headers, payload.clone())
                .await?
                .await?;
        }
        Ok(())
    }
}
