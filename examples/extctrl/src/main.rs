use anyhow::{Result, anyhow};
use async_nats;
use bpaf::Bpaf;
use mqns_example_extctrl::{
    LinkStateEntry, LinkStateMsg, PathAction, PathInstructions, Southbound, sec_to_time_slot,
};
use serde_json;
use std::{
    collections::{HashMap, HashSet},
    env, slice,
};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Bpaf)]
enum Mode {
    /// Proactive forwarding, Centralized control, Async timing
    PCA,
    /// Reactive forwarding, Centralized control, Sync timing
    RCS,
}

impl std::str::FromStr for Mode {
    type Err = String;

    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "PCA" => Ok(Mode::PCA),
            "RCS" => Ok(Mode::RCS),
            _ => Err(format!("invalid mode: {s}")),
        }
    }
}

#[derive(Debug, Clone, Bpaf)]
#[bpaf(adjacent)]
struct SyncTiming {
    /// SYNC timing mode phase durations in seconds
    #[bpaf(long("sync_timing"))]
    _sync_timing: (),

    /// EXTERNAL phase duration in seconds
    #[bpaf(positional("t_ext"))]
    t_ext: f64,

    /// ROUTING phase duration in seconds
    #[bpaf(positional("t_rtg"))]
    t_rtg: f64,

    /// INTERNAL phase duration in seconds
    #[bpaf(positional("t_int"))]
    t_int: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Bpaf)]
enum PathOpt {
    ASAP,
    L2R,
    R2L,
    Disabled,
}

impl PathOpt {
    fn to_swap(&self) -> Vec<u32> {
        match self {
            PathOpt::ASAP => vec![1, 0, 0, 1],
            PathOpt::L2R => vec![2, 0, 1, 2],
            PathOpt::R2L => vec![2, 1, 0, 2],
            _ => vec![0, 0, 0, 0],
        }
    }
}

impl std::str::FromStr for PathOpt {
    type Err = String;

    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "asap" => Ok(PathOpt::ASAP),
            "l2r" => Ok(PathOpt::L2R),
            "r2l" => Ok(PathOpt::R2L),
            "disabled" => Ok(PathOpt::Disabled),
            _ => Err(format!("invalid path option: {s}")),
        }
    }
}

/// MQNS extctrl example Control Plane application
#[derive(Debug, Clone, Bpaf)]
#[bpaf(options, fallback_to_usage)]
struct Args {
    /// Prefix of NATS subjects
    #[bpaf(
        long("nats_prefix"),
        argument("PREFIX"),
        fallback(String::from("mqns.classicbridge"))
    )]
    nats_prefix: String,

    /// Simulation accuracy in time slots per second
    #[bpaf(long("sim_accuracy"), argument("ACCURACY"), fallback(1_000_000))]
    sim_accuracy: u64,

    /// Simulation duration in seconds
    #[bpaf(long("sim_duration"), argument("DURATION"), fallback(60))]
    sim_duration: u64,

    /// Application and timing mode
    #[bpaf(argument("MODE"), fallback(Mode::PCA))]
    mode: Mode,

    #[bpaf(external, fallback(SyncTiming{_sync_timing:(), t_ext:0.024990, t_rtg:0.000020, t_int:0.024990}))]
    sync_timing: SyncTiming,

    /// S1-D1 path enablement and swap order (asap|l2r|r2l|disabled)
    #[bpaf(fallback(PathOpt::L2R))]
    path1: PathOpt,

    /// S1-D1 path insertion time in seconds
    #[bpaf(long("path1_i"), argument("SEC"), fallback(0))]
    path1_i: u64,

    /// S1-D1 path deletion time in seconds (no deletion if omitted)
    #[bpaf(long("path1_d"), argument("SEC"))]
    path1_d: Option<u64>,

    /// S2-D2 path enablement and swap order (asap|l2r|r2l|disabled)
    #[bpaf(fallback(PathOpt::Disabled))]
    path2: PathOpt,

    /// S2-D2 path insertion time in seconds
    #[bpaf(long("path2_i"), argument("SEC"), fallback(0))]
    path2_i: u64,

    /// S2-D2 path deletion time in seconds (no deletion if omitted)
    #[bpaf(long("path2_d"), argument("SEC"))]
    path2_d: Option<u64>,
}

struct PathArgs {
    name: &'static str,
    req_id: u32,
    opt: PathOpt,
    t_insert: u64,
    t_delete: Option<u64>,
    route: &'static str,
}

impl Args {
    fn paths(&self) -> [PathArgs; 2] {
        [
            PathArgs {
                name: "path1",
                req_id: 10,
                opt: self.path1,
                t_insert: self.path1_i,
                t_delete: self.path1_d,
                route: "S1-R1-R2-D1",
            },
            PathArgs {
                name: "path2",
                req_id: 20,
                opt: self.path2,
                t_insert: self.path2_i,
                t_delete: self.path2_d,
                route: "S2-R1-R2-D2",
            },
        ]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let args = args().run();

    let nc = async_nats::connect(nats_url).await?;
    let sb = Southbound::new(nc, &args.nats_prefix).await?;

    match args.mode {
        Mode::PCA => tx_loop_pca(&args, &sb).await,
        Mode::RCS => {
            let (tx, rx) = mpsc::channel(100);
            let rx_handle = {
                let sb = sb.clone();
                tokio::spawn(async move { rx_loop_rcs(&sb, tx).await })
            };
            let tx_result = tx_loop_rcs(&args, &sb, rx).await;
            rx_handle.abort();
            tx_result
        }
    }?;

    Ok(())
}

async fn tx_loop_pca(args: &Args, sb: &Southbound) -> Result<()> {
    let paths = args.paths();
    let mut events: Vec<(u64, PathAction, &PathArgs)> = Vec::new();
    for path in paths.iter() {
        if path.opt == PathOpt::Disabled {
            continue;
        }
        events.push((path.t_insert, PathAction::Insert, path));
        if let Some(t_delete) = path.t_delete {
            if t_delete <= path.t_insert {
                return Err(anyhow!("{} deletion must occur after insertion", path.name));
            }
            events.push((t_delete, PathAction::Delete, path));
        }
    }
    events.sort_by_key(|&(sec, act, path)| (sec, act as u8, path.req_id));

    for (sec, act, path) in events {
        if sec >= args.sim_duration {
            println!(
                "[{sec}] <tx_loop_pca> skipping {act} {}: past sim_duration",
                path.name
            );
            continue;
        }

        let t = sec * args.sim_accuracy;
        println!(
            "[{sec}] <tx_loop_pca> sending {act} {}: route {}",
            path.name, path.route
        );

        let inst = PathInstructions::new(path.req_id, PathInstructions::split_route(path.route))
            .bufferspace_mv(vec![1; 6])
            .swap(path.opt.to_swap())
            .build();

        sb.path_do(act, t, path.req_id, slice::from_ref(&inst))
            .await?;

        println!("[{sec}] <tx_loop_pca> DP is allowed to run until time slot {t}");
        sb.update_gate(t).await?;
        println!("[{sec}] <tx_loop_pca> DP has reached time slot {t}");
    }

    let t = args.sim_duration * args.sim_accuracy;
    println!(
        "[{}] <tx_loop_pca> DP is instructed to stop at time slot {t}",
        args.sim_duration
    );
    sb.stop(t).await?;
    Ok(())
}

async fn rx_loop_rcs(sb: &Southbound, link_state_ch: mpsc::Sender<LinkStateMsg>) -> Result<()> {
    sb.recv_link_states(link_state_ch).await?;
    Ok(())
}

#[derive(Debug, Clone)]
struct TopoLinkState {
    etgs: HashMap<String, HashSet<String>>,
}

impl TopoLinkState {
    fn new() -> Self {
        Self {
            etgs: ["S1-R1", "S2-R1", "R1-R2", "R2-D1", "R2-D2"]
                .into_iter()
                .map(|link| (link.to_string(), HashSet::new()))
                .collect(),
        }
    }

    fn clear(&mut self) {
        for qubits in self.etgs.values_mut() {
            qubits.clear();
        }
    }

    fn add(&mut self, entry: &LinkStateEntry) {
        let link = format!("{}-{}", entry.node, entry.neighbor);
        if let Some(qubits) = self.etgs.get_mut(&link) {
            qubits.insert(entry.qubit.clone());
        }
    }

    fn try_consume(&mut self, route: &Vec<String>) -> Option<Vec<String>> {
        let mut path = Vec::with_capacity(route.len() - 1);

        for pair in route.windows(2) {
            let link = format!("{}-{}", pair[0], pair[1]);
            if let Some(qubits) = self.etgs.get(&link) {
                if let Some(qubit) = qubits.iter().next() {
                    path.push(qubit.clone());
                    continue;
                }
            }
            return None;
        }

        for (i, pair) in route.windows(2).enumerate() {
            let link = format!("{}-{}", pair[0], pair[1]);
            let qubits = self.etgs.get_mut(&link).unwrap();
            qubits.remove(&path[i]);
        }

        Some(path)
    }
}

async fn tx_loop_rcs(
    args: &Args,
    sb: &Southbound,
    mut link_state_ch: mpsc::Receiver<LinkStateMsg>,
) -> Result<()> {
    let paths = args.paths();
    let t_ext = sec_to_time_slot(args.sync_timing.t_ext, args.sim_accuracy);
    let t_rtg = sec_to_time_slot(args.sync_timing.t_rtg, args.sim_accuracy);
    let t_int = sec_to_time_slot(args.sync_timing.t_int, args.sim_accuracy);
    let t_slot = t_ext + t_rtg + t_int;
    let t_stop = sec_to_time_slot(args.sim_duration as f64, args.sim_accuracy) / t_slot * t_slot;
    let t_offset = t_ext + t_rtg / 2;

    let mut tls = TopoLinkState::new();
    let mut t = t_offset;
    while t < t_stop {
        let sec = t / args.sim_accuracy;
        let t_ext_lbound = t - t_offset;
        let sec_f = t as f64 / args.sim_accuracy as f64;

        println!("[{sec_f}] <tx_loop_rcs> DP is running until time slot {t}");
        sb.update_gate(t).await?;

        tls.clear();
        while let Ok(msg) = link_state_ch.try_recv() {
            if msg.t < t_ext_lbound {
                continue;
            }
            for entry in &msg.ls {
                tls.add(entry);
            }
        }
        println!("[{sec_f}] <tx_loop_rcs> {:?}", tls);

        for path in paths.iter() {
            if path.opt != PathOpt::Disabled
                && path.t_insert <= sec
                && sec < path.t_delete.unwrap_or(args.sim_duration)
            {
                let route = PathInstructions::split_route(path.route);
                if let Some(consumed) = tls.try_consume(&route) {
                    let inst = PathInstructions::new(path.req_id, route)
                        .reactive_qubits(consumed)
                        .swap(path.opt.to_swap())
                        .build();

                    println!(
                        "[{sec_f}] <tx_loop_rcs> sending PATH_INSERT {}: qubits {}",
                        path.name,
                        serde_json::to_string(&inst.reactive_qubits).unwrap()
                    );
                    sb.path_insert(t, inst.path_id, slice::from_ref(&inst))
                        .await?;
                }
            }
        }

        t += t_slot;
    }

    println!(
        "[{}] <tx_loop_rcs> DP is instructed to stop at time slot {t_stop}",
        args.sim_duration
    );
    sb.stop(t_stop).await?;

    Ok(())
}
