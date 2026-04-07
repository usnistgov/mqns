use anyhow::Result;
use async_nats::{self, jetstream};
use bpaf::Bpaf;
use mqns_example_extctrl::{
    LinkStateEntry, LinkStateMsg, PathInstructions, Southbound, sec_to_time_slot,MultiplexingVectorElem
};
use std::{collections::{HashMap,HashSet}, env, time::Duration};
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
    fn to_swap(&self) -> Vec<i32> {
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

    #[bpaf(external, fallback(SyncTiming{_sync_timing:(), t_ext:0.03, t_rtg:0.00005, t_int:0.0002}))]
    sync_timing: SyncTiming,

    /// S1-D1 path enablement and swap order (asap|l2r|r2l|disabled)
    #[bpaf(fallback(PathOpt::L2R))]
    path1: PathOpt,

    /// S1-D1 path install time in seconds
    #[bpaf(long("path1_i"), argument("SEC"), fallback(0))]
    path1_i: u64,

    /// S2-D2 path enablement and swap order (asap|l2r|r2l|disabled)
    #[bpaf(fallback(PathOpt::Disabled))]
    path2: PathOpt,

    /// S2-D2 path install time in seconds
    #[bpaf(long("path2_i"), argument("SEC"), fallback(0))]
    path2_i: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let args = args().run();

    let nc = async_nats::connect(nats_url).await?;
    let js = jetstream::new(nc);

    let sb = Southbound::new(js, &args.nats_prefix);

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

fn make_path(req_id: u32, nodes: &str, opt: PathOpt) -> PathInstructions {
    let route: Vec<String> = nodes.split('-').map(String::from).collect();
    let len = route.len();
    PathInstructions {
        req_id,
        route,
        swap: opt.to_swap(),
        swap_cutoff: vec![-1; len],
        m_v: Some(vec![MultiplexingVectorElem::Count(1, 1); len - 1]),
        purif: HashMap::new(),
    }
}

fn replace_path_mv(mut path: PathInstructions, qubits: Vec<String>) -> PathInstructions {
    path.m_v = Some(qubits.iter().map(|key| MultiplexingVectorElem::Key(key.clone())).collect());
    path
}

async fn tx_loop_pca(args: &Args, sb: &Southbound) -> Result<()> {
    for sec in 0..=args.sim_duration {
        println!("Simulation Time: {} / {}", sec, args.sim_duration);
        let t = sec * args.sim_accuracy;

        if args.path1 != PathOpt::Disabled && args.path1_i == sec {
            let path = make_path(10, "S1-R1-R2-D1", args.path1);
            println!("    Installing path1");
            sb.install_path(t, 10, &path).await?;
        }

        if args.path2 != PathOpt::Disabled && args.path2_i == sec {
            let path = make_path(20, "S2-R1-R2-D2", args.path2);
            println!("    Installing path2");
            sb.install_path(t, 20, &path).await?;
        }

        if args.sim_duration == sec {
            sb.stop(t).await?;
        }

        sb.update_gate(t).await?;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

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
        let mut path = Vec::with_capacity(route.len()-1);

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

        println!("Simulation Time: {} / {}", t, t_stop);
        sb.update_gate(t).await?;
        tokio::time::sleep(Duration::from_millis(1000)).await;

        tls.clear();
        while let Ok(msg) = link_state_ch.try_recv() {
            if msg.t < t_ext_lbound {
                continue;
            }
            for entry in &msg.ls {
                tls.add(entry);
            }
        }
        println!("    {:?}", tls);

        if args.path1 != PathOpt::Disabled && args.path1_i <= sec {
            let path = make_path(10, "S1-R1-R2-D1", args.path1);
            if let Some(consumed) = tls.try_consume(&path.route) {
                let path = replace_path_mv(path, consumed);
                println!("    Installing path1: {:?}", path.m_v);
                sb.install_path(t, 10, &path).await?;
            }
        }

        if args.path2 != PathOpt::Disabled && args.path2_i <= sec {
            let path = make_path(20, "S2-R1-R2-D2", args.path2);
            if let Some(consumed) =tls.try_consume(&path.route) {
                let path = replace_path_mv(path, consumed);
                println!("    Installing path2: {:?}", path.m_v);
                sb.install_path(t, 20, &path).await?;
            }
        }

        t += t_slot;
    }

    sb.stop(t_stop).await?;
    sb.update_gate(t_stop).await?;

    Ok(())
}
