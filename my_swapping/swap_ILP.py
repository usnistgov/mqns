import pulp

def solve_linear_chain_ILP(N, T, ranks):
    """
    Solve the ILP for a linear chain of N nodes, with an upper time bound T,
    and integer ranks for each node given by 'ranks'.

    Parameters
    ----------
    N : int
        Number of nodes (0, 1, ..., N-1).
    T : int
        Maximum discrete time-step index.
    ranks : list or dict
        A list (or dict) of length N, where ranks[i] gives r_i.

    Returns
    -------
    model : pulp.LpProblem
        The PuLP model (already solved).
    x_vars : dict
        Dictionary of x_{i->j,t} variables.  Keyed by (i, j, t).
    y_vars : dict
        Dictionary of y_{i,t} variables.  Keyed by (i, t).
    R_vars : dict
        Dictionary of R_{i,k,t} variables. Keyed by (i, k, t).
    z_vars : dict
        Dictionary of z_{j->i,k,t} variables. Keyed by (j, i, k, t).
        Represents "j has k at time t" AND "j->i message sent at t".
    """

    # -----------------------
    # 1) Define adjacency for linear chain
    # -----------------------
    neighbors = {i: [] for i in range(N)}
    for i in range(N):
        if i == 0:
            neighbors[i] = [1] if N > 1 else []
        elif i == N-1:
            neighbors[i] = [N-2] if N > 1 else []
        else:
            neighbors[i] = [i-1, i+1]

    # -----------------------
    # 2) Create a PuLP model
    # -----------------------
    model = pulp.LpProblem("LinearChainILP", pulp.LpMinimize)

    # -----------------------
    # 3) Define decision variables
    # -----------------------

    # x_{i->j,t} in {0,1} for each valid edge (i->j) and t in [0..T]
    x_vars = {}
    for i in range(N):
        for j in neighbors[i]:
            for t in range(T+1):
                x_vars[(i, j, t)] = pulp.LpVariable(
                    f"x_{i}_to_{j}_{t}", cat=pulp.LpBinary
                )

    # y_{i,t} in {0,1} for i in intermediate nodes [1..N-2] and t in [0..T]
    y_vars = {}
    for i in range(1, N-1):
        for t in range(T+1):
            y_vars[(i, t)] = pulp.LpVariable(
                f"y_{i}_{t}", cat=pulp.LpBinary
            )

    # -- Force rank-0 nodes to execute exactly at time 0 --
    for i in range(1, N-1):
        if ranks[i] == 0:
            # Must execute at t=0
            #model += (
            #    y_vars[(i, 0)] == 1,
            #    f"force_rank0_node_{i}_t0"
            #)
            # Disallow executing at any other time
            for t in range(0, T+1):
                model += (
                    y_vars[(i, t)] == 1,
                    f"no_other_time_for_rank0_node_{i}_t{t}"
                )

    # -- Forbid time-0 execution for nodes whose rank > 0 --
    for i in range(1, N-1):
        if ranks[i] > 0:
            model += (
                y_vars[(i, 0)] == 0,
                f"no_t0_exec_for_node_{i}_rank_{ranks[i]}"
            )

    # R_{i,k,t} in {0,1} for i in [0..N-1], k in [1..N-2], t in [0..T]
    R_vars = {}
    for i in range(N):
        for k in range(1, N-1):
            for t in range(T+1):
                R_vars[(i, k, t)] = pulp.LpVariable(
                    f"R_{i}_{k}_{t}", cat=pulp.LpBinary
                )

    # z_{j->i,k,t} in {0,1}: "Node j has result k at time t AND sends it to i at time t"
    z_vars = {}
    for i in range(N):
        for j in neighbors[i]:
            for k in range(1, N-1):
                for t in range(T+1):
                    z_vars[(j, i, k, t)] = pulp.LpVariable(
                        f"z_{j}_to_{i}_{k}_{t}", cat=pulp.LpBinary
                    )

    # -----------------------
    # 4) Objective: Minimize total messages
    # -----------------------
    model += pulp.lpSum(x_vars[(i, j, t)] 
                        for (i, j, t) in x_vars.keys()), "Minimize_total_messages"

    # -----------------------
    # 5) Constraints
    # -----------------------

    # --- 5.1 Exactly one execution per intermediate node ---
    for i in range(1, N-1):
        model += (
            pulp.lpSum(y_vars[(i, t)] for t in range(T+1)) == 1,
            f"one_execution_node_{i}"
        )

    # --- 5.2 No operation at end-nodes ---
    # (Implicit, since y_{0,t} or y_{N-1,t} are not defined.)

    # --- 5.3 Rank-based enablement ---
    # "Node i with rank_i > 0 can only execute if it has results from all j s.t. rank_j < rank_i."
    # y_{i,t} <= R_{i,j,t-1} for j with rank_j < rank_i
    for i in range(1, N-1):
        for t in range(1, T+1):
            for j in range(1, N-1):
                if ranks[j] < ranks[i]:
                    model += (
                        y_vars[(i, t)] <= R_vars[(i, j, t-1)],
                        f"rank_enable_i{i}_t{t}_depOn_j{j}"
                    )

    # --- 5.4 Self-result constraints ---
    # Lower bound: once i executes at tau <= t, R_{i,i,t} >= 1
    for i in range(1, N-1):
        for t in range(T+1):
            for tau in range(t+1):
                model += (
                    R_vars[(i, i, t)] >= y_vars[(i, tau)],
                    f"self_result_LB_i{i}_t{t}_tau{tau}"
                )

    # NEW Upper bound: no "magic" own-result before executing
    # R_{i,i,t} <= sum_{tau=0..t} y_{i,tau}
    for i in range(1, N-1):
        for t in range(T+1):
            model += (
                R_vars[(i, i, t)] <= pulp.lpSum(y_vars[(i, tau)] for tau in range(t+1)),
                f"self_result_UB_i{i}_t{t}"
            )

    # --- 5.5 Message-based propagation constraints ---
    # 5.5a (Lower bound):
    #    R_{i,k,t+1} >= R_{j,k,t} + x_{j->i,t} - 1
    for i in range(N):
        for j in neighbors[i]:
            for k in range(1, N-1):
                for t in range(T):
                    model += (
                        R_vars[(i, k, t+1)] >= R_vars[(j, k, t)]
                                             + x_vars[(j, i, t)] 
                                             - 1,
                        f"prop_lb_i{i}_j{j}_k{k}_t{t}"
                    )

    # 5.5b z_{j->i,k,t} = 1 iff R_{j,k,t}=1 AND x_{j->i,t}=1
    for i in range(N):
        for j in neighbors[i]:
            for k in range(1, N-1):
                for t in range(T+1):
                    model += (
                        z_vars[(j, i, k, t)] <= R_vars[(j, k, t)],
                        f"z_leq_R_j{j}_i{i}_k{k}_t{t}"
                    )
                    model += (
                        z_vars[(j, i, k, t)] <= x_vars[(j, i, t)],
                        f"z_leq_x_j{j}_i{i}_k{k}_t{t}"
                    )
                    model += (
                        z_vars[(j, i, k, t)] >= R_vars[(j, k, t)]
                                               + x_vars[(j, i, t)]
                                               - 1,
                        f"z_geq_Rplusx_minus1_j{j}_i{i}_k{k}_t{t}"
                    )

    # 5.5c (Upper bound):
    #    R_{i,k,t+1} <= R_{i,k,t} + sum_{j} z_{j->i,k,t}
    for i in range(N):
        for k in range(1, N-1):
            for t in range(T):
                model += (
                    R_vars[(i, k, t+1)] <= R_vars[(i, k, t)]
                    + pulp.lpSum(z_vars[(j, i, k, t)] for j in neighbors[i]),
                    f"prop_ub_i{i}_k{k}_t{t}"
                )

    # 5.5d "No send if node j has no results"
    # x_{j->i,t} <= sum_{k} R_{j,k,t}
    for j in range(N):
        for i in neighbors[j]:
            for t in range(T+1):
                model += (
                    x_vars[(j, i, t)] <= pulp.lpSum(R_vars[(j, k, t)] for k in range(1, N-1)),
                    f"no_send_if_no_results_j{j}_i{i}_t{t}"
                )

    # --- 5.6 Initialization: R_{i,k,0} = 0 if i != k
    # We do not forcibly set R_{k,k,0}=0. If rank(k)=0 and y_{k,0}=1, that can make R_{k,k,0}=1.
    for i in range(N):
        for k in range(1, N-1):
            if i != k:
                model += (
                    R_vars[(i, k, 0)] == 0,
                    f"init_no_result_i{i}_k{k}"
                )

    # --- 5.7 End-node final coverage ---
    # R_{0,k,T} = 1 and R_{N-1,k,T} = 1 for k in [1..N-2]
    for k in range(1, N-1):
        model += (
            R_vars[(0, k, T)] == 1,
            f"end_node0_has_k{k}"
        )
        model += (
            R_vars[(N-1, k, T)] == 1,
            f"end_nodeN1_has_k{k}"
        )

    # -----------------------
    # 6) Solve the model
    # -----------------------
    solver = pulp.PULP_CBC_CMD(msg=0)  # default solver from PuLP; set msg=1 to see more logs
    model.solve(solver)

    print("Status:", pulp.LpStatus[model.status])
    print("Objective value (total messages):", pulp.value(model.objective))

    return model, x_vars, y_vars, R_vars, z_vars


# ------------------------------
if __name__ == "__main__":
    N = 5
    T = 5
    #ranks = [2, 0, 1, 0, 2]       # doubling
    #ranks = [3, 0, 1, 2, 3]       # sequential 
    ranks = [1, 0, 0, 0, 1]       # parallel
    #ranks = [2, 1, 0, 1, 2]       # other
    #ranks = [3, 0, 2, 0, 1, 0, 3]

    model, x_vars, y_vars, R_vars, z_vars = solve_linear_chain_ILP(N, T, ranks)

    print("\nNon-zero decision variables:")
    for var in model.variables():
        val = pulp.value(var)
        if abs(val) > 1e-6:  # effectively 1 if binary
            print(f"{var.name} = {val:.0f}")
