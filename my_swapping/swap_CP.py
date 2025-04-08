from ortools.sat.python import cp_model

def solve_full_message_propagation(ranks, T=10):
    """
    Solve the full message propagation problem in a linear chain,
    enforcing detailed constraints on execution order, result propagation,
    message minimization, and ensuring that any rank-0 node sends at
    least one message to a neighbor of higher rank.
    """

    # 1. Create the model
    model = cp_model.CpModel()

    N = len(ranks)

    # Build the set of directed edges for a linear chain
    # (i -> i+1) and (i+1 -> i), for i in [0..N-2].
    edges = []
    for i in range(N - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    # ------------------------------------------------------------------
    # 2. Decision Variables
    # ------------------------------------------------------------------

    #
    # y[i,t] = 1 if node i executes its operation at time t
    #
    y = {}
    for i in range(N):
        for t in range(T + 1):
            y[(i, t)] = model.NewBoolVar(f"y[{i},{t}]")

    #
    # x[i->j,t] = 1 if node i sends a message to node j at time t
    #
    x = {}
    for (i, j) in edges:
        for t in range(T + 1):
            x[(i, j, t)] = model.NewBoolVar(f"x[{i}->{j},{t}]")

    #
    # R[i,k,t] = 1 if node i holds the result from node k at time t
    #
    R = {}
    for i in range(N):
        for k in range(N):
            for t in range(T + 1):
                R[(i, k, t)] = model.NewBoolVar(f"R[{i},{k},{t}]")
                
    # Initialization constraints at time 0
    for i in range(N):
        for k in range(N):
            if i == k:
                # If rank[i] = 0, node i executes at t=0 => it holds its own result at t=0
                if ranks[i] == 0:
                    model.Add(R[(i, i, 0)] == 1)
                else:
                    model.Add(R[(i, i, 0)] == 0)
            else:
                # No cross results at time 0
                model.Add(R[(i, k, 0)] == 0)

    #
    # p[i,j,k,t] = 1 if "node j" sends a message to "node i" at time t
    #              AND that message includes the result of node k.
    #
    # This captures the logical product:
    #    p[i,j,k,t] = x[j->i,t] AND R[j,k,t].
    #
    p = {}
    for (j, i) in edges:
        for k in range(N):
            for t in range(T + 1):
                p[(i, j, k, t)] = model.NewBoolVar(f"p[{i},{j},{k},{t}]")

    # ------------------------------------------------------------------
    # 3. Constraints
    # ------------------------------------------------------------------

    # 3.1. Execution Constraints
    # ------------------------------------------------------------------

    # (a) Exactly one execution for each *intermediate* node (if that is the problem definition).
    #     If the boundary nodes also must execute exactly once, include them too.
    for i in range(1, N - 1):
        model.Add(sum(y[(i, t)] for t in range(T + 1)) == 1)

    # (b) Rank-0 nodes execute exactly once at time 0
    for i in range(N):
        if ranks[i] == 0:
            model.Add(y[(i, 0)] == 1)
            for t in range(1, T + 1):
                model.Add(y[(i, t)] == 0)

    # (c) Rank-based enablement:             
    # For each node i, if it needs results from j (rank[j] < rank[i]),
    # we enforce: y[i,t] <= R[i,j,t].
    for i in range(N):
        needed_nodes = [j for j in range(N) if ranks[j] < ranks[i]]
        for t in range(T + 1):
            for j in needed_nodes:
                model.Add(y[(i, t)] <= R[(i, j, t)])


    # 3.2. Result Possession & Propagation
    # ------------------------------------------------------------------

    # (a) Self-result retention:
    #     Once node i executes at time t, it keeps its own result (k=i) forever after t.
    #     => y[i,t] => R[i,i,t'] = 1 for all t' >= t
    for i in range(N):
        for t in range(T + 1):
            for t2 in range(t, T + 1):
                model.Add(R[(i, i, t2)] == 1).OnlyEnforceIf(y[(i, t)])

    # (b) No node i can hold the result of k before k has executed.
    #     => R[i,k,t] = 1 => (sum_{u=0..t} y[k,u]) >= 1
    for i in range(N):
        for k in range(N):
            if i != k:
                for t in range(T + 1):
                    model.Add(sum(y[(k, u)] for u in range(t + 1)) >= R[(i, k, t)])

    # (c) Detailed propagation logic using x and p:
    #
    #  i. p[i,j,k,t] = x[j->i,t] AND R[j,k,t]  (logical product)
    #     => p[i,j,k,t] <= x[j->i,t]
    #        p[i,j,k,t] <= R[j,k,t]
    #        p[i,j,k,t] >= x[j->i,t] + R[j,k,t] - 1
    #
    for (j, i) in edges:
        for k in range(N):
            for t in range(T + 1):
                model.Add(p[(i, j, k, t)] <= x[(j, i, t)])
                model.Add(p[(i, j, k, t)] <= R[(j, k, t)])
                model.Add(p[(i, j, k, t)] >= x[(j, i, t)] + R[(j, k, t)] - 1)

    #  ii. R[i,k,t+1] = 1 if node i already had it OR it gets it from neighbor j at time t
    #      => R[i,k,t+1] >= R[i,k,t]
    #         R[i,k,t+1] >= p[i,j,k,t] for each neighbor j
    #         R[i,k,t+1] <= R[i,k,t] + sum_j p[i,j,k,t]
    #
    #      We do this for t in [0..T-1] because R[*][t+1] is not defined for t=T.
    for i in range(N):
        neighbors_of_i = []
        if i - 1 >= 0:
            neighbors_of_i.append(i - 1)
        if i + 1 < N:
            neighbors_of_i.append(i + 1)

        for k in range(N):
            for t in range(T):
                model.Add(R[(i, k, t + 1)] >= R[(i, k, t)])
                for j in neighbors_of_i:
                    model.Add(R[(i, k, t + 1)] >= p[(i, j, k, t)])
                model.Add(
                    R[(i, k, t + 1)]
                    <= R[(i, k, t)] + sum(p[(i, j, k, t)] for j in neighbors_of_i)
                )

    # 3.3. Boundary requirements:
    #     End nodes (0 and N-1) must hold the results of all intermediate nodes [1..N-2] by time T.
    for k in range(1, N - 1):
        model.Add(R[(0, k, T)] == 1)
        model.Add(R[(N - 1, k, T)] == 1)

    # 3.4. Force rank-0 nodes to send at least one message to a neighbor of higher rank
    #     This ensures that a rank-0 node does not remain silent.
    for i in range(N):
        if ranks[i] == 0:
            # Find neighbors whose rank is strictly greater
            neighbor_candidates = []
            if i - 1 >= 0 and ranks[i - 1] > ranks[i]:
                neighbor_candidates.append(i - 1)
            if i + 1 < N and ranks[i + 1] > ranks[i]:
                neighbor_candidates.append(i + 1)

            if neighbor_candidates:
                model.Add(
                    sum(x[(i, j, t)] for j in neighbor_candidates for t in range(T + 1))
                    >= 1
                )

    # ------------------------------------------------------------------
    # 4. Objective: Minimize total messages
    # ------------------------------------------------------------------
    model.Minimize(
        sum(x[(i, j, t)] for (i, j) in edges for t in range(T + 1))
    )

    # ------------------------------------------------------------------
    # 5. Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # ------------------------------------------------------------------
    # 6. Print solution (if found)
    # ------------------------------------------------------------------
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n=====================================================")
        print(" A FEASIBLE (OR OPTIMAL) SOLUTION WAS FOUND ")
        print("=====================================================")
        print(f"Objective (total messages) = {solver.ObjectiveValue()}\n")

        # Execution schedule
        print("---- Execution Schedule ----")
        for i in range(N):
            times_executed = [
                t for t in range(T + 1) if solver.Value(y[(i, t)]) == 1
            ]
            if times_executed:
                print(f"Node {i} executes at time(s): {times_executed}")
            else:
                print(f"Node {i} does NOT execute (possibly boundary node).")

        # Messages sent
        print("\n---- Messages Sent ----")
        for (i, j) in edges:
            for t in range(T + 1):
                if solver.Value(x[(i, j, t)]) == 1:
                    print(f"Time {t}: Node {i} -> Node {j}")

        # Result possessions
        print("\n---- Result Possessions (R[i,k,t]==1) ----")
        for t in range(T + 1):
            line = [f"Time {t}:"]
            for i in range(N):
                held = []
                for k in range(N):
                    if solver.Value(R[(i, k, t)]) == 1:
                        held.append(k)
                line.append(f"Node {i}=[{','.join(map(str,held))}]")
            print(" | ".join(line))
    else:
        print("No solution found (INFEASIBLE or UNKNOWN).")


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example with 5 nodes, ranks = [3, 0, 1, 2, 3].
    # Node 1 has rank=0 => forced to execute at time 0,
    #   plus it must send at least one message to a neighbor of higher rank (node 0 or node 2).
    # Node 0 and Node 4 have rank=3 (end nodes).
    # Node 2 has rank=1, Node 3 has rank=2, etc.
    #
    # Increase or decrease T depending on how large you want the scheduling horizon.
    ranks = [3, 0, 1, 2, 3]
    T = 10
    solve_full_message_propagation(ranks, T)
