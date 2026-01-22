import os
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import binom


def bal_tree(node_list):
    # create a balance tree from a list
    tree0 = node_list
    tree = []
    while tree0:
        level = tree0[::2]  # odd indices
        tree.append(level)
        tree0 = tree0[1::2]  # even indices
    return tree


def order2tree(order, path=None):
    # convert an order to a tree
    n = len(order)
    if path is None:
        path = list(range(1, n + 1))

    tree = []
    nl = [order[0]]
    idn = [path.index(order[0])]

    for i in range(1, n):
        s = order[i]
        ids = path.index(s)

        if (ids + 1 in idn) or (ids - 1 in idn):  # if adjacent
            tree.append(nl)
            path = list(set(path) - set(nl))
            nl = [s]
            idn = [path.index(s)]
        else:
            nl.append(s)
            idn.append(ids)

    tree.append(nl)
    return tree


def tree2order(tree):
    # convert a tree to an order - list of indices
    order = []
    for level in tree:
        order += level
    return order


def merge_close(numbers: list[float], relative_threshold_percent=0.5) -> list[float]:
    # Group relatively close elements, replace them with average
    if not numbers:
        return []

    threshold_fraction = relative_threshold_percent / 100.0
    n = len(numbers)

    # Sort numbers
    indexed_numbers = sorted([(num, i) for i, num in enumerate(numbers)])
    sorted_values = [item[0] for item in indexed_numbers]
    original_indices = [item[1] for item in indexed_numbers]

    # Find clusters
    clusters = []
    current_cluster_indices = [original_indices[0]]
    current_cluster_values = [sorted_values[0]]

    for i in range(1, n):
        current_val = sorted_values[i]
        prev_val = sorted_values[i - 1]

        denominator = (abs(current_val) + abs(prev_val)) / 2.0

        if abs(denominator) < 1e-9:
            is_close = abs(current_val - prev_val) < 1e-9
        else:
            is_close = abs(current_val - prev_val) / abs(denominator) <= threshold_fraction

        if is_close:
            current_cluster_values.append(current_val)
            current_cluster_indices.append(original_indices[i])
        else:
            # End of cluster. Store (Average, Original Indices)
            avg = sum(current_cluster_values) / len(current_cluster_values)
            clusters.append((avg, current_cluster_indices))

            # Start a new cluster
            current_cluster_values = [current_val]
            current_cluster_indices = [original_indices[i]]

    # Add the last cluster
    avg = sum(current_cluster_values) / len(current_cluster_values)
    clusters.append((avg, current_cluster_indices))

    # Back to the original order
    result = [0.0] * n
    for cluster_avg, indices in clusters:
        for original_index in indices:
            result[original_index] = cluster_avg

    return result


def get_Bq(N: int, q=0.5):
    # get Binomial coeffs
    fname = os.path.join(os.path.dirname(__file__), ".vora_cache", f"B_q{q}_N{N}.pkl")
    if os.path.exists(fname):
        Bq = pickle.load(open(fname, "rb"))
        return Bq
    else:
        q1 = 1 - q
        Bq = np.zeros((N + 1, N + 1))
        for k in range(N + 1):
            Bq[k, k] = 1 if k == 0 else Bq[k - 1, k - 1] * q
            for n in range(k, N):
                Bq[n + 1, k] = Bq[n, k] * q1 * (n + 1) / (n + 1 - k)

        pickle.dump(Bq, open(fname, "wb"))

        return Bq


def approx_pd(p, cutoff=0.999):
    # truncate distribution tail
    if cutoff == 1:
        return p
    else:
        S = 1 - np.sum(p)  # start at 0
        for i in range(len(p)):
            S += p[i]
            if S >= cutoff:
                p[i] += 1 - S
                break
        return p[: i + 1]


def binomial(p, n, k):
    # Binomial coeff
    C = 1
    kmin = min(k, n - k)
    for i in range(kmin):
        C *= (n - i) / (kmin - i)
    return C * (p**k) * (1 - p) ** (n - k)


def bin_pd(n, p, cutoff=1):
    # Binomial distribution
    upper_bound = binom.ppf(cutoff, n, p)
    lower_k = 0  # max(0, int(lower_bound) - 1)
    upper_k = min(n, int(upper_bound) + 1)
    k_values = np.arange(lower_k, upper_k + 1)
    pmf_values = binom.pmf(k_values, n, p)
    pmf_values /= pmf_values.sum()
    return pmf_values


def swap_pds(p1, p2, q, cutoff=1, Bq=None):
    # swapping to prob. distributinos
    Lc = min(len(p1), len(p2))
    p = np.zeros(Lc)
    for k in np.arange(Lc):
        for i in np.arange(k, len(p1)):
            p2b = 0
            for j in np.arange(k, len(p2)):
                if j <= i:
                    b = binomial(q, j + 1, k + 1) if Bq is None else Bq[j + 1, k + 1]
                p2b += p2[j] * b
            p[k] += p1[i] * p2b

    p = approx_pd(p, cutoff)
    EXT = np.dot(1 + np.arange(len(p)), p)
    return EXT, p


def remove_item(node_list, s):
    if s == 0:
        return node_list[1:]
    elif s == len(node_list) - 1:
        return node_list[:-1]
    else:
        return node_list[:s] + node_list[s + 1 :]


def greedyswap(C, p=1, q=0.5, prnt=False, cutoff=1, Bq=None, Ts=1):
    start_time = time.time()
    path_len = len(C)
    n = path_len + 1  # number of nodes

    p = [p] * path_len if type(p) in [float, int] else p
    q = [q] * (path_len - 1) if type(q) in [float, int] else q

    P = {}  # dict - link capacity
    Lc = np.zeros((n, n), dtype=int)
    for i in range(path_len):
        w = C[i]
        Lc[i, i + 1] = w
        P[(i, i + 1)] = bin_pd(round(C[i]), p[i], cutoff)

    node_list = list(range(1, path_len))  # 1,2,...,C
    S = {}  # dict - Node swapping (score, distribution)
    for i in node_list:
        S[i] = swap_pds(P[(i - 1, i)], P[(i, i + 1)], q[i - 1], cutoff, Bq)

    swap_sq = []

    for _ in range(path_len - 1):  # greedy swapping loop
        # get score
        scores = [S[i][0] for i in node_list]
        max_score = max(scores)
        ids = [i for i, score_i in enumerate(scores) if score_i == max_score]
        s = ids[0]
        sn = node_list[s]  # swap_node
        swap_sq.append(sn)
        node_list = remove_item(node_list, s)

        if prnt:
            print(scores, "\tswap:", sn)

        x = np.where(Lc[:, sn])[0][0]
        z = np.where(Lc[sn, :])[0][0]
        # update link capacity
        Lc[x, z] = min(Lc[x, sn], Lc[sn, z])
        P[(x, z)] = S[sn][1]
        Lc[x, sn], Lc[sn, z] = 0, 0  # consumed

        # update score of adjacent nodes: X and Z if they are repeaters, not SD
        if x > 0:
            x_pre = np.where(Lc[:, x])[0][0]
            S[x] = swap_pds(P[(x_pre, x)], P[(x, z)], q[x - 1], cutoff, Bq)

        if z < path_len:
            z_post = np.where(Lc[z, :])[0][0]
            S[z] = swap_pds(P[(x, z)], P[(z, z_post)], q[z - 1], cutoff, Bq)

    p_e2e = P[(0, path_len)]
    EXT = np.dot(1 + np.arange(len(p_e2e)), p_e2e) / Ts
    runtime = time.time() - start_time
    if prnt:
        print("Swapping order", swap_sq, "  => E[ent.]: {:.6f}".format(EXT), "\ttime: {:.6f}".format(runtime))

    return {"order": swap_sq, "EXT": EXT, "P": P, "time": runtime}


def thruput(C, swap_sq, p=1, q=0.5, prnt=False, cutoff=1, Bq=None, Ts=1):
    if swap_sq == "asap":
        return None
    start_time = time.time()
    path_len = len(C)
    n = path_len + 1  # number of nodes

    p = [p] * path_len if type(p) in [float, int] else p
    q = [q] * (path_len - 1) if type(q) in [float, int] else q

    P = {}  # dict - link capacity
    Lc = np.zeros((n, n), dtype=int)
    for i in range(path_len):
        Lc[i, i + 1] = C[i]
        P[(i, i + 1)] = bin_pd(round(C[i]), p[i], cutoff)

    node_list = list(range(1, path_len))  # 1,2,...,C
    S = {}  # dict - Node swapping (score, distribution)
    for i in node_list:
        S[i] = swap_pds(P[(i - 1, i)], P[(i, i + 1)], q[i - 1], cutoff, Bq)

    for sn in swap_sq:  # swapping loop
        # if prnt:
        # scores = [S[i][0] for i in node_list]
        # print(scores, '\tswap:', sn)

        x = np.where(Lc[:, sn])[0][0]
        z = np.where(Lc[sn, :])[0][0]
        # update link capacity
        Lc[x, z] = min(Lc[x, sn], Lc[sn, z])
        P[(x, z)] = S[sn][1]
        Lc[x, sn], Lc[sn, z] = 0, 0  # consumed
        # update score of adjacent nodes: X and Z if they are repeaters, not SD
        if x > 0:
            x_pre = np.where(Lc[:, x])[0][0]
            S[x] = swap_pds(P[(x_pre, x)], P[(x, z)], q[x - 1], cutoff, Bq)

        if z < path_len:
            z_post = np.where(Lc[z, :])[0][0]
            S[z] = swap_pds(P[(x, z)], P[(z, z_post)], q[z - 1], cutoff, Bq)

    p_e2e = P[(0, path_len)]
    EXT = np.dot(1 + np.arange(len(p_e2e)), p_e2e) / Ts

    runtime = time.time() - start_time

    if prnt:
        print("Swapping order", swap_sq, "  => E[ent.]: {:.6f}".format(EXT), "\ttime: {:.6f}".format(runtime))

    return {"order": swap_sq, "EXT": EXT, "P": P, "time": runtime}


def s_tr(tree, names=""):
    if names:
        for l in range(len(tree)):
            level = tree[l]
            for lf in range(len(level)):
                leaf = level[lf]
                tree[l][lf] = names[leaf - 1]

    tree_str = str(tree).replace("[[", "[").replace("]]", "]").replace(",", "")
    return tree_str


def voraswap(
    C: list[int],
    p: float | list[float] = 1,
    q: float | list[float] = 0.5,
    *,
    prnt=False,
    cutoff=0.9995,
    Bq: np.ndarray,
    Ts: float = 1,
    names="",
):
    """Compare GREEDYSWAP vs. a BALANCEDTREE, and return:
    - EXT:      expected thruput
    - order:    swapping order
    - P:        Probability distributions of all links, physical & virtual
    - scores:   [GreedySwap_score, BalancedTree_score]
    - time:     runtime
    - greedy_order
    """
    path_len = len(C)
    tree = bal_tree(list(range(1, path_len)))

    if max(C) > Bq.shape[0]:
        print("Warning: Consider increasing Bq size to", max(C))
        Bq = get_Bq(int(int(max(C) / 1e3) + 1) * 1000, q)

    greedy_heu = greedyswap(C, p, q, Bq=Bq, cutoff=cutoff, Ts=Ts)

    baln_heu = thruput(C, tree2order(tree), p, q, Bq=Bq, cutoff=cutoff, Ts=Ts)
    baln_heu["order"] = tree2order(tree)

    if prnt:
        print("\nThruput Est. (w/Prob tail truncation@" + str(cutoff * 100) + "%):")
        print(
            "\tgreedyswap = %.2f" % (greedy_heu["EXT"]),
            "\truntime = %.3e (s)" % (greedy_heu["time"]),
            "\tSwapOrder:",
            s_tr(order2tree(greedy_heu["order"]), names),
            "\t <== HEU. SOL." if greedy_heu["EXT"] >= baln_heu["EXT"] else "",
        )

        print(
            "\tbaln__tree = %.2f" % (baln_heu["EXT"]),
            "\truntime = %.3e (s)" % (baln_heu["time"]),
            "\tSwapOrder:",
            s_tr(tree, names),
            "\t <== HEU. SOL." if greedy_heu["EXT"] < baln_heu["EXT"] else "",
        )

    res = greedy_heu if greedy_heu["EXT"] >= baln_heu["EXT"] else baln_heu
    res["scores"] = [greedy_heu["EXT"], baln_heu["EXT"]]
    res["greedy_order"] = greedy_heu["order"]
    return res


def net_chain3(L, p=None, q=None, C=None, M=None, note="", unit=""):
    # Example usage:
    # L_values = [1, 2, 3, 4]
    # net_chain(L_values, 0.5, 0.8)
    l = "---"
    e = ""
    net = f"[S]{l}"
    for i in range(len(L) - 1):
        net += f"{e}{L[i]:.3f}{unit}{e}{l}(R{i + 1}){l}"
    net += f"{e}{L[-1]:.3f}{unit}{e}{l}[D]" + note
    if len(L) > 1:
        print("\nPath:", net)
    else:
        print("\nLink:", net[2:-2])

    if M is not None:
        if type(M) is int:
            M = [M] * (len(L) + 1)
        M_line = " " * len(net)
        for i in range(len(M)):
            if i == 0:
                ri = 0
            elif i == len(M) - 1:
                ri = net.find("[D]")
            else:
                ri = net.find(f"(R{i})")
            M_str = str(M[i])
            M_line = M_line[:ri] + M_str + M_line[ri + len(M_str) :]
        print(" Mem: ", M_line)

    if C is not None:
        C_line = " " * len(net)
        for i in range(len(C)):
            ri = 2 if i == 0 else net.find(f"(R{i})") + 3
            C_str = str(int(C[i]))
            C_line = C_line[:ri] + C_str + C_line[ri + len(C_str) :]
        print("   C:   ", C_line)
    if p is not None:
        if type(p) is int or type(p) is float:
            print("p:" + " " * 8 + "%.2f" % (p))
        else:
            p_line = " " * len(net)
            for i in range(len(L)):
                ri = 4 if i == 0 else net.find(f"(R{i})") + 5
                p_str = "%.3f" % (p[i])
                p_line = p_line[:ri] + p_str + p_line[ri + 5 :]

            print("   p:", p_line)

    if q is not None:
        print("   q: " + " " * net.find(f"(R{1})"), q)


def plot_path(L_list, M=None, order=None, endnodes=["S", "D"]):
    n = len(L_list) + 1
    # Create a graph
    G = nx.Graph()
    nodes = [str(i) for i in range(n)]
    if order is not None:
        nodes[0], nodes[-1] = endnodes
        for i, node in enumerate(order):
            nodes[node] = str(i + 1)

    G.add_nodes_from(nodes)

    # Add edges with distances
    edges = [(nodes[i], nodes[i + 1], L_list[i]) for i in range(n - 1)]
    G.add_weighted_edges_from(edges)

    # Create a custom horizontal layout
    pos = {}
    pos_lower = {}
    current_x = 0
    for i, node in enumerate(nodes):
        pos[node] = (current_x, 0)
        pos_lower[node] = (current_x, -0.02)
        if i < len(edges):
            current_x += edges[i][2] / 10  # Scale down the distances for better visualization

    plt.figure(figsize=(1.5 * n, 0.3))  # , constrained_layout=True)
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=12)  # , font_weight='bold')
    # mem={}
    # if M is not None:
    #     mem['S']=M[0]
    #     for i in range(len(M)-1): mem[nodes[i]] =(M[i]+M[i+1])
    #     mem['D']=M[-1]
    #     nx.draw_networkx_labels(G, pos_lower, mem, font_size=10)

    labels = nx.get_edge_attributes(G, "weight")
    for i, lab in enumerate(labels):
        labels[lab] = "%.1f" % (labels[lab]) + " km"
        if M is not None:
            labels[lab] += "\nM: " + str(M[i]) if type(M[i]) is int else "\nM: %.1f" % (M[i])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()
    plt.close()
