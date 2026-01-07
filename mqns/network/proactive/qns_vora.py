import pandas as pd
from vora_utils import get_Bq, merge_close, net_chain3, plot_path, voraswap

path = "./vora-training_20251125/"

scenarios = ["3", "4", "5"]
dist_alloc = ["decreasing", "increasing", "mid_bottleneck", "uniform"]

q = 0.5  # swapping prob
M = 25  # memory pairs per link
T_cutoff = 10e-3

# precompute Binomial coeffs
max_capacity = 2000
Bq = get_Bq(max_capacity, q)

alldata = {}

for s in scenarios:
    print(s)
    for dist_profile in dist_alloc:
        print("\n", dist_profile.upper())

        # get link EVAL data
        data = pd.read_csv(path + s + "-" + dist_profile + "-1.csv")
        alldata[s + "-" + dist_profile] = data
        print(data.head(10))

        # Link characteristics
        L_list = merge_close(list(data["L"]))
        C0 = merge_close(list(data["Attempts rate"]))
        P = merge_close(list(data["Success rate"]), 0.75)

        tau = 2 * sum(L_list) / 2e5  # for heralding
        T_ext = T_cutoff - tau  # for external phase

        # actual capaciy for finding order: #attemps/ time_slot
        C = [round(c * M * T_ext) for c in C0]

        # print for fun
        net_chain3(L_list, P, q, C, M=2 * M, note="\t (Km)\t" + dist_profile.upper())

        # find order
        vora = voraswap(C, P, q, Bq=Bq, Ts=T_cutoff, prnt=True, cutoff=0.9995)

        # plot for fun
        plot_path(L_list, M=[M] * len(C), order=vora["order"])
