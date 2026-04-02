"""
Convert params-*.toml to bash variables.
"""

from srt_detail.defs import ParamsArgs

if __name__ == "__main__":
    args = ParamsArgs().parse_args()
    print(f"SEED_BASE={args.params['seed_base']}")
    print(f"RUNS={args.params['runs']}")
    print(f"ENABLE_SEQUENCE={int(args.params['enable_sequence'])}")
    print(f"NS_NODES=({' '.join([str(s['nodes']) for s in args.params['network_sizes']])})")
    print(f"NS_EDGES=({' '.join([str(s['edges']) for s in args.params['network_sizes']])})")
    print(f"CPUSET_CPUS=({' '.join([str(c) for c in args.params.get('cpuset_cpus', [])])})")
