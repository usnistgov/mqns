import time

from mqns.utils import rng


def pull_1000_randoms():
    return [rng.integers(1000000) for _ in range(1000)]


def test_rng_seed():
    unseeded = pull_1000_randoms()

    seed = int(time.time())
    rng.reseed(seed)
    seeded0 = pull_1000_randoms()

    rng.reseed(seed)
    seeded1 = pull_1000_randoms()

    assert unseeded != seeded0
    assert seeded0 == seeded1
