import hashlib
import random
import statistics


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """95% percentile-bootstrap confidence interval of the mean."""
    values = list(values)
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int(n_boot * alpha / 2)]
    hi = means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]
    return lo, hi


def paired_diff(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Mean of (a_i - b_i) with a bootstrap CI. a and b must come from the same games."""
    if len(a) != len(b):
        raise ValueError("paired lists must have the same length")
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, n_boot, alpha, seed)
    return statistics.fmean(diffs), lo, hi


def distinct_outcomes(episodes):
    """How many different (reward, tics) results there are. Seeds do not give independent
    games in defend_the_center, so this is an upper bound on the number of different games."""
    return len({(e["reward"], e["steps"]) for e in episodes})


def file_sha256(path):
    """First 12 hex chars of the file's SHA-256: identifies which model file was evaluated."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]