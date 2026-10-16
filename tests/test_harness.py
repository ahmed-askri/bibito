import pytest

from bibito.harness import bootstrap_ci, distinct_outcomes, file_sha256, paired_diff


def test_bootstrap_ci_constant_values():
    lo, hi = bootstrap_ci([7.0] * 20)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_bootstrap_ci_contains_the_mean_and_is_ordered():
    values = [float(x) for x in range(1, 11)]  # mean 5.5
    lo, hi = bootstrap_ci(values)
    assert lo <= 5.5 <= hi
    assert lo < hi


def test_bootstrap_ci_is_reproducible():
    values = [1.0, 4.0, 2.0, 9.0, 3.0]
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_diff_constant_gap():
    mean, lo, hi = paired_diff([3, 4, 5], [1, 2, 3])
    assert mean == pytest.approx(2.0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_paired_diff_rejects_different_lengths():
    with pytest.raises(ValueError):
        paired_diff([1, 2, 3], [1, 2])


def test_distinct_outcomes_counts_unique_reward_and_tics():
    eps = [
        {"reward": 17, "steps": 1268},
        {"reward": 17, "steps": 1268},
        {"reward": 17, "steps": 1300},
        {"reward": 12, "steps": 1268},
    ]
    assert distinct_outcomes(eps) == 3


def test_file_sha256_same_content_same_hash_and_different_content_differs(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    c.write_bytes(b"world")
    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) != file_sha256(c)
    assert len(file_sha256(a)) == 12