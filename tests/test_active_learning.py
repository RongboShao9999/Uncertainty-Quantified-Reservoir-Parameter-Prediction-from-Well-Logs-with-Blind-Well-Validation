import numpy as np
import pytest

from bnn_inversion.active_learning import score_pool, score_pool_components, select_batch


def test_mixed_score_normalizes_each_target_before_macro_average() -> None:
    epistemic = np.array([[0.0, 10.0], [1.0, 20.0]])
    inside = np.array([[True, True], [False, True]])

    score = score_pool(epistemic, inside, strategy="mixed", penalty=2.0)

    assert score.tolist() == [0.0, 2.0]


def test_constant_uncertainty_normalizes_to_zero() -> None:
    epistemic = np.ones((3, 2))
    inside = np.ones((3, 2), dtype=bool)
    assert score_pool(epistemic, inside, strategy="epistemic").tolist() == [0.0] * 3


def test_random_exploration_never_duplicates_high_score_samples() -> None:
    chosen = select_batch(
        np.arange(20.0),
        np.arange(20),
        budget=10,
        random_fraction=0.2,
        seed=4,
    )
    assert len(chosen) == len(set(chosen.tolist())) == 10
    assert set(range(12, 20)).issubset(chosen)


def test_selection_is_reproducible_and_validates_budget() -> None:
    first = select_batch(np.zeros(8), np.arange(8), budget=4, random_fraction=1.0, seed=9)
    second = select_batch(np.zeros(8), np.arange(8), budget=4, random_fraction=1.0, seed=9)
    np.testing.assert_array_equal(first, second)

    with pytest.raises(ValueError, match="budget"):
        select_batch(np.zeros(2), np.arange(2), budget=3, seed=1)


def test_score_pool_ignores_masked_targets() -> None:
    epistemic = np.array([[0.0, 10.0], [1.0, 20.0]])
    inside = np.array([[True, False], [False, False]])
    mask = np.array([[True, False], [True, False]])
    score = score_pool(epistemic, inside, strategy="mixed", penalty=2.0, mask=mask)
    assert score.tolist() == [0.0, 3.0]


def test_score_pool_components_expose_auditable_terms() -> None:
    epistemic = np.array([[0.0, 10.0], [1.0, 20.0]])
    inside = np.array([[True, True], [False, True]])

    components = score_pool_components(
        epistemic, inside, strategy="mixed", penalty=2.0
    )

    assert components["score"].tolist() == [0.0, 2.0]
    assert components["epistemic_component"].tolist() == [0.0, 1.0]
    assert components["inconsistency_component"].tolist() == [0.0, 1.0]
    assert components["valid_target_count"].tolist() == [2, 2]
