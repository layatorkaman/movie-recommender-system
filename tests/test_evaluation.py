import pytest

from src.evaluation.evaluation import (
    validate_test_size,
    validate_max_test_samples,
)

from src.evaluation.evaluation import (
    validate_test_size,
    validate_max_test_samples,
    validate_min_common_ratings,
)

@pytest.mark.parametrize(
    "test_size",
    [0, 1, -0.2, 1.5],
)
def test_validate_test_size_rejects_invalid_values(test_size):
    with pytest.raises(ValueError):
        validate_test_size(test_size)


@pytest.mark.parametrize(
    "max_test_samples",
    [0, -1, 1.5, True],
)
def test_validate_max_test_samples_rejects_invalid_values(
    max_test_samples,
):
    with pytest.raises(ValueError):
        validate_max_test_samples(max_test_samples)

def test_validate_test_size_accepts_valid_value():
    validate_test_size(0.2)


def test_validate_max_test_samples_accepts_valid_value():
    validate_max_test_samples(500)

@pytest.mark.parametrize(
    "min_common_ratings",
    [0, -1, 1.5, True],
)
def test_validate_min_common_ratings_rejects_invalid_values(
    min_common_ratings,
):
    with pytest.raises(ValueError):
        validate_min_common_ratings(min_common_ratings)


def test_validate_min_common_ratings_accepts_valid_value():
    validate_min_common_ratings(3)