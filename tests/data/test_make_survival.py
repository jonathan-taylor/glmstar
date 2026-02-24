import numpy as np
import pandas as pd
import pytest
from glmnet.data import make_survival

@pytest.mark.parametrize("n_samples,n_features,n_informative,snr,baseline_hazard", [
    (50, 8, 4, None, 0.1),
    (30, 5, 2, 2.0, 0.2),
    (10, 10, 10, 1.5, 0.05),
])
@pytest.mark.parametrize("start_id", [False, True])
def test_survival_parametrized(n_samples, n_features, n_informative, snr, baseline_hazard, start_id):
    X, y, coef = make_survival(n_samples=n_samples, n_features=n_features, n_informative=n_informative, snr=snr, baseline_hazard=baseline_hazard, start_id=start_id)
    assert X.shape == (n_samples, n_features)
    assert isinstance(y, pd.DataFrame)
    expected_cols = {'event', 'status'} | ({'start'} if start_id else set())
    assert set(y.columns) == expected_cols
    assert coef.shape == (n_features,)
    assert np.all(y['event'] > 0)
    assert set(np.unique(y['status'])).issubset({0, 1})

def test_basic_survival():
    X, y, coef = make_survival(n_samples=50, n_features=8)
    assert X.shape == (50, 8)
    assert isinstance(y, pd.DataFrame)
    assert set(y.columns) == {'event', 'status'}
    assert coef.shape == (8,)
    assert np.all(y['event'] > 0)
    assert set(np.unique(y['status'])).issubset({0, 1})

def test_survival_start_id():
    X, y, coef = make_survival(n_samples=30, n_features=5, start_id=True)
    assert X.shape == (30, 5)
    assert set(y.columns) == {'start', 'event', 'status'}
    assert np.all(y['start'] >= 0)  # Start times should be non-negative
    assert np.all(y['start'] < y['event'])  # Start times should be strictly less than event times
    assert np.all(y['event'] > 0)
    assert set(np.unique(y['status'])).issubset({0, 1})


def test_survival_discretize():
    """Test that discretize option creates ties in the data."""
    # Test without discretization (should have no ties)
    X, y, coef = make_survival(n_samples=100, n_features=10, discretize=False, random_state=42)
    unique_times = len(np.unique(y['event']))
    assert unique_times == len(y['event'])  # No ties
    
    # Test with discretization (should have ties)
    X, y, coef = make_survival(n_samples=100, n_features=10, discretize=True, random_state=42)
    unique_times = len(np.unique(y['event']))
    assert unique_times < len(y['event'])  # Should have ties
    assert unique_times > 0  # Should still have some unique values


def test_survival_discretize_start_id():
    """Test discretize option with start_id=True."""
    # Test without discretization
    X, y, coef = make_survival(n_samples=50, n_features=8, start_id=True, discretize=False, random_state=42)
    unique_event_times = len(np.unique(y['event']))
    unique_start_times = len(np.unique(y['start']))
    assert unique_event_times == len(y['event'])  # No ties in event times
    assert unique_start_times > 1  # Should have multiple unique start times
    assert np.all(y['start'] < y['event'])  # Start times should be strictly less than event times
    
    # Test with discretization
    X, y, coef = make_survival(n_samples=50, n_features=8, start_id=True, discretize=True, random_state=42)
    unique_event_times = len(np.unique(y['event']))
    unique_start_times = len(np.unique(y['start']))
    assert unique_event_times < len(y['event'])  # Should have ties in event times
    assert unique_start_times > 1  # Should have multiple unique start times (discretized)
    assert np.all(y['start'] < y['event'])  # Start times should be strictly less than event times


def test_survival_discretize_rounding():
    """Test that discretization properly rounds to 1 decimal place."""
    X, y, coef = make_survival(n_samples=50, n_features=5, discretize=True, random_state=42)
    
    # Check that all event times are rounded to 1 decimal place
    for time in y['event']:
        # Convert to string and check decimal places
        time_str = f"{time:.10f}".rstrip('0').rstrip('.')
        if '.' in time_str:
            decimal_places = len(time_str.split('.')[1])
            assert decimal_places <= 1, f"Time {time} has more than 1 decimal place"
    
    # Check that we have some ties
    unique_times = len(np.unique(y['event']))
    assert unique_times < len(y['event']), "No ties created with discretize=True"


def test_survival_start_times_reasonable():
    """Test that start times are reasonable compared to event times."""
    X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True, random_state=42)
    
    # Check that start times are strictly less than event times
    assert np.all(y['start'] < y['event']), "Some start times are not strictly less than event times"
    
    # Check that start times are not too small (should be at least 10% of event times)
    start_event_ratio = y['start'] / y['event']
    assert np.all(start_event_ratio >= 0.1), "Some start times are too small relative to event times"
    
    # Check that start times are not too close to event times (should be at most 80% of event times)
    assert np.all(start_event_ratio <= 0.8), "Some start times are too close to event times"

