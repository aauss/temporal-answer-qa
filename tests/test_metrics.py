import numpy as np

from temp_answer_qa.metrics import absolute_scaled_error, symmetric_absolute_percentage_error


def test_sape():
    predictions = np.array([-967, -967, 1307, 1307, 0, 0, np.nan])
    references = np.array([-967, +967, 1307, 1500, 0, 5, 5])
    expected = np.array([0.0, 100.0, 0.0, 6.87566797, 0.0, 100.0, 100.0])
    assert np.allclose(symmetric_absolute_percentage_error(predictions, references), expected)


def test_ase():
    predictions = np.array([-967, -967, 1307, 1307, 0, 0, np.nan])
    references = np.array([-967, +967, 1307, 1500, 0, 5, 5])
    deviances = np.array([20, 20, 1307, 250, 1, 1, 1])
    expected = np.array([0.0, 96.7, 0.0, 0.772, 0.0, 5.0, np.nan])
    assert np.allclose(
        absolute_scaled_error(predictions, references, deviances), expected, equal_nan=True
    )
