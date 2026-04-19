import numpy as np
import pytest

from nextrec.basic.features import DenseFeature, SequenceFeature, SparseFeature
from nextrec.utils.onnx_utils import to_dense_array, to_sequence_array, to_sparse_array


def test_to_dense_array_reshapes_1d_input():
    feature = DenseFeature(name="dense", input_dim=1)
    array = to_dense_array(feature, [1.0, 2.0, 3.0])

    assert array.shape == (3, 1)
    assert array.dtype == np.float32


def test_to_dense_array_validates_feature_width():
    feature = DenseFeature(name="dense", input_dim=2)

    with pytest.raises(ValueError, match="expects 2 dims"):
        to_dense_array(feature, [1.0, 2.0, 3.0])


def test_to_sparse_array_flattens_single_column():
    feature = SparseFeature(name="sparse", vocab_size=10)
    array = to_sparse_array(feature, [[1], [2], [3]])

    assert array.shape == (3,)
    assert array.dtype == np.int64


def test_to_sequence_array_pads_and_truncates():
    feature = SequenceFeature(name="sequence", vocab_size=10, max_len=4, padding_idx=0)

    padded = to_sequence_array(feature, [[1, 2]])
    truncated = to_sequence_array(feature, [[1, 2, 3, 4, 5]])

    assert padded.tolist() == [[1, 2, 0, 0]]
    assert truncated.tolist() == [[1, 2, 3, 4]]
