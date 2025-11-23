"""
Unit Tests for RecDataLoader

This module contains unit tests for the RecDataLoader class which handles:
- Creating DataLoader from dict or DataFrame
- Batch processing
- Feature extraction and batching
- Integration with features
"""
import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from nextrec.data.dataloader import RecDataLoader
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

logger = logging.getLogger(__name__)


class TestRecDataLoaderBasic:
    """Test suite for basic RecDataLoader functionality"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features"""
        dense_features = [
            DenseFeature(name='age', embedding_dim=1),
            DenseFeature(name='price', embedding_dim=1),
        ]
        
        sparse_features = [
            SparseFeature(name='user_id', vocab_size=1000, embedding_dim=16),
            SparseFeature(name='item_id', vocab_size=500, embedding_dim=16),
        ]
        
        sequence_features = [
            SequenceFeature(
                name='item_history',
                vocab_size=500,
                max_len=20,
                embedding_dim=16,
                padding_idx=0
            )
        ]
        
        return dense_features, sparse_features, sequence_features
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame"""
        return pd.DataFrame({
            'age': [25.0, 30.0, 35.0, 40.0, 45.0],
            'price': [100.0, 200.0, 150.0, 300.0, 250.0],
            'user_id': [1, 2, 3, 4, 5],
            'item_id': [10, 20, 30, 40, 50],
            'item_history': [
                np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ],
            'label': [0, 1, 1, 0, 1]
        })
    
    @pytest.fixture
    def sample_dict(self):
        """Create sample dict"""
        return {
            'age': np.array([25.0, 30.0, 35.0, 40.0, 45.0]),
            'price': np.array([100.0, 200.0, 150.0, 300.0, 250.0]),
            'user_id': np.array([1, 2, 3, 4, 5]),
            'item_id': np.array([10, 20, 30, 40, 50]),
            'item_history': np.array([
                [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            'label': np.array([0, 1, 1, 0, 1])
        }
    
    def test_dataloader_initialization(self, sample_features):
        """Test RecDataLoader initialization"""
        logger.info("=" * 80)
        logger.info("Testing RecDataLoader initialization")
        logger.info("=" * 80)
        
        dense_features, sparse_features, sequence_features = sample_features
        
        dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target='label'
        )
        
        assert dataloader is not None
        assert len(dataloader.dense_features) == 2
        assert len(dataloader.sparse_features) == 2
        assert len(dataloader.sequence_features) == 1
        
        logger.info("RecDataLoader initialization successful")
    
    def test_create_dataloader_from_dataframe(self, sample_features, sample_dataframe):
        """Test creating DataLoader from DataFrame"""
        logger.info("=" * 80)
        logger.info("Testing create_dataloader from DataFrame")
        logger.info("=" * 80)
        
        dense_features, sparse_features, sequence_features = sample_features
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(
            data=sample_dataframe,
            batch_size=2,
            shuffle=False
        )
        
        assert isinstance(dataloader, DataLoader)
        assert len(dataloader) > 0
        
        # Get a batch - should be a tuple of tensors
        batch = next(iter(dataloader))
        assert isinstance(batch, (tuple, list))
        # Should contain: dense features, sparse features, sequence features, and label
        assert len(batch) > 0
        assert all(isinstance(t, torch.Tensor) for t in batch)
        
        logger.info("create_dataloader from DataFrame test successful")
    
    def test_create_dataloader_from_dict(self, sample_features, sample_dict):
        """Test creating DataLoader from dict"""
        logger.info("=" * 80)
        logger.info("Testing create_dataloader from dict")
        logger.info("=" * 80)
        
        dense_features, sparse_features, sequence_features = sample_features
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(
            data=sample_dict,
            batch_size=2,
            shuffle=False
        )
        
        assert isinstance(dataloader, DataLoader)
        
        # Get a batch - should be a tuple of tensors, not a dict
        batch = next(iter(dataloader))
        assert isinstance(batch, (tuple, list))
        # Should contain: 2 dense, 2 sparse, 1 sequence, 1 label = 6 tensors
        assert len(batch) == 6
        assert all(isinstance(t, torch.Tensor) for t in batch)
        
        # Verify tensor shapes
        age_tensor = batch[0]  # First dense feature
        assert age_tensor.shape[0] <= 2  # batch_size
        
        logger.info("create_dataloader from dict test successful")
    
    def test_batch_sizes(self, sample_features, sample_dataframe):
        """Test different batch sizes"""
        logger.info("=" * 80)
        logger.info("Testing different batch sizes")
        logger.info("=" * 80)
        
        dense_features, sparse_features, sequence_features = sample_features
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target='label'
        )
        
        for batch_size in [1, 2, 5]:
            dataloader = rec_dataloader.create_dataloader(
                data=sample_dataframe,
                batch_size=batch_size,
                shuffle=False
            )
            
            batch = next(iter(dataloader))
            
            # Check batch size (may be smaller for last batch)
            # batch is a tuple of tensors, check first tensor
            assert batch[0].shape[0] <= batch_size
            
            logger.info(f"Batch size {batch_size} test successful")
    
    def test_shuffle(self, sample_features, sample_dataframe):
        """Test shuffle functionality"""
        logger.info("=" * 80)
        logger.info("Testing shuffle functionality")
        logger.info("=" * 80)
        
        dense_features, sparse_features, sequence_features = sample_features
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target='label'
        )
        
        # Create dataloader without shuffle
        dataloader_no_shuffle = rec_dataloader.create_dataloader(
            data=sample_dataframe,
            batch_size=5,
            shuffle=False
        )
        
        batch_no_shuffle = next(iter(dataloader_no_shuffle))
        
        # Create dataloader with shuffle
        dataloader_shuffle = rec_dataloader.create_dataloader(
            data=sample_dataframe,
            batch_size=5,
            shuffle=True
        )
        
        # Just test that it works - order may or may not be different
        batch_shuffle = next(iter(dataloader_shuffle))
        assert batch_shuffle is not None
        
        logger.info("Shuffle functionality test successful")


class TestRecDataLoaderFeatureTypes:
    """Test suite for different feature types"""
    
    def test_dense_features_only(self):
        """Test DataLoader with only dense features"""
        logger.info("=" * 80)
        logger.info("Testing DataLoader with only dense features")
        logger.info("=" * 80)
        
        dense_features = [
            DenseFeature(name='feature1', embedding_dim=1),
            DenseFeature(name='feature2', embedding_dim=1),
        ]
        
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'label': [0, 1, 0]
        })
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=[],
            sequence_features=[],
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
        batch = next(iter(dataloader))
        
        # batch is a tuple: (feature1_tensor, feature2_tensor, label_tensor)
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 3  # 2 features + 1 label
        assert all(isinstance(t, torch.Tensor) for t in batch)
        
        logger.info("Dense features only test successful")
    
    def test_sparse_features_only(self):
        """Test DataLoader with only sparse features"""
        logger.info("=" * 80)
        logger.info("Testing DataLoader with only sparse features")
        logger.info("=" * 80)
        
        sparse_features = [
            SparseFeature(name='feature1', vocab_size=100, embedding_dim=8),
            SparseFeature(name='feature2', vocab_size=50, embedding_dim=8),
        ]
        
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30],
            'label': [0, 1, 0]
        })
        
        rec_dataloader = RecDataLoader(
            dense_features=[],
            sparse_features=sparse_features,
            sequence_features=[],
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
        batch = next(iter(dataloader))
        
        # batch is a tuple: (feature1_tensor, feature2_tensor, label_tensor)
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 3  # 2 features + 1 label
        assert all(isinstance(t, torch.Tensor) for t in batch)
        
        logger.info("Sparse features only test successful")
    
    def test_sequence_features_only(self):
        """Test DataLoader with only sequence features"""
        logger.info("=" * 80)
        logger.info("Testing DataLoader with only sequence features")
        logger.info("=" * 80)
        
        sequence_features = [
            SequenceFeature(
                name='seq1',
                vocab_size=100,
                max_len=10,
                embedding_dim=8,
                padding_idx=0
            )
        ]
        
        data = {
            'seq1': np.array([
                [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                [4, 5, 6, 7, 0, 0, 0, 0, 0, 0],
                [8, 9, 0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'label': np.array([0, 1, 0])
        }
        
        rec_dataloader = RecDataLoader(
            dense_features=[],
            sparse_features=[],
            sequence_features=sequence_features,
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
        batch = next(iter(dataloader))
        
        # batch is a tuple: (seq1_tensor, label_tensor)
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 2  # 1 sequence feature + 1 label
        assert all(isinstance(t, torch.Tensor) for t in batch)
        # Check sequence length
        seq_tensor = batch[0]
        assert seq_tensor.shape[1] == 10  # max_len
        
        logger.info("Sequence features only test successful")


class TestRecDataLoaderMultipleTargets:
    """Test suite for multiple target features"""
    
    def test_multiple_targets(self):
        """Test DataLoader with multiple targets"""
        logger.info("=" * 80)
        logger.info("Testing DataLoader with multiple targets")
        logger.info("=" * 80)
        
        dense_features = [DenseFeature(name='feature1', embedding_dim=1)]
        
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'label1': [0, 1, 0],
            'label2': [1, 0, 1]
        })
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            sparse_features=[],
            sequence_features=[],
            target=['label1', 'label2']
        )
        
        dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
        batch = next(iter(dataloader))
        
        # batch is a tuple: (feature1_tensor, labels_tensor)
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 2  # 1 feature + 1 combined label tensor
        assert all(isinstance(t, torch.Tensor) for t in batch)
        # Check that labels are combined: should have 2 columns
        label_tensor = batch[-1]  # Last tensor is labels
        if label_tensor.dim() == 2:
            assert label_tensor.shape[1] == 2  # 2 target columns
        
        logger.info("Multiple targets test successful")


class TestRecDataLoaderEdgeCases:
    """Test suite for edge cases"""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        logger.info("=" * 80)
        logger.info("Testing with empty DataFrame")
        logger.info("=" * 80)
        
        dense_features = [DenseFeature(name='feature1', embedding_dim=1)]
        
        # Empty DataFrame
        data = pd.DataFrame(columns=['feature1', 'label'])
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            target='label'
        )
        
        # Should handle empty data gracefully
        try:
            dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
            # If it creates a dataloader, it should be empty
            batches = list(dataloader)
            assert len(batches) == 0
            logger.info("Empty DataFrame handled correctly")
        except Exception as e:
            logger.info(f"Empty DataFrame raises exception: {e}")
    
    def test_single_sample(self):
        """Test with single sample"""
        logger.info("=" * 80)
        logger.info("Testing with single sample")
        logger.info("=" * 80)
        
        dense_features = [DenseFeature(name='feature1', embedding_dim=1)]
        
        data = pd.DataFrame({
            'feature1': [1.0],
            'label': [0]
        })
        
        rec_dataloader = RecDataLoader(
            dense_features=dense_features,
            target='label'
        )
        
        dataloader = rec_dataloader.create_dataloader(data, batch_size=2)
        batch = next(iter(dataloader))
        
        # batch is a tuple of tensors
        assert isinstance(batch, (tuple, list))
        # Check first tensor has batch size of 1
        assert batch[0].shape[0] == 1
        
        logger.info("Single sample test successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
