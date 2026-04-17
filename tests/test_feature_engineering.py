from ar_nids.config import ARNIDSConfig
from ar_nids.data import make_synthetic_dataset
from ar_nids.feature_engineering import prepare_training_data


def test_prepare_training_data_shapes():
    config = ARNIDSConfig()
    dataset = make_synthetic_dataset(config, flows=32)
    prepared = prepare_training_data(dataset.frame, config)
    assert prepared.packet_windows.shape == (32, 50, 80)
    assert prepared.sequences.shape == (32, 100, 40)
    assert prepared.labels.shape == (32,)
