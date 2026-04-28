from mvst_bts.data.dataset import ICBHIDataset

def test_dataset_getitem():
    ds = ICBHIDataset()
    item = ds.__getitem__(0)
    waveform, label = item
    assert len(waveform) == 16000
    assert label == 0
