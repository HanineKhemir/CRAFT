class ICBHIDataset:
    def __init__(self, metadata_csv=None, transform=None):
        self.transform = transform
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        waveform = [0.0] * 16000
        label = 0
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    return list(waveforms), list(labels)
