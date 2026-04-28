def bandpass_filter(waveform, low=20, high=2000, sr=16000):
    """Stub bandpass filter."""
    return waveform

def resample_audio(waveform, orig_sr, target_sr=16000):
    """Stub resample."""
    return waveform

def pad_audio(waveform, target_len):
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return waveform + [0.0] * (target_len - len(waveform))

def z_norm(waveform):
    if not waveform:
        return waveform
    mean = sum(waveform) / len(waveform)
    var = sum((x - mean) ** 2 for x in waveform) / len(waveform)
    std = var ** 0.5 if var > 0 else 1.0
    return [(x - mean) / std for x in waveform]
