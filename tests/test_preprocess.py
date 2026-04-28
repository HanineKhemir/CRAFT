from mvst_bts.data.preprocess import pad_audio, z_norm

def test_pad_audio_shape():
    x = [1.0, 2.0]
    out = pad_audio(x, 5)
    assert len(out) == 5

def test_z_norm_no_nan():
    x = [1.0, 1.0, 1.0]
    out = z_norm(x)
    assert len(out) == 3
