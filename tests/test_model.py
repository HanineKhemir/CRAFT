from mvst_bts.models.mvst_bts_plus import MVSTBTSPlus

def test_model_forward():
    model = MVSTBTSPlus()
    out = model.forward([0.0, 1.0])
    assert out == [0.0, 1.0]
