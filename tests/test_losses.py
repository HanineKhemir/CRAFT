from mvst_bts.training.losses import FocalLoss

def test_focal_loss_zero():
    loss = FocalLoss()
    assert loss([], []) == 0.0
