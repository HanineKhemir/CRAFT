from mvst_bts.augmentation.patch_mix import patch_mix
from mvst_bts.augmentation.rep_augment import rep_augment

def test_patch_mix_identity():
    x = [[0.0, 1.0]]
    assert patch_mix(x) == x

def test_rep_augment_identity():
    x = [0.0, 1.0]
    assert rep_augment(x) == x
