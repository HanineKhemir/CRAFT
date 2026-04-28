class FocalLoss:
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, logits, targets):
        return 0.0

class SoftFocalLoss(FocalLoss):
    pass
