class Trainer:
    def __init__(self, model, optimizer=None, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_one_epoch(self, dataloader):
        return {"loss": 0.0}
