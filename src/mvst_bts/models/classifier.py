class ClassifierHead:
    def __init__(self, in_dim, n_classes):
        self.in_dim = in_dim
        self.n_classes = n_classes

    def forward(self, x):
        return x
