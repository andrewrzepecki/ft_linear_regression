try:
    from Model import Model
except Exception:
    from src.Model import Model

class Optimizer():

    def __init__(self, model : Model):
        self.loss_history = []
        self.model = model

    def __call__(self, loss):
        return self.forward(loss)

    def forward(self, loss):
        early_stop = False
        if len(self.loss_history) > 1 and self.loss_history[-1] < loss:
            self.model.lr = self.model.lr / 10
        if len(self.loss_history) > 10:
            early_stop = True
            for l in self.loss_history[-10:]:
                if l > loss:
                    early_stop = False
        self.loss_history.append(loss)
        return early_stop