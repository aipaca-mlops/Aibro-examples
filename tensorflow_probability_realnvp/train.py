from utils import *
from train_model.real_nvp import RealNVPModel
from tensorflow.keras.optimizers import Adam
from aibro.training import online_fit

def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

class Trainer:

    def __init__(self):
        self.realnvp_model = RealNVPModel()
        self.realnvp_model.build((1, 32, 32, 3))

        self.train_ds = load_dataset('train')
        self.val_ds = load_dataset('val')
        self.test_ds = load_dataset('test')

        self.realnvp_model.compile(loss=nll, optimizer=Adam())
        # realnvp_model.fit(train_ds, validation_data=val_ds, epochs=20)

    def train(self):
        history = self.realnvp_model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=20,
        )
        return history

    
    def save(self):
        self.realnvp_model.save('model')


    def test(self):
        self.realnvp_model.evaluate(self.test_ds)
