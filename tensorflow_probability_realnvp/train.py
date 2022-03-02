from utils import *
from train_model.real_nvp import RealNVPModel
from tensorflow.keras.optimizer import Adam
from aibro.training import online_fit


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


realnvp_model = RealNVPModel()
realnvp_model.build((1, 32, 32, 3))

train_ds = load_dataset('train')
val_ds = load_dataset('val')
test_ds = load_dataset('test')

realnvp_model.compile(loss=nll, optimizer=Adam())
# realnvp_model.fit(train_ds, validation_data=val_ds, epochs=20)
history = online_fit(
    model=realnvp_model,
    train_ds=train_ds,
    valid_ds=val_ds,
    epochs=20,
    directory_to_save_ckpt="./model",
)
