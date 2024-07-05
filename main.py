# =================================
# PyTorch baselines implementation
# =================================

from data_loader import DataLoaders
from model_train import model_trainer
from model_baselines import *

# Hyperparameters
model_name = "GRU"
input_size = 784
n_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# For RNN
in_size = 28
seq_len = 28
hidden_size = 128
num_layers = 2

# Initialize model: Choose one from the baselines
models = {
    "CNN": SimpleCNN(in_channels=1, n_classes=10),
    "GRU": SimpleGRU(input_size=in_size, sequence_length=seq_len,
                     hidden_size=hidden_size, num_layers=num_layers, n_classes=n_classes),
    "LSTM": BLSTM(input_size=in_size, hidden_size=hidden_size,
                  num_layers=num_layers, n_classes=n_classes),
}

assert model_name in models.keys(), "Invalid model name. Please choose from: " + ", ".join(list(models.keys()))
model = models[model_name].to(DEVICE)

#load Datasets
data = "mnist"
data_loaders = DataLoaders(data=data, batch_size=batch_size).load_data()
# model trainer: wrapping the model over the dataset
trainer = model_trainer(model=model, model_name=model_name, n_epochs=num_epochs, lr=learning_rate, is_rnn=True) # set is_rnn = True for RNN, LSTM, GRU

# training the model using the dataset and validating it
trainer.train(loaders=data_loaders, load_checkpoint=True)
