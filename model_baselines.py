import torch
import torch.nn as nn
import torch.nn.functional as F


# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create fully-connected model
class SimpleNet(nn.Module):
    def __init__(self, input_size, n_classes):
        """
        Initializes a new instance of the SimpleNet class.

        Args:
            input_size (int): The size of the input data.
            n_classes (int): The number of classes in the output.

        Initializes the following attributes:
            - fc1 (nn.Linear): A fully connected linear layer with input size `input_size` and output size 50.
            - fc2 (nn.Linear): A fully connected linear layer with input size 50 and output size `n_classes`.
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create convolutional model (CNN)
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1) # n_out = 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #n_out = 14x14
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # n_out = 7x7
        self.linear1 = nn.Linear(16*7*7, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        return x

# Create RNN model
class SimpleGRU(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, n_classes):
        super(SimpleGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # also could use nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size*self.sequence_length, n_classes)

    def forward(self, x):
        """
        x.shape = (batch_size, sequence_length, input_size)
        out.shape = (batch_size, hidden_size*sequence_length)
        """
        x = x.squeeze(1) # remove channel dimension

        # Initialize the hidden state with zeros: actually no need to initialize the hidden state \\
        # because it will be initialized to zero and the LSTM will take care of it but just in case\\
        # we initialize it here
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # Perform the forward pass through the RNN
        out, _ = self.rnn(x, h0)
        # Flatten the output tensor
        out = out.reshape(out.shape[0], -1)
        # Pass the output through the fully connected layer
        out = self.fc(out)
        return out
# Create a bidirectional LSTM model
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes):
        super(BLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, n_classes)

    def forward(self, x):
        x = x.squeeze(1) # remove channel dimension
        # here there is no need to initialize the hidden state or cell state because they will be initialized to zero
        # and the LSTM will take care of it but just in case we initialize them
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.LSTM(x, (h0, c0)) # here the output of self.LSTM is a tuple (output, (hidden_state:hn, cell_state:cn))
        out = out[:, -1, :] # take the last time step
        out = self.fc(out)
        return out

