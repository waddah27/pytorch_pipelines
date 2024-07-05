from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DataLoaders:

    def __init__(self, data: str = "mnist", batch_size=64):
        self.batch_size = batch_size
        self.data = data
        self.data_loaders = {}
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if self.data == "mnist":
            training_data = datasets.MNIST("dataset/", train=True, download=True, transform=transform)
            testing_data = datasets.MNIST("dataset/", train=False, download=True, transform=transform)
            self.data_loaders = {
                "train": DataLoader(training_data, batch_size=self.batch_size, shuffle=True),
                "test": DataLoader(testing_data, batch_size=self.batch_size, shuffle=True)
        }

        return self.data_loaders