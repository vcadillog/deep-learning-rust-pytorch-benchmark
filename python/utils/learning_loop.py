import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from utils.logger import Logger, LogCSV, LogConsole
from utils.network import Network


class LearningAlgorithm:
    def __init__(self, config):
        super(LearningAlgorithm, self).__init__()
        self.EPOCHS = config['epochs']
        self.SHUFFLE = config['shuffle']
        self.IMAGE_DIM = config['image_dim']
        self.LABELS = config['labels']
        self.UNITS = config['units']
        self.BATCH_SIZE = config['batch_size']
        self.TEST_BATCH_SIZE = config['test_batch_size']
        self.RUNS = config['runs']
        self.LEARNING_RATE = config['learning_rate']
        self.DATA_PATH = config['data_path']
        self.LOG_DIR = config['log_dir']
        self.USE_CUDA = config['use_cuda']
        self.LAYERS_CPU = config['layers_cpu']
        self.LAYERS_CUDA = config['layers_cuda']
        self.FILENAME = 'python_log_'

    def train_epoch(self, layers):

        self.model = Network(self.IMAGE_DIM, self.LABELS, self.UNITS, layers)
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        training_time = time.time()

        for epoch in range(self.EPOCHS):
            for (_, (train_data, train_target)) in enumerate(self.train_loader):
                self.model.train()
                train_data = train_data.to(self.device)
                train_target = train_target.to(self.device)
                output = self.model(train_data)
                loss = F.cross_entropy(output, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        training_time = time.time() - training_time

        return loss.item(), training_time

    def test_model(self):
        test_size = self.test_loader.__len__()
        self.model.eval()
        accuracy = 0
        loss = 0
        test_time = time.time()

        for (_, (test_data, test_target)) in enumerate(self.test_loader):
            test_data = test_data.to(self.device)
            test_target = test_target.to(self.device)
            with th.no_grad():
                out_validation = self.model(test_data)
                loss += F.cross_entropy(out_validation, test_target)
                accuracy += test_target == th.argmax(out_validation)
        test_time = time.time() - test_time

        return 100*accuracy.item()/test_size, loss.item()/test_size, test_time

    def run_on_device(self):

        logs = Logger()
        log_csv = LogCSV()
        logs.set_names(['hidden_layers', 'run', 'training_loss',
                        'training_time', 'test_loss', 'test_accuracy',
                        'test_time'])

        for layers in self.LAYERS:
            for run in range(self.RUNS):
                loss, training_time = self.train_epoch(layers)
                accuracy, test_loss, test_time = self.test_model()
                logs.set_data([layers, run, loss, training_time, test_loss,
                               accuracy, test_time])
                logs.strategy = LogConsole()
                logs.get()
                logs.strategy = log_csv
                logs.get()

        log_csv.save_csv(self.LOG_DIR + '/' + self.FILENAME + self.device_name)

    def run(self):
        for use_cuda in self.USE_CUDA:
            train_kwargs = {'batch_size': self.BATCH_SIZE,
                            'shuffle': self.SHUFFLE,
                            'drop_last': True}
            test_kwargs = {'batch_size': self.TEST_BATCH_SIZE}
            if not use_cuda:
                self.device_name = 'cpu'
                self.LAYERS = self.LAYERS_CPU
            elif use_cuda:
                self.device_name = 'cuda'
                cuda_kwargs = {'num_workers': 1,
                               'pin_memory': True}

                train_kwargs.update(cuda_kwargs)
                test_kwargs.update(cuda_kwargs)
                self.LAYERS = self.LAYERS_CUDA

            else:
                raise Exception('Unknown device')

            transform = transforms.Compose([
                  transforms.ToTensor(),
              ])
            train_dataset = datasets.MNIST(self.DATA_PATH, train=True,
                                           download=True, transform=transform)
            test_dataset = datasets.MNIST(self.DATA_PATH, train=False,
                                          download=True, transform=transform)
            self.train_loader = th.utils.data.DataLoader(train_dataset,
                                                         **train_kwargs)
            self.test_loader = th.utils.data.DataLoader(test_dataset,
                                                        **test_kwargs)
            self.device = th.device(self.device_name)

            self.run_on_device()
