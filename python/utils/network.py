import torch as th
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 units=16,
                 layers=1,
                 ):
        super(Network, self).__init__()
        assert layers > 0 and isinstance(layers, int), \
               "Layers should be an positive int"
        assert units > 0 and isinstance(units, int), \
               "Units should be an positive int"
        self.input_dim = input_dim
        self.layers = layers
        self.model = nn.ModuleList([nn.Linear(input_dim, units)])
        self.model.append(nn.ReLU())
        if layers > 1:
            for i in range(layers):
                if i < layers//2:
                    input_units = units*2**i
                    output_units = units*2**(i+1)
                elif layers % 2 == 1 and i == layers//2:
                    input_units = units*2**i
                    output_units = input_units
                elif i >= layers//2:
                    input_units = units*2**(layers-i)
                    output_units = units*2**(layers-i-1)
                else:
                    raise Exception("Unknown error")
                self.model.append(nn.Linear(input_units, output_units))
                self.model.append(nn.ReLU())
        self.model.append(nn.Linear(units, output_dim))

    def forward(self, x: th.Tensor) -> th.tensor:
        x = x.view(-1, self.input_dim)
        for _, layer in enumerate(self.model):
            x = layer(x)
        return x
