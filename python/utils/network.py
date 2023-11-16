import torch.nn as nn


class Network(nn.Sequential):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 units=16,
                 hidden_layers=1,
                 ):
        super(Network, self).__init__()
        assert hidden_layers > 0 and isinstance(hidden_layers, int), \
               "Layers should be an positive int"
        assert units > 0 and isinstance(units, int), \
               "Units should be an positive int"
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = nn.Sequential(
                nn.Linear(input_dim, units),
                nn.ReLU()
                )
        if hidden_layers > 1:
            for i in range(hidden_layers):
                if i < hidden_layers//2:
                    input_units = 2**i
                    output_units = 2**(i+1)
                elif hidden_layers % 2 == 1 and i == hidden_layers//2:
                    input_units = 2**i
                    output_units = input_units
                elif i >= hidden_layers//2:
                    input_units = 2**(hidden_layers-i)
                    output_units = 2**(hidden_layers-i-1)
                else:
                    raise Exception("Unknown error")
                input_units *= units
                output_units *= units
                self.model.append(nn.Linear(input_units, output_units))
                self.model.append(nn.ReLU())
        self.model.append(nn.Linear(units, output_units))
