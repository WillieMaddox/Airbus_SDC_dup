import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def save_checkpoint(filepath, model):
    checkpoint = {'input_size': model.input_shape,
                  'output_size': model.output_size,
                  'conv_layers': model.conv_layers,
                  'fc_layers': model.fc_layers,
                  # 'conv_layers': [each.out_features for each in model.conv_layers],
                  # 'fc_layers': [each.out_features for each in model.fc_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = DupCNN(checkpoint['input_size'],
                   checkpoint['output_size'],
                   checkpoint['conv_layers'],
                   checkpoint['fc_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


def create_loss_and_optimizer(model, learning_rate=0.001):
    # loss = nn.MSELoss()
    loss = nn.BCELoss()
    # loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss, optimizer


class DupCNN(nn.Module):

    def __init__(self, input_shape, output_size, conv_layers, fc_layers):
        super(DupCNN, self).__init__()

        self.input_shape = input_shape
        self.output_size = output_size
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers

        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 16384 input features, 128 output features (see sizing flow below)
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        # 128 input features, 1 output feature
        self.fc2 = nn.Linear(128, output_size)

        # self.conv_layers = nn.ModuleList([nn.Conv2d(input_shape[0], conv_layers[0], kernel_size=3, stride=1, padding=1)])
        # conv_layer_sizes = zip(conv_layers[:-1], conv_layers[1:])
        # for h1, h2 in conv_layer_sizes:
        #     self.conv_layers.extend([nn.Conv2d(h1, h2, kernel_size=3, stride=1, padding=1)])
        # self.pool = nn.MaxPool2d(kernel_size=2)
        # k_size = input_shape[1] >> len(conv_layers)
        # self.flat_size = conv_layers[-1] * k_size * k_size
        # self.fc_layers = nn.ModuleList([nn.Linear(self.flat_size, fc_layers[0])])
        # if len(fc_layers) > 1:
        #     fc_layer_sizes = zip(fc_layers[:-1], fc_layers[1:])
        #     self.fc_layers.extend([nn.Linear(fc1, fc2) for fc1, fc2 in fc_layer_sizes])
        # self.output = nn.Linear(fc_layers[-1], output_size)

    def forward(self, x):

        # for each in self.conv_layers:
        #     x = F.relu(each(x))
        #     x = self.pool(x)
        # x = x.view(-1, self.flat_size)
        # for each in self.fc_layers:
        #     x = F.relu(each(x))
        # x = self.output(x)
        # return torch.sigmoid(x)

        x = self.conv1(x)  # Size changes from (6, 256, 256) to (16, 256, 256)
        x = F.relu(x)  # Computes the activation of the first convolution
        x = self.pool(x)  # Size changes from (16, 256, 256) to (16, 128, 128)

        x = self.conv2(x)  # Size changes from (16, 128, 128) to (32, 128, 128)
        x = F.relu(x)  # Computes the activation of the second convolution
        x = self.pool(x)  # Size changes from (32, 128, 128) to (32, 64, 64)

        x = self.conv3(x)  # Size changes from (32, 64, 64) to (64, 64, 64)
        x = F.relu(x)  # Computes the activation of the third convolution
        x = self.pool(x)  # Size changes from (64, 64, 64) to (64, 32, 32)

        x = self.conv4(x)  # Size changes from (64, 32, 32) to (128, 32, 32)
        x = F.relu(x)  # Computes the activation of the fourth convolution
        x = self.pool(x)  # Size changes from (128, 32, 32) to (128, 16, 16)

        x = self.conv5(x)  # Size changes from (128, 16, 16) to (256, 16, 16)
        x = F.relu(x)  # Computes the activation of the fifth convolution
        x = self.pool(x)  # Size changes from (256, 16, 16) to (256, 8, 8)

        # Reshape data to input to the input layer of the neural net
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 256 * 8 * 8)  # Size changes from (256, 8, 8) to (1, 16384)

        x = self.fc1(x)  # Size changes from (1, 16384) to (1, 128)
        x = F.relu(x)  # Computes the activation of the first fully connected layer

        x = self.fc2(x)  # Size changes from (1, 128) to (1, 1)
        x = torch.sigmoid(x)  # Computes the activation of the second fully connected layer
        return x
