import torch
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # encoder
        # takes 4D environment state, and encodes it to 2D
        self.enc_1 = nn.Linear(4, 16)
        self.enc_2 = nn.Linear(16, 16)
        self.enc_3 = nn.Linear(16, 2)

        # policy
        # takes 2D robot state and 2D encoded state, and outputs 2D action
        self.pi_1 = nn.Linear(2+2, 16)
        self.pi_2 = nn.Linear(16, 16)
        self.pi_mean = nn.Linear(16, 2)
        self.pi_std = nn.Linear(16, 2)

        # other stuff
        self.apply(weights_init_)
        self.mse_func = nn.MSELoss()

    def state_encoder(self, state):
        x = torch.relu(self.enc_1(state))
        x = torch.relu(self.enc_2(x))
        phi = torch.tanh(self.enc_3(x))
        return phi

    def policy(self, state, phi):
        x = torch.cat((state, phi), 1)
        x = torch.tanh(self.pi_1(x))
        x = torch.relu(self.pi_2(x))
        x_mean = self.pi_mean(x)
        x_std = torch.exp(0.5 * self.pi_std(x))
        eps = torch.randn_like(x_std)
        x = x_mean + x_std * eps
        return x
