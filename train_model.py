import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import MyModel
import pickle
import matplotlib.pyplot as plt


class MyData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


# load dataset
dataset = pickle.load(open("data/demos.pkl", "rb"))
num_demos, demo_len, data_dim = np.shape(dataset)
train_demos = np.reshape(dataset, (len(dataset)*demo_len, data_dim))

# training parameters
EPOCH = 2000
LR = 1e-4
BATCH_SIZE = int(num_demos*demo_len/4.)
torch.manual_seed(0)

# model and optimizer
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_data = MyData(train_demos)
train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# main training loop
losses = []
for epoch in range(EPOCH + 1):
    for batch, x in enumerate(train_set):

        robot_states = x[:, :2]
        env_states = x[:, 2:6]
        actions = x[:, 6:]

        phi = model.state_encoder(env_states)
        predicted_actions = model.policy(robot_states, phi)
        loss = model.mse_func(predicted_actions, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    if epoch % 500 == 0:
        print(epoch, loss.item())

# plot training loss
plt.figure()
plt.plot(losses)
plt.show()

# save model
torch.save(model.state_dict(), "data/model.pt")
