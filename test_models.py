import torch
import pickle
import numpy as np
from model import MyModel
from get_data import generate_scenario, plot_state

from csv import writer
from copy import deepcopy
import matplotlib.pyplot as plt


def rollout_policy(my_model, scene):
    """

    :param my_model:
    :param start:
    :return:
    """

    # copy starting state
    current_state = scene.copy()

    xi_hat = []
    for _ in range(MAX_STEPS):

        xi_hat.append(current_state.copy())

        # current state
        robot_state = current_state[:2]
        env_state = current_state[2:6]

        # output action
        robot_action = my_model.policy(torch.FloatTensor([robot_state]),
                                       my_model.state_encoder(torch.FloatTensor([env_state])))

        # update current state
        current_state[:2] += robot_action.detach().flatten().numpy()

    return np.array(xi_hat)


MAX_STEPS = 10

# load dataset
dataset = pickle.load(open("data/demos.pkl", "rb"))

# load trained model
model = MyModel()
model.load_state_dict(torch.load('data/model.pt'))
model.eval()

# test rollout (final state error)
fse_result = []
for i in range(100):

    # generate new scenario
    scenario = generate_scenario()
    goal_state = scenario[2:4].copy()

    # simulate robot actions
    xi_model = rollout_policy(model, scenario)
    final_robot_state = xi_model[-1, :2]
    fse_model = np.linalg.norm(goal_state - final_robot_state)

    if i % 50 == 0:
        plt.figure()
        plot_state(scenario)
        plt.plot(xi_model[:, 0], xi_model[:, 1], 'm-')

    fse_result.append(fse_model)

# see plots
plt.show()

# print result
fse_result = np.mean(fse_result, axis=0)
print("Average final state error:", fse_result)
