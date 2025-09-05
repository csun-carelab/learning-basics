import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def generate_scenario():
    """
    Assume a 10 by 10 map.
    Samples a random robot position (x_robot, y_robot) in the map.
    Samples a random target position (x_goal, y_goal) in the map.
    Samples a distracting object position (x_obj, y_obj) in the map.

    :return: a vector of [x_robot, y_robot, x_goal, y_goal, x_obj, y_obj]
    """

    robot = 10*np.random.rand(2)
    goal = 10*np.random.rand(2)
    object = 10*np.random.rand(2)

    return np.concatenate([robot, goal, object])


def generate_trajectory(scene):
    """
    Move robot in a straight line to the goal. Ignore distracting object.
    :param scene: starting state of the scenario
    :return: sequence of state-action pairs
    """

    # copy starting state
    current_state = scene.copy()

    xi = []
    for _ in range(MAX_STEPS):

        # current state
        robot_state = current_state[:2]
        goal_state = current_state[2:4]

        # calculate robot action (move in a straight line towards goal)
        robot_action = goal_state - robot_state

        # scale robot action to be less than 1 in length
        if np.linalg.norm(robot_action) > 1.0:
            robot_action /= np.linalg.norm(robot_action)

        # add some noise to robot action (simulate real-world error)
        robot_action += np.random.rand(2) - 0.5

        # save data point [curent_state, action]
        data_point = np.concatenate([current_state, robot_action])
        xi.append(data_point.copy())

        # update current state (add robot action to robot state)
        current_state[:2] += robot_action

    return np.array(xi)


def plot_state(state):
    plt.plot(state[0], state[1], 'ro')
    plt.plot(state[2], state[3], 'go')
    plt.plot(state[4], state[5], 'k*')
    plt.xlim([-1, 11])
    plt.ylim([-1, 11])


MAX_STEPS = 10


if __name__ == "__main__":

    # read the number of trajectories specified at input (default is 50)
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=int, default=50)
    args = parser.parse_args()

    # generate trajectories
    demos = []
    for n_traj in range(args.trajectories):

        # sample start and goal state
        scenario = generate_scenario()

        # trajectory
        traj = generate_trajectory(scenario)

        # save
        demos.append(traj.copy())

    # plot demos (uncomment the code below to plot)
    # for demo in demos:
    #     plt.figure()
    #     plot_state(demo[0, :10])
    #     plt.plot(demo[:, 0], demo[:, 1])
    # plt.show()

    # save data
    pickle.dump(demos, open("data/demos.pkl", "wb"))

    print("Generated", len(demos), "trajectories.")
