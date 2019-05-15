import pickle
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def dump(Q, file_path):
  with open(file_path, 'wb') as file:
    pickle.dump(Q, file)


def load_dump(file_path):
  with open(file_path, 'rb') as file:
    return pickle.load(file)

def plot_value_function(V, title="Value Function"):
    min_x = 1
    max_x = V.shape[0]
    min_y = 1
    max_y = V.shape[1]

    x_range = np.arange(min_x, max_x)
    y_range = np.arange(min_y, max_y)
    X, Y = np.meshgrid(x_range, y_range)
    Z = V[X, Y]

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -50)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z, title)


def plot_error_vs_episode(sqrt_error, lambdas, num_train=1000000, num_episodes=1000,
                          title='MSE vs Episode Number', save_as_file=False):
    assert num_episodes != 0
    x_range = np.arange(0, num_train*num_episodes, num_episodes)
    
    assert len(sqrt_error) == len(lambdas)
    for e in sqrt_error:
    	print(len(e), len(x_range))
    	assert num_train == len(e)
    
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)
    ax.set_ylabel('MSE')
    ax.set_xlabel('Episode Number')

    for i in range(len(sqrt_error)-1, -1, -1):
    	ax.plot(x_range, sqrt_error[i], label='λ {}'.format(lambdas[i]))
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

    if save_as_file:
        plt.savefig(title)
   
    plt.show()

def plot_error_vs_lambda(sqrt_error, lambdas, title='MSE vs λ', save_as_file=False):

    assert len(sqrt_error) == len(lambdas)

    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)
    ax.set_ylabel('MSE')
    ax.set_xlabel('λ')
    y = [s[-1] for s in sqrt_error]
    ax.plot(lambdas, y)
    
    if save_as_file:
        plt.savefig(title)
    
    plt.show()


def mean_squared_error(true_Q, Agent, environment, lambdas, num_episodes=1000, num_train=10):
    errors = []
    for l in lambdas:
        error = []
        agent = Agent(environment, _lambda=l)
        for j in range(num_train):
            Q = agent.train(num_episodes)
            error.append(np.square(np.subtract(true_Q, Q)).mean())
        errors.append(error)
    return errors


