import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):

    max_q_next = np.max(Q[sprime])
    Q[s][a] += alpha * (r + gamma * max_q_next - Q[s][a])
    
    return Q

def epsilon_greedy(Q, s, epsilon):

    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(len(Q[0])))
    else:
        action = np.argmax(Q[s])
    
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01

    gamma = 0.8 

    epsilon = 0.2

    n_epochs = 20 
    max_itr_per_epoch = 100
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )


        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))


    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
