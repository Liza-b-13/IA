import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update the Q-table using the Q-learning algorithm.
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(sprime, a')) - Q(s, a))
    """
    best_next_action = np.argmax(Q[sprime])
    td_target = r + gamma * Q[sprime, best_next_action]
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error
    return Q

def epsilon_greedy(Q, s, epsilon):
    """
    Epsilon-greedy action selection.
    With probability epsilon, select a random action.
    Otherwise, select the action with the highest Q-value.
    """
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s]) 

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

            S = Sprime

            if done:
                break

        print("Episode #", e, ": reward =", r)
        rewards.append(r)

    print("Average reward =", np.mean(rewards))

    plt.plot(range(n_epochs), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards over time')
    plt.show()

    print("Training finished.\n")
    
    env.close()
