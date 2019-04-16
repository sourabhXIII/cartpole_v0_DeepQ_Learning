# @author sourabhxiii

import numpy as np
import gym
import random
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

DEBUG = False

MAX_EPISODE = 500
MAX_TIMESTEP = 200
GAMMA = 0.99
ALPHA = 0.65
EPSILON = 0.1
MIN_LEARNING_RATE = 0.1

# Environment
# Here we encapsulate the continuous environment as a discretized one
class CartPole:
    '''
    Initialize the object
    '''
    def __init__(self, num_buckets):
        
        ## initialize the "Cart-Pole" environment
        self.env = gym.make('CartPole-v0')

        # number of discrete states (bucket) per state dimension
        self.num_buckets = num_buckets  # a tuple (x, x', theta, theta')

        # number of actions in this environment, (left, right) in this case
        self.num_actions = self.env.action_space.n

        # initialize the Q table, for each state-action pair Q(s,a)
        # remember our Q table is [state<tuple>][action]
        self.Q = np.zeros(self.num_buckets + (self.num_actions, ))
        # self.Q = np.random.uniform(low = 0, high = 1.2, size = (self.num_buckets + (self.num_actions, )))

        # initialize update count, updt(s,a). this is needed for adaptive learning rate
        self.U = np.ones(self.num_buckets + (self.num_actions, ) + (1, ))

        # set learning rate and exploration rate
        self.alpha = ALPHA
        self.epsilon = EPSILON

        # set the state bounds
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]

    '''
    Here we mimic the gym environment step() function. Returns the same 
    observation, reward, done, info.
    Since the gym environment is continuous we first discretize it
    before returning the observation.
    '''
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # discretize obs
        state = self.discretize_obs(obs)
        return state, reward, done, info

    '''
    discretizes the continuous state
    it produces a tuple representing the state
    '''
    def discretize_obs(self, obs):
        # discretize obs
        state = [] # we want a tuple

        # for each state variable
        for i in range(len(obs)):
            idx = 0
            stepSize = (abs(self.state_bounds[i][1]) + abs(self.state_bounds[i][0])) / self.num_buckets[i]
            # Find bucket, put values on the outer size of the range in the last bucket
            while ((idx < self.num_buckets[i] - 1) and obs[i] > self.state_bounds[i][0] + stepSize * (idx + 1)):
                idx += 1
            state.append(idx)
        return tuple(state)

    '''
    returns the discretized state
    '''
    def get_state(self, s):
        return self.discretize_obs(s)

    '''
    returns the gym environment
    '''
    def get_gym_env(self):
        return self.env

    '''
    wrapper of the gym env's close()
    '''
    def close(self):
        self.env.env.close()

    def get_exploration_rate(self, t):
        # if t > 1000:
        #     return 0.1
        # else:
        #     return EPSILON
        return max(EPSILON, min(1, 1.0 - math.log10((t+1)/25)))
        # if (t + 1) % 100 == 0:
        #     self.epsilon = self.epsilon / (int((t+1)/100) * 2)
        #     return self.epsilon
        # else:
        #     return self.epsilon
        # return

    def get_learning_rate(self, s, a):
        # adaptive learning rate
        # return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
        alpha = ALPHA / self.U[s][a]
        self.U[s][a] += 0.005
        return alpha

# Class Environment end




# selecting from possible actions in epsilon greedy manner
def randomize_action(a, eps, env):
    # we'll use epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return env.action_space.sample()
    return

# Here we will play the game
if __name__ == '__main__':
        
    # create the environment
    env = CartPole((1,1,6,3))  
    Q = env.Q

    # keep track of how performance improves with episode
    score = []
    # keep track of the eps changes
    eps = []
    # keep track of the updates of Q(s)
    update_count = {}

    for episode in range(MAX_EPISODE):
        print('Episode: %d' %episode)
        t = 0
        lc = rc = 0
        explore_rate = env.get_exploration_rate(episode)
        eps.append(explore_rate)
        biggest_change = 0

        # inital state, from where the game starts
        state_0 = env.get_state(env.env.reset())
        # pick the BEST action for the state
        a = int(np.argmax(Q[state_0]))
        while True:
            # visualize the environment
            # env.env.render()
            # select an action with exploration
            a = randomize_action(a, explore_rate, env.env)
            if a == 0:
                lc += 1
            else:
                rc +=1
            # perform the selected action
            state_1, r, done, _ = env.step(a)

            # in Q-Learning we will use this max[a']{ Q(s',a')} in our update
            # even if we do not end up taking this action in the next step.
            # This makes Q-learning a off-policy method.
            # Point to note that SARSA does the opposite, thus is an on-policy method.
            max_q_s1a1 = np.amax(Q[state_1])

            # get learning rate
            alpha = env.get_learning_rate(state_0, a)

            # update Q(s, a)
            old_qsa = Q[state_0][a]
            Q[state_0][a] = Q[state_0][a] + alpha*(r + GAMMA*max_q_s1a1 - Q[state_0][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[state_0][a]))
            # keep track of which state was updated how many times
            update_count[state_0] = update_count.get(state_0, 0) + 1

            t += 1
            if DEBUG:
                print("t = %d" % t)
                print('Old State: %s' %str(state_0))
                print("Action: %d" % a)
                print("State: %s" % str(state_1))
                print("Reward: %f" % r)
                print('Q update %f' %np.abs(old_qsa - Q[state_0][a]))
                print("Best Q: %f" % max_q_s1a1)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % alpha)

            # update the state and action
            state_0 = state_1
            a = int(np.argmax(Q[state_1])) # best action for state_1

            if done or t >= MAX_TIMESTEP: # game over
                break
        score.append(t)
        print('     Ran for %d timestep' %t)
        print('     Actions taken: L %d, R %d' %(lc, rc))
        print('     Biggest change is Qsa %f' %biggest_change)


    print(sorted(update_count.items()))
    plt.figure(1)
    plt.subplot(311)
    plt.plot(score)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    
    plt.subplot(312)
    plt.plot(eps)
    plt.ylabel('Epsilon')
    plt.xlabel('Episode')
    

    plt.subplot(313)
    # x, y = zip(*sorted(update_count.items()))
    od = OrderedDict(sorted(update_count.items()))
    plt.bar(range(len(od)), od.values(), align='center')
    plt.xticks(range(len(od)), od.keys())
    plt.xlabel('States')
    plt.ylabel('#Updates')
    plt.show()



    # done with the game. close the environment
    env.close()