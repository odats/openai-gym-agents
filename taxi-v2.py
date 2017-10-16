import numpy as np
import time

import gym
env = gym.make('Taxi-v2')

print(env.observation_space.n)
print(env.action_space.n)

#-------------------------------------------------------------------------------
# Train
#-------------------------------------------------------------------------------
Q = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.618
gamma = 0.5
for episode in range(1,10001):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            Q[state,action] += alpha * (reward + gamma*np.max(Q[state2]) - Q[state,action])
            G += reward
            state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

#-------------------------------------------------------------------------------
# Play
#-------------------------------------------------------------------------------
avg_g = 0
r_list = []
for episode in range(1,100):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            G += reward
            state = state2

            #time.sleep(1)
            #env.render()
            #print('Reward:', G)
    print(G)
    avg_g += G
    r_list.append(G)
avg_g = avg_g / 100
print('avg G:', avg_g)
print(np.average(r_list))
