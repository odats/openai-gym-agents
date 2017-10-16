import numpy as np
import time

import gym
env = gym.make('FrozenLake-v0')
# 0,1,2,3 = left, down, right, up

print(env.observation_space.n)
print(env.action_space.n)

#-------------------------------------------------------------------------------
# Train
#-------------------------------------------------------------------------------

def get_q_table(alpha = 0.3, gamma=0.5, episodes = 10000):
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for episode in range(1, episodes):
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
            #time.sleep(1)
            #env.render()
            #print(Q)

            action = np.argmax(Q[state])
            #action = env.action_space.sample()
            state2, reward, done, info = env.step(action)
            if reward == 0:
                if done == True:
                    reward = -10
                else:
                    reward = -0.01
            else:
                reward = reward
            #print(info['prob'])
            #Q[state,action] += alpha * (reward + gamma*np.max(Q[state2]) - Q[state,action])

            Q[state,action] = (1-alpha)*Q[state,action] + alpha * (reward + gamma*np.max(Q[state2]))
            G += reward
            state = state2

        # if episode % 50 == 0:
        #     print('Episode {} Total Reward: {}'.format(episode,G))
        # if reward > 0:
        #     print("URA!!!!!!!!!!!!!!!!!!!", G)

    return Q


#-------------------------------------------------------------------------------
# Play
#-------------------------------------------------------------------------------

def test_q_table(Q, episodes = 1000):
    avg_g = 0
    for episode in range(1,episodes):
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
            action = np.argmax(Q[state])
            #action = env.action_space.sample()
            state2, reward, done, info = env.step(action)
            G += reward
            state = state2

            #time.sleep(1)
            #env.render()
            #print('Reward:', G)
        print(G)
        avg_g += G

    avg_g = avg_g / episodes

    return avg_g

super_training = False
super_avg = 0
super_alpha = 0
super_gamma = 0

if super_training:
    alpha = 0.1
    while alpha < 1:
        gamma = 0.1
        while gamma <= 1:
            Q = get_q_table(alpha, gamma)
            avg_g = test_q_table(Q)
            print("avg:", avg_g, alpha, gamma)

            if avg_g > super_avg:
                super_avg = avg_g
                super_alpha = alpha
                super_gamma = gamma
                print("New Super:", super_avg, super_alpha, super_gamma)

            gamma += 0.1
        alpha += 0.1

    print("Final:", super_avg, super_alpha, super_gamma)

Q = get_q_table()
avg_g = test_q_table(Q)
print("avg:", avg_g)
