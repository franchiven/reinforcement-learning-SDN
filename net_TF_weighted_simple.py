#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Francesco Venerandi
Based on a tutorial by The Beginner Programmer: http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
and a tutorial by Arthur Juliani: https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from operator import itemgetter

# Test network used to run the file without the SDN Mininet simulation
switch_list_weighted = [(9, 8, 600), (9, 11, 320), (9, 10, 730), 
              (9, 7, 565), (9, 6, 350), (8, 2, 450), 
              (8, 11, 820), (8, 5, 300), (8, 6, 400), 
              (8, 4, 1090), (3, 1, 760), (3, 6, 390), (3, 5, 210), 
              (3, 4, 660), (3, 2, 550), (2, 1, 1310), (2, 5, 390), 
              (1, 7, 740), (1, 4, 390), (7, 10, 320), (7, 6, 730), 
              (7, 4, 340), (6, 5, 220), (5, 11, 930), (4, 10, 660), 
              (11, 10, 820)]

# Make topology by seperating out the weights from the switches
switch_list_weighted_minus1 = []
for i in switch_list_weighted:   
    pair = ()
    for j in i:
        # If j > 100 its a weight and should not be reduced by 1.
        if (j < 100):        
            pair = pair + (j - 1,)
        else:
            pair = pair + (j,)
    switch_list_weighted_minus1.append(pair)
       
switch_list_unweighted = [x[:-1] for x in switch_list_weighted_minus1] 

weights_dict = {(a,b):c for a,b,c in switch_list_weighted_minus1}

#Finds the largest node to calculate size of matrix
max1 = max(switch_list_unweighted, key = itemgetter(0))[0] 
max2 = max(switch_list_unweighted, key = itemgetter(1))[1] 
largest_state = max(max1, max2)

initial_state = 0

#Create networkx graphs with weights
G=nx.Graph()
G.add_edges_from(switch_list_unweighted)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()
 
# Highest numbered switch + 1 (as we start from switch 0)
MATRIX_SIZE = largest_state + 1

# Create matrix x*y
R_weight = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R_weight *= -1

# Assign zeros to paths and 100 to final_state-reaching point
for point in switch_list_unweighted:
    if point[1] == largest_state:
        R_weight[point] = 100
    else:
        R_weight[point] = 0

    if point[0] == largest_state:
        R_weight[point[::-1]] = 100
    else:
        R_weight[point[::-1]]= 0
        
# Add final_state point round trip
R_weight[largest_state, largest_state]= 100

Q_weight = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# Normalised weights between 0 and 1
maxWeight = max(v for k, v in weights_dict.items() if v != 0)
minWeight = min(v for k, v in weights_dict.items() if v != 0)

# Makes the weighted matrix. 
for switch, weight in weights_dict.items():
    Q_weight[switch[0], switch[1]] = 1-(weight-minWeight)/(maxWeight-minWeight)
    Q_weight[switch[1], switch[0]] = 1-(weight-minWeight)/(maxWeight-minWeight)
    
Q_weight[largest_state, largest_state]= 1

R = np.add(R_weight, Q_weight)


np.set_printoptions(precision=1) #Formatting
print(R)

# Which actions can be taken?
def available_actions(state):
    current_state_row = R[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

#What should the next action be?
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range,1))
    return next_action

tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape = [1, MATRIX_SIZE], dtype = tf.float32)
# Store the possible actions to take
actions_placeholder = tf.placeholder(shape = [MATRIX_SIZE], dtype = tf.bool)
W = tf.Variable(tf.random_uniform([MATRIX_SIZE, MATRIX_SIZE], 0, 0.01)) #Outputs random values in uniform distibution shape [16,4]. Min value 0, max value 0.01
Qout = tf.matmul(inputs1, W) #Matrix multiplication
predict = tf.argmax(tf.boolean_mask(Qout, actions_placeholder, axis = 1), 1) #Max from the tensor? Returns where it is in the tensor [4]?

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1 ,MATRIX_SIZE], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout)) #Total loss value. Squared makes it +ve. e.g 6
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1) #Optimizer
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer() 

y = .99 #Gamma
e = 1
number_episodes = 50000
# Create lists to contain total rewards and steps per episode
stateList = [] 
rewardList = [] 
steps = []
maxRewardList = [0]
with tf.Session() as sess:
    sess.run(init)
    for i in range(number_episodes):
        print(i)
        
        rAll = 0 
        d = False
        current_state = 0
        j = 0 

        state_list = [0]
        while j < 99:

            j+=1
            possible_actions = available_actions(current_state)
            boolean_actions = [True if i in possible_actions else False for i in range(MATRIX_SIZE)]
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(MATRIX_SIZE)[current_state:current_state+1], 
                              actions_placeholder:boolean_actions}) 
            action = possible_actions[a[0]]
            
            # Random action                                                        
            if np.random.rand(1) < e:
                action = sample_next_action(possible_actions)
                
            #Get new state and reward from environment
            next_state = action
            reward = R[current_state, action]

            d = True if reward >= 100 else False
#            d = False # For graph

            
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(MATRIX_SIZE)[next_state:next_state+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)

            targetQ = allQ
            targetQ[0,action] = reward + y*maxQ1 
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(MATRIX_SIZE)[current_state:current_state+1],nextQ:targetQ})
            rAll -= reward 
            state_list.append(next_state)

            current_state = next_state
          
            if d == True:
                rAll += 2*reward
                #Reduce chance of random action as we train the model.
                if e > 0.1:
                    e = e - 1./((i/500) + 1000)
                else:
                    e = 0.1
                break
            
        rewardList.append(rAll)
stateList.append(state_list)
  
steps = stateList[0] 
print(steps)

plt.plot(rewardList)


fontTitle = {'family': 'serif',
        'color':  'darkorange',
        'weight': 'normal',
        'size': 14,
        }
fontText = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

plt.plot(rewardList)
plt.xlabel('Number of Episodes', fontdict = fontTitle)
plt.ylabel('Score', fontdict = fontTitle)
plt.title('Shortest Weighted Paths Q-Learning, TensorFlow', fontdict = fontTitle)
plt.show()

