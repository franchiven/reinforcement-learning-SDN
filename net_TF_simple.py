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
switch_list_unweighted1 = [(9, 8), (9, 11), (9, 10), 
              (9, 7), (9, 6), (8, 2), 
              (8, 11), (8, 5), (8, 6), 
              (8, 4), (3, 1), (3, 6), (3, 5), 
              (3, 4), (3, 2), (2, 1), (2, 5), 
              (1, 7), (1, 4), (7, 10), (7, 6), 
              (7, 4), (6, 5), (5, 11), (4, 10), 
              (11, 10)]

# Make topology by seperating out the weights from the switches. NOT USED
switch_list_unweighted = []
for i in switch_list_unweighted1:   
    pair = ()
    for j in i:
        pair = pair + (j - 1,) 

    switch_list_unweighted.append(pair)

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

#Highest numbered switch + 1 (as we start from switch 0)
MATRIX_SIZE = largest_state + 1

# Create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1

# Assign zeros to paths and 100 to final_state-reaching point
for point in switch_list_unweighted:
    print(point)
    if point[1] == largest_state:
        R[point] = 100
    else:
        R[point] = 0

    if point[0] == largest_state:
        R[point[::-1]] = 100
    else:
        # reverse of point
        R[point[::-1]]= 0

# Add final_state point round trip
R[largest_state, largest_state]= 100

def available_actions(state):
    current_state_row = R[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

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
predict = tf.argmax(tf.boolean_mask(Qout, actions_placeholder, axis = 1), 1) 

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1 ,MATRIX_SIZE], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout)) #Total loss value. Squared makes it +ve. e.g 6
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1) #Optimizer
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer() 

y = .99 
e = 1
number_episodes = 2500
# Create lists to contain total rewards and steps per episode
stateList = [] 
rewardList = [] 
steps = []
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
            # Which actions can be taken?
            possible_actions = available_actions(current_state)
            boolean_actions = [True if i in possible_actions else False for i in range(MATRIX_SIZE)]
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(MATRIX_SIZE)[current_state:current_state+1], 
                              actions_placeholder:boolean_actions}) 
            action = possible_actions[a[0]]
             
            # Take random action                                                       
            if np.random.rand(1) < e:
                action = sample_next_action(possible_actions)
                
            #Get new state and reward from environment
            next_state = action
            reward = R[current_state, action]
            d = True if reward == 100 else False
#            d = False #Set this to false to produce graph. Will take a while!
            
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(MATRIX_SIZE)[next_state:next_state+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,action] = reward + y*maxQ1
            
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(MATRIX_SIZE)[current_state:current_state+1],nextQ:targetQ})
            rAll += reward
            state_list.append(next_state)

            current_state = next_state

            if d == True:
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


plt.xlabel('Number of Episodes', fontdict = fontTitle)
plt.ylabel('Score', fontdict = fontTitle)
plt.title('Shortest Paths Q-Learning, TensorFlow', fontdict = fontTitle)
plt.show()
