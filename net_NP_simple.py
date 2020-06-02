#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Francesco Venerandi
Based on a tutorial by The Beginner Programmer: http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
and a tutorial by Arthur Juliani: https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb
'''
import numpy as np
import pylab as plt
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


# Finds the largest node to calculate size of matrix
max1 = max(switch_list_unweighted, key = itemgetter(0))[0] 
max2 = max(switch_list_unweighted, key = itemgetter(1))[1] 
largest_state = max(max1, max2)
 
# What is the starting state?
initial_state = 0

# Create networkx graphs with weights
G=nx.Graph()
G.add_edges_from(switch_list_unweighted)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

FinalOutput3 = {}

def ComputeShortestPaths():
    
    # Loop to find all start to finish shortest paths
    for j in range(largest_state + 1):
        print('Computing shortest paths table for switch %s' %(j))
                
        # Learning rate
        gamma = 0.8

        final_state = j

        #Highest numbered switch + 1 (as we start from switch 0)
        MATRIX_SIZE = largest_state + 1
        
        # Create R-Matrix
        R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
        R *= -1
        
        # Assign zeros to paths and 100 to final_state reaching point.
        for point in switch_list_unweighted:
            if point[1] == final_state:
                R[point] = 100
            else:
                R[point] = 0
        
            if point[0] == final_state:
                R[point[::-1]] = 100
            else:
                # reverse of point
                R[point[::-1]]= 0
        
        # Add final_state point round trip
        R[final_state, final_state]= 100
        
        Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
                
        # What are the possible actions to take? i.e. next switch hop
        def available_actions(state):
            current_state_row = R[state, ]
            av_act = np.where(current_state_row >= 0)[1]
            return av_act
        
        available_act = available_actions(initial_state)  #From (node) where can it go?
        
        # Which action should be taken?
        def sample_next_action(available_actions_range):
            next_action = int(np.random.choice(available_act,1))
            return next_action
        
        action = sample_next_action(available_act) #Pick a random way. Can go the way it came
        
        # Update Q-Matrix
        def update(current_state, action, gamma):
            
            max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
          
            if max_index.shape[0] > 1:
                max_index = int(np.random.choice(max_index, size = 1))
            else:
                max_index = int(max_index)
              
            max_value = Q[action, max_index]
          
            # Q-equation
            Q[current_state, action] = R[current_state, action] + gamma * max_value
              
            if (np.max(Q) > 0):
                return(np.sum(Q/np.max(Q)*100))
            else:
                return (0)
            
        update(initial_state, action, gamma)
        
        # Training for number of episodes
        scores = []
        for i in range(1000):
            current_state = np.random.randint(0, int(Q.shape[0]))
            available_act = available_actions(current_state)
            action = sample_next_action(available_act)
            score = update(current_state,action,gamma)
            scores.append(score)
        
        FinalOutput1 = {}
        FinalOutput2 = {}

        # Testing
        #Finds the shortest path to final_state for all initial states
        for k in range(largest_state + 1):  
            current_state = k
            InitialOutput = {}
            steps = [current_state]    
            while current_state != final_state:
            
                next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
                if next_step_index.shape[0] > 1:
                    next_step_index = int(np.random.choice(next_step_index, size = 1))
                else:
                    next_step_index = int(next_step_index)
                
                steps.append(next_step_index)
                current_state = next_step_index
            np.set_printoptions(formatter={'float': lambda Q: "{0:0.0f}".format(Q)})

            # Reverse steps and add 1 to each in order to start from switch 1, not switch 0.
            # Places it into the 'S0' format to be read
            steps.reverse()
            steps1 = []
            for i in steps:
                i += 1
                steps1.append(u'S%s' %(i))

            InitialOutput.update({steps1[-1]: steps1})
            FinalOutput1.update(InitialOutput)
                  
          
        FinalOutput2.update({'S%s' %(j+1):FinalOutput1})

        FinalOutput3.update(FinalOutput2)
        
    FinalOutput3.update(FinalOutput2)
    print(FinalOutput3)        
    return FinalOutput3

if __name__ == '__main__':
    spaths = ComputeShortestPaths()
    











