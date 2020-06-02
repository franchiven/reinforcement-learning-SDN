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
import random
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

# Starting state
initial_state = 0

# Create networkx graphs with weights
G=nx.Graph()
G.add_weighted_edges_from(switch_list_weighted)
G.add_edges_from(switch_list_unweighted)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

FinalOutput3 = {}

def ComputeShortestPaths():
    
    # For all starting and ending switches
    for j in range(largest_state + 1):
        print('Computing shortest paths table for switch %s' %(j))
        
        current_state = 0
        
        # Learning rate.
        gamma = 0.5
        # Exploration vs Exploitation rate.
        e = 1
        
        final_state = j
        
        # Highest numbered switch + 1 (as we start from switch 0).
        MATRIX_SIZE = largest_state + 1
        
        # Create matrix.
        R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
        R *= -1
        
        # Assign zeros to paths and 100 to final_state reaching point.
        for point in switch_list_unweighted:
            if point[1] == final_state:
                R[point] = 10
            else:
                R[point] = 0
        
            if point[0] == final_state:
                R[point[::-1]] = 10
            else:
                R[point[::-1]]= 0
        
        # Add final_state point round trip.
        R[final_state, final_state]= 10
        
        Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
        
        # What are the possible actions to take? i.e. next switch hop
        def available_actions(state):
            current_state_row = R[state, ]
            av_act = np.where(current_state_row >= 0)[1]
            return av_act
        
        available_act = available_actions(initial_state)  
        
        # Which action should be taken. Explore vs exploit.
        def sample_next_action(available_actions_range):
            # Best next action is the hop which has the smallest non-zero reward. (0 implies the two nodes are not connected).
            try:
                max_next_hop = np.where(Q[current_state, ] == np.min(Q[current_state, ][np.nonzero(Q[current_state, ])]))[1]
            except:
                max_next_hop = np.where(Q[current_state, ] == np.min(Q[current_state, ]))[1]
            
            # Explore vs Exploit. Initially expore all possible paths, then over time exploit the best paths.
            if (random.random() < e): 
                next_action = int(np.random.choice(available_act, 1))
            else:
                if (max_next_hop.shape[0] > 1):
                    next_action = int(np.random.choice(available_act, 1))
                else:
                    next_action = int(max_next_hop)     
            return next_action
        
        action = sample_next_action(available_act)
        
        # Find weights from network topology
        def determine_weights():
            weight = weights_dict.get((current_state, action)) or weights_dict.get((action, current_state))   
            
            if (weight == None):
                return(1)
            else:
                weight = 1 + (weight / 10000)
                return(weight)
                
        # Neccessary for the first loop
        weight = 1
        
        # Update Q-Matrix
        def update(current_state, action, gamma):      
            try:
                min_index = np.where(Q[action, ] == np.min(Q[action, ][np.nonzero(Q[action, ])]))[1]
            except:
                min_index = np.where(Q[action, ] == np.min(Q[action, ]))[1]
              
            if min_index.shape[0] > 1:
                min_index = int(np.random.choice(min_index, size = 1))
            else:
                min_index = int(min_index)
        
            min_value = Q[action, min_index]
            weighted_min_value = np.multiply(min_value, weight)
            
            # Current Q value.
            current_q = Q[current_state, action]
            
            # Reward going from current_state to next state, with the weights considered.
            current_q_min = weighted_min_value - current_q
        
            try:
                next_q_min = np.min(Q[action, ][np.nonzero(Q[action, ])])
            except:
                next_q_min = 0
        
            # Update Q value using version of Bellmans equation. 
            Q[current_state, action] = (R[current_state, action] * weight) + current_q + gamma*(current_q_min + next_q_min - current_q)       
        
            if (np.max(Q) > 0):      
                return(np.sum(Q/np.max(Q) * 100))
            else:
                return (0)
            
        update(initial_state, action, gamma)
        
        # Training for number of episodes
        scores = []
        for i in range(50000):
            current_state = np.random.randint(0, int(Q.shape[0]))
            available_act = available_actions(current_state)
            action = sample_next_action(available_act)
            weight = determine_weights()
            score = update(current_state,action,gamma)
            scores.append(score)
            
            # Reduce exploration and increase exploitation. 
            if (e > 0.1):
                e -= 0.0001  

        FinalOutput1 = {}
        FinalOutput2 = {}

        # Testing
        #Finds the shortest path to final_state for all initial states
        for k in range(largest_state + 1):  
            current_state = k
            InitialOutput = {}
            steps = [current_state]    
            while current_state != final_state:
                try:
                    next_step_index = np.where(Q[current_state, ] == np.min(Q[current_state, ][np.nonzero(Q[current_state, ])]))[1]
                except:
                    next_step_index = np.where(Q[current_state, ] == np.min(Q[current_state, ]))[1]
                    
                if next_step_index.shape[0] > 1:
                    next_step_index = int(np.random.choice(next_step_index, size = 1))
                else:
                    next_step_index = int(next_step_index)
                    
                steps.append(next_step_index)
                current_state = next_step_index   

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
    











