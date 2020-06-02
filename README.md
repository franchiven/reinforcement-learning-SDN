# reinforcement-learning-SDN
**University of Bristol Masters Thesis, 2019**

**Read the full thesis in /Thesis/FrancescoVenerandiThesis.pdf**

This project investigated how Reinforcement Learning could be applied to different parts of a software defined network. The code included in this repository specifically looks at how Q-learning could be used for path finding in a SDN. 

## Table of Contents
- [Introduction](#introduction)
- [Run using Mininet](#run-using-mininet)

## Introduction
The primary aim of this project was to investigate how feasible a reinforcement learning, specifically a Q-learning algorithm, would be in finding the shortest paths within a network topology.

Below shows an example network (pan european network topology) which was used in the TensorFlow Q-learning.ipynb.

<img src = "/Thesis/TopologyUse.png" width="650">

A nerual network trained on this topology managed to find the shortest path (by weight) from 0-10 (0 3 6 8 10) within a few thousand episodes (~5 seconds).

<img src = "/Thesis/short_paths.png" width="650">

**TensorFlow Q-learning.ipynb can be run on its own (put in your desired network topology) or in conjunction with a mininet virtual machine.**

## Run using Mininet
To run the entire SDN architecture:

1. Install and open a virtual machine (VM) running Mininet.
2. Install Ryu controller inside the virtual machine.
3. Install all the necessary packages.
4. Inside the VM open the folder: ryu/ryu/app. (This can be done with the command cd ryu/ryu/app once the machine has just started).
5. Copy the following files in this repo into the new folder:
    - l2DestForwardStaticRyuNS.py
    - ShortestPathBridgeNet_NP.py
    - TensorFlow Q-learning.ipynb
    - NetRunnerNS.py
    - PanEuroNet.json
    - g_switchesOut.pickle
5. Place your own IP address in NetRunnerNs line 82 and your own network topology (if desired) in NetRunnerNS line 83 and in ShortestPathBridgeNet_NP line 139.
6. Follow the instructions in chapter 5 of the thesis to simulate the network.
7. These are the two main commands to start the network:
    - sudo python NetRunnerNS.py -f PanEuroNet.json -ip 192.168.56.101 
    - python l2DestForwardStaticRyuNS.py --netfile=PanEuroNet.json
