# Exploring RL Approaches for Random Environment Navigation
### ECE 595RL

Authors:
- Rohan Rajesh
- Heneil Patel
- Tucker Dickson

## Project Objective
The primary objective of this project is to investigate the effectiveness of various reinforcement learning (RL) algorithms in training an agent to navigate unknown environments with minimal information. Using OpenAI Gymnasium’s Frozen Lake simulation as a testbed, we aim to simulate conditions in which an agent encounters and must navigate dynamically generated layouts of “safe” and “unsafe” areas without prior knowledge of the environment.

This scenario models real-world challenges, where autonomous agents—such as robots, drones, or autonomous vehicles—are required to navigate unfamiliar or dynamically changing spaces while avoiding hazards and optimizing their path towards a goal.

## Getting Started
### Project File Structure
```
├── README.md  
├── ActorCritic.py 
├── DQN.py 
├── PartiallyObservableFrozenLake.py 
└── QLearning.py
```
### Setup and Installation
1. Clone the repository
```
git clone https://github.com/tuckerdickson/ECE595-RL.git
cd ECE595-RL
```
2. Create a new conda environment from the provided `environment.yml` file
```
conda env create -f environment.yml
```
3. Activate the environment
```
conda activate ece595rl
```
4. If using requirements.txt
```
pip install -r requirements.txt
```
5. Run the script for the desired algorithm
```
python3 DQN.py
python3 ActorCritic.py dynamic
python3 QLearning.py
```

## Contact
For questions or collaborations, please reach out:
- Rohan Rajesh [rajesh17@purdue.edu]
- Heneil Patel [pate2199@purdue.edu​]
- Tucker Dickson [ndickso@purdue.edu]
