# custom-reinforcement-grid-environment
## A custom grid environment for a Q learning algorithm to match numbers on the grid coded in Python.
Can be run in  a notebook with python or in an environment that supports python.
### About
This project was carried out to gain experience and a deeper understanding in the area of reinforcement learning by using a Q learning algorithm, and creating a custom environment supported by OpenAi's Gymnasium library gave me greater exposure to a library that has great importance in this area of machine learning. The agent is a number randomly generated between 1-4, and its goal is to reach the corner of the grid environment with the same number in the fewest moves without going to any of the other corners.  Using a Jupyter notebook allowed for greater analysis of the environment and algorithm, in which the agent was rewarded positively for reaching the position of the correct corner.

Example of the grid environment:
```
1	-	-	-	2	

-	-	-	-	-	

-	-	4	-	-	

-	-	-	-	-	

3	-	-	-	4	
```
# Evaluation
![Total Reward over Episodes](https://github.com/user-attachments/assets/855561ec-8e3b-4265-b697-60d6badd5083)
The graph plotting the total reward against the current episode shows a positive trend, proving that the Q learning algorithm was able to improve its performance over time by having more successful outcomes in fewer moves. To improve the model, more episodes could be carried out to see if this trend continues, and the alpha, gamma and epsilon parameters of the algorithm could be further experimented with to see if a quicker and greater improvement in performance could be achieved.
