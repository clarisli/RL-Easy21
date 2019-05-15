
## Reinforcement Learning in Easy21

The goal is to apply reinforcement learning methods to a simplified Blackjack card game called Easy21:

* Write an environment that implements the game Easy21
* Apply Monte-Carlo control to Easy21
* Implement TD Learning Sarsa(λ) in Easy21
* Implement Linear Function Approximation in Easy21
* Understand the pros and cons of bootstrapping
* Understand the pros and cons of function approximation

[//]: # (Image References)

[image1]: ./images/mc_prediction_10000.png "MC Prediction 10000 episodes"
[image2]: ./images/mc_prediction_500000.png "MC Prediction 500000 episodes"
[image3]: ./images/gpi.png "GPI"
[image4]: ./images/mc_control.png "MC Control"
[image5]: ./images/sarsa_pseudo.png "Sarsa pseudo code"
[image6]: ./images/sarsa_mse_lambda.png "Sarsa MSE vs Lambda"
[image7]: ./images/sarsa_mse_num_episodes.png "Sarsa MSE vs Number of Episodes"
[image8]: ./images/sarsa_approx_mse_num_episodes.png "Sarsa Function Approx MSE vs Number of Episodes"

## Setup
Make sure you have following installed:

* Python 3.x
* Numpy
* Matplotlib

To train and plot the value functions, uncomment the code at the bottom of the following files first then run in terminal.

```
# Monte-Carlo Prediction
$ python mc_prediction.py

# Monte-Carlo Control
$ python mc_agent.py

# Sarsa(λ)
$ python td_sarsa_agent.py

# funcion approximation
$ python td_sarsa_approx_agent.py
```

To test the game environment implementation, run following in terminal:

```
$ python -m unittest -v test.py
```

## Game and Algorithms

### Task 1: Implement Environment

#### Game Rules

Easy21 is similar to the Blackjack example in Sutton and Barto 5.3 but with some differences.

*The object of the popular casino card game of blackjack is to obtain cards the sum of whose numerical values is as great as possible without exceeding 21. All face cards count as 10, and an ace can count either as 1 or as 11. We consider the version in which each player competes independently against the dealer. The game begins with two cards dealt to both dealer and player. One of the dealer’s cards is face up and the other is face down. If the player has 21 immediately (an ace and a 10-card), it is called a natural. He then wins unless the dealer also has a natural, in which case the game is a draw. If the player does not have a natural, then he can request additional cards, one by one (hits), until he either stops (sticks) or exceeds 21 (goes bust). If he goes bust, he loses; if he sticks, then it becomes the dealer’s turn. The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome—win, lose, or draw—is determined by whose final sum is closer to 21.*

The differences in rules are described below, and the code for the environment is in the file `environment.py`, note the following:

##### Deck and Card

* The game is played with an infinite deck of cards (i.e. cards are sampled with replacement)
* Each draw from the deck results in a value between 1 and 10 (uniformly distributed) with a colour of red (probability 1/3) or black (probability 2/3).
* There are no aces or picture (face) cards in this game

I did this in lines 9 to 28.

##### State

A state is the dealer’s first card 1–10 and the player’s sum 1–21. Beside this, a state might be terminal if the game is finished. I did this in lines 39 to 43.

##### Environment

Here's the core game implementation. 

At the start of the game both the player and the dealer draw one black card (fully observed). I did this in `init_state()` in lines 59 to 62.

At each turn, the player may either stick or hit. I implemented a function `step()` to takes in the current state `s` and action `a`, and returns the next state `s'` and reward `r`. I did this in lines 64 to 70.

If the player hits then she draws another card from the deck. After each hit, I will calculate the values of the player's cards, black ones are added and red ones are subtracted. I did this in `_draw_and_get_value()` in lines 59 to 62, and using this value to check if the player "goes bust",  i.e., if the player's sum exceeds 21 or less than 1,  then he loses the game. I did this in `_player_hit()` and `_is_bust()` in lines 72 to 80.

If the player sticks then the dealer starts taking turns. The dealer always sticks on any sum of 17 or greater, and hits otherwise. I did this in `_dealer_policy()` in lines 94 to 95. If the dealer goes bust, then the player wins; otherwise, the outcome – win (reward +1), lose (reward -1), or draw (reward 0) – is the player with the largest sum. I did this in `_dealer_play()` and `_get_reward_dealer_stick()` in lines 82 to 92 and lines 97 to 103. 


### Task 2: Apply Monte-Carlo Control

Characteristics of Monte-Carlo methods:

* Learn directly from episodes of experience.
* Model-free: no knowledge of MDP transitions / rewards 
* No bootstrapping: learns from complete episodes
* Value = mean return
* Only works with episodic MDPs. All episodes must terminate.

I created a base class for agent in `rl_agent.py`, it contains functions that are common among different types of agents.

#### Monte-Carlo Policy Evaluation

Before we can improve a policy, we have to know how to evaluate one. 

I created an agent with the same policy as the dealer - sticks only on 17 or greater - and used First-Visit Monte-Carlo Policy Evaluation to evaluate its value in `mc_prediction.py`. 

Here's the pseudo code:

```
for a large number of episodes:
    Generate an episode following π: S1,A1,R2,...,Sk
    for every step in the episode:
    	if it's the first step that state s is visited in an episode:
    		Increment counter: N(s) ← N(s) + 1
    		Increment total return: G_s(s) ← G_s(s) + Gt
       	Value is estimated by mean return V(s) = S(s)/N(s)
``` 

By law of large number, the value function will get close to the real value function as N(s) → ∞.

Below are the value functions of every possible state. Comparing with 10,000 episodes, after 500,000 episodes the value function is very well approximated:

![alt text][image1]
![alt text][image2]

#### Monte-Carlo Control

Now, we are ready to use policy evaluation in control. The goal of control is to find the best possible policy. 

Here's the pseudo code:

```
for a large number of episodes:
    Generate an episode following π: S1,A1,R2,...,Sk
    for every step t in the episode:
    	Increment counter: N(st, at) ← N(st, at) + 1
    	error ← Gt − Q(st, at)
    	Q(st, at) ← Q(st, at) + αt * error
```  

I used a a time-varying scalar step-size of `αt = 1/N(st, at)`, in this way I have a running mean that forgets old episodes. I did this in `_get_alpha()` lines 14 to 15 in `rl_agent.py`.

##### Generalized Policy Iteration

To achieve this, we apply the idea of **generalized policy iteration (GPI)**. We repeatedly alternate between two steps - evaluation and imporvement, and at the end both the policy and value function will reach optimality. In evaluation, as described in the section above, we approximate the value function for the current policy; in improvement, we improve the policy according to the value function. 

![alt text][image3]

##### ε-Greedy Exploration

In the step of improvement, an important concept to understand is the famous delimma in reinforcement learning: **exploration and exploitation**. This involves a fundamental choice between to exploit, i.e., make the best decision given current information, or to explore, try something new to gather more information. Though exploration may involve short-term sacrifices, it is necessary because there's the possibility that we haven't found the best action yet. If we only exploit current information and choose the best option from things we've tried (it's called taking a greedy action), we may be giving up the largest reward without even knowing about it.

Therefore, beside exploitation, we want to ensure continual exploration and all actions are tried:

* With probability 1 − ε choose the greedy action 
* With probability ε choose an action at random

I used an ε-Greedy strategy of `ε_t = N0/(N0 + N(s_t))`, the implementation is in `_get_epsilon()`, lines 17 to 19 in `rl_agent.py`.

I implemented Monte-Carlo control in `mc_agent.py`, and below is the optimal value function:

![alt text][image4]


### Task 3: Implement TD Learning Sarsa(λ)

Main differences between Temporal-Difference (TD) Learning and Monte-Carlo:

* TD **bootstraps**, i.e., it learns from incomplete episode. 
	* Monte-Carlo: update the value towards **actual return Gt**: `V(St)←V(St)+α(Gt −V(St))`
	* TD: update the value towards **estimate return R_t+1 +γV(S_t+1)**: `V(St)←V(St)+α(R_t+1 +γV(S_t+1)−V(St))`
* TD learns online, instead of wait until the end of every episode to update the policy, it updates at every time step.

I implemented the backward view algorithm, here's the pseudo code:

![alt text][image5]

##### Eligibility Traces

Eligibility trace is like a short-term memory. When we are in a state and take a certain action, the corresponding trace for this state-action pair bumps up and then begins to fade away. Learning will occur for that state-action pair before its trace falls back to zero. λ ∈ [0,1] is the trace-decay parameter, it determines the rate at which the trace falls.

```
E0(s,a) = 0
Et(s,a)=γλE_t−1(s,a) + 1(St =s,At =a)
```

I did this in `td_sarsa_agent.py`.

I ran the algorithm with parameter values λ ∈ {0, 0.1, 0.2, ..., 1}. Stopped each run after 1000 episodes and reported the mean-squared error over all states s and actions a. I used Monte-Carlo's value functions computed in the previous section as the true values with the estimated values Q(s, a) computed by Sarsa. 

Here's the plot of the mean squared error against λ:

![alt text][image6]

The plot of mean squared error against number of episodes, with λ ∈ {0,1}

![alt text][image7]

These results make sense because the values of λ produce a family of methods spanning a spectrum between Monte Carlo at one end (λ = 1) and one-step TD methods at the other (λ = 0). Observe that TD (λ = 0) performance better than Monte Carlo (λ = 1). It's because MC has high variance, i.e., it finds estimate that minimize the squared error on the observed return, where TD has low variance and it finds the estimates of the exact correct for the maximum-likelihood model of the Markov model. That said, TD is usually more efficient in Markov environments, and Easy21 is a Markov environment.

### Task 4: Implement Linear Function Approximation

Sometimes we want to scale up the model-free methods for prediction and control to solve large MDPs. To do this, we estimate value function with funciton approximation.

I implemetented a linear value function approximation Q(s, a) = φ(s, a)Tθ, with a constant exploration of ε = 0.05 and a constant step-size α of 0.01.

φ(s, a) is the binary feature vector with 3 * 6 * 2 = 36 features. Each binary feature
has a value of 1 iff (s, a) lies within the cuboid of state-space corresponding to that feature, and the action corresponding to that feature. The cuboids have the following overlapping intervals:

* dealer(s) = {[1, 4], [4, 7], [7, 10]}
* player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
* a = {hit, stick}

Here's the plot of mean squared error against number of episodes, with λ ∈ {0,1}

![alt text][image8]

Comparing with the previous plot for Sarsa(λ), function approximation seems to generalize faster than Sarsa(λ), its mean squared error drops down quickly.

## Discussion

#### What are the pros and cons of bootstrapping in Easy21?

TD bootstraps and Monte Carlo doesn't; TD has low variance while Monte Carlo has high variance. 

Pros:

* More efficient because can learn online at every step
* Low variance and can generalize better 
* Learn from incomplete sequence

Cons:

* More sensitive to initial value

#### Would you expect bootstrapping to help more in blackjack or Easy21? Why?

I expect bootstrapping help more in Easy21. Though Easy21 has less number of states than Blackjack - with a usable ace Blackjack comes with 200 states (the players current sum 12-21, the dealer's one showing card ace-10, and whether or not he holds an ace), and Easy21 only has 100 states in the absence of the usable ace. However, the randomness brought by card colors may make each episode takes longer to end. Without bootstrapping, learning only happen at the end of each episode, where with bootstrapping learning happens at each step within the episode.

#### What are the pros and cons of function approximation in Easy21?

The pros of function approximation are reduced learning time and space. Faster because with a function it generalize the learning to update fewer parameters. Easy21 has small number of states, so the space saving benefit might not be that obvious. The cons are that the features have to be selected appropriately to the task with prior domain knowledge, and it may be more difficult to manage and understand how the agent learns.

#### How would you modify the function approximator suggested in this section to get better results in Easy21?

A limitation of linear methods is that it doesn't take into account of any interactions between features, for example, we may regard a feature being good only in the presence of another feature. For example, in Blackjack, there are strategies like when player's hand is 14, 15, or 16 and the dealer's up card is 2-6, player should stand, for the rest of the case the player should hit. I could try use a polynomial function to capture this.

## References

* [Reinforcement Learning Course by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)