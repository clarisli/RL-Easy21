import numpy as np
from environment import *

class RLAgent:
	def __init__(self, environment, N0=100, discount_factor=1, _lambda=0.1):
		self.env = environment
		self.N0 = N0
		self.discount_factor = discount_factor
		self._lambda = _lambda

	def _init_tenor(self):
		return np.zeros((self.env.max_dealer_sum + 1, self.env.max_player_sum + 1, self.env.num_actions))

	def _get_alpha(self, s, a):
		return 1/self.returns_count[s.dealer_sum][s.player_sum][a]

	def _get_epsilon(self, s):
		epsilon = self.N0/(self.N0 + sum(self.returns_count[s.dealer_sum][s.player_sum]))
		return epsilon

	def get_value_function(self, Q):
		V = np.zeros([self.env.max_dealer_sum + 1, self.env.max_player_sum + 1])
		for d in range(1, self.env.max_dealer_sum + 1):
			for p in range(1, self.env.max_player_sum + 1):
				V[d][p] = self._get_value(State(d, p), Q)#np.max(Q[d][p])
		return V 

	def _get_value(self, s, Q):
		return np.max(Q[s.dealer_sum][s.player_sum])
