from rl_agent import *
from utils import *
import sys

class MCPrediction(RLAgent):

	def __init__(self, environment, N0=100, discount_factor=1, _lambda=0.1):
		super().__init__(environment, N0, discount_factor, _lambda)
		self.returns_sum = self._init_tenor()
		self.returns_count = self._init_tenor()
		self.V = self._init_tenor()

	def _init_tenor(self):
		return np.zeros((self.env.max_dealer_sum + 1, self.env.max_player_sum + 1))

	def train(self, num_episodes=1000):
		for e in range(num_episodes):
			episode = self._generate_episode()
			for i, (s, _, _) in enumerate(episode):
				first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s)
				if i == first_occurence_idx:
					self.returns_count[s.dealer_sum][s.player_sum] += 1
					G = sum([x[2].value*(self.discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
					self.returns_sum[s.dealer_sum][s.player_sum] += G
					self.V[s.dealer_sum][s.player_sum] = self.returns_sum[s.dealer_sum][s.player_sum] / self.returns_count[s.dealer_sum][s.player_sum]
			
			if e % 1000 == 0:
				print("\rEpisode {}/{}.".format(e, num_episodes), end="")

		return self.V


	def _generate_episode(self):
		episode = []
		state = self.env.init_state()
		while not state.is_terminal:
			action = self._policy(state)
			next_state, reward = self.env.step(state, Action(action))
			episode.append((state, action, reward))
			state = next_state
		return episode

	def _policy(self, state):
		if state.player_sum >= 17:
			return Action.STICK.value
		else:
			return Action.HIT.value

# easy21 = Environment()
# mc = MCPrediction(easy21)
# V = mc.train(500000)
# plot_value_function(V, title="500,000 Episodes, MC Prediction")

# easy21 = Environment()
# mc = MCPrediction(easy21)
# V = mc.train(10000)
# plot_value_function(V, title="10,000 Episodes, MC Prediction")
