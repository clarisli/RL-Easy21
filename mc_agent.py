from rl_agent import *
from utils import *


class MCAgent(RLAgent):

	def __init__(self, environment, N0=100, discount_factor=1, _lambda=0.1):
		super().__init__(environment, N0, discount_factor, _lambda)
		self.Q = self._init_tenor()
		self.returns_count = self._init_tenor()

	def train(self, num_episodes=10):
		for e in range(num_episodes):
			episode = self._generate_episode()
			for j, (s, a, _) in enumerate(episode):
				self.returns_count[s.dealer_sum][s.player_sum][a] += 1
				G = sum([x[2].value*(self.discount_factor**i) for i,x in enumerate(episode[j:])])
				mc_delta = G - self.Q[s.dealer_sum][s.player_sum][a]
				self.Q[s.dealer_sum][s.player_sum][a] += self._get_alpha(s, a) * mc_delta

			if e % 1000 == 0:
				print("\rEpisode {}/{}.".format(e, num_episodes), end="")

		return self.Q

	def _generate_episode(self):
		episode = []
		state = self.env.init_state()
		while not state.is_terminal:
			action = self._policy(state)
			next_state, reward = self.env.step(state, Action(action))
			episode.append((state, action, reward))
			state = next_state
		return episode

	def _policy(self, s):
		epsilon = self._get_epsilon(s)
		state_actions = np.ones(self.env.num_actions, dtype=float) * epsilon/self.env.num_actions
		greedy_action = np.argmax(self.Q[s.dealer_sum][s.player_sum])
		state_actions[greedy_action] += (1.0 - epsilon)
		action = np.random.choice(np.arange(len(state_actions)), p=state_actions)
		return action


# easy21 = Environment()
# mc_agent = MCAgent(easy21)
# Q = mc_agent.train(500000)
# dump(Q, 'mcQ.pickle')
# V = mc_agent.get_value_function(Q)
# plot_value_function(V, title="MC Control Value function N0=100")
