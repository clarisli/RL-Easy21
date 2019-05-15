from rl_agent import *
from utils import *

class TDSarsaAgent(RLAgent):

	def __init__(self, environment, N0=100, discount_factor=1, _lambda=1):
		super().__init__(environment, N0, discount_factor, _lambda)
		self.Q = self._init_tenor()
		self.returns_count = self._init_tenor()

	def train(self, num_episodes=10):
		for e in range(num_episodes):
			E = self._init_tenor()
			s = self.env.init_state()
			a = self._policy(s)
			while not s.is_terminal:
				self.returns_count[s.dealer_sum][s.player_sum][a] += 1
				next_s, r = self.env.step(s, Action(a))
				next_a = self._policy(next_s)
				td_delta = self._td_delta(s, a, next_s, next_a, r)
				E[s.dealer_sum][s.player_sum][a] += 1
				self.Q += self._get_alpha(s, a)*td_delta*E
				E = self.discount_factor*self._lambda*E
				s = next_s
				a = next_a

			if e % 10 == 0:
				print("\rEpisode {}/{}.".format(e, num_episodes), end="")


		return self.Q

	def _policy(self, s):
		if s.is_terminal: return None
		num_actions = len(Action)
		epsilon = self._get_epsilon(s)
		state_actions = np.ones(num_actions, dtype=float) * epsilon/num_actions
		greedy_action = np.argmax(self.Q[s.dealer_sum][s.player_sum])
		state_actions[greedy_action] += (1.0 - epsilon)
		action = np.random.choice(np.arange(len(state_actions)), p=state_actions)
		return action


	def _td_delta(self, s, a, next_s, next_a, r):
		if next_s.is_terminal:
			return r.value - self.Q[s.dealer_sum][s.player_sum][a]
		else:
			return (r.value + self.discount_factor*self.Q[next_s.dealer_sum][next_s.player_sum][next_a]) - self.Q[s.dealer_sum][s.player_sum][a]



# easy21 = Environment()
# sarsa_agent = TDSarsaAgent(easy21, _lambda=1)
# Q = sarsa_agent.train(500000)
# V = sarsa_agent.get_value_function(Q)
# plot_value_function(V, title="Sarsa(λ) N0=100 λ=1")

# easy21 = Environment()
# mcQ = load_dump('mcQ.pickle')
# num_episodes = 1000
# num_train= 100
# lambdas = [float(x)/10 for x in range(11)]
# errors = mean_squared_error(mcQ, TDSarsaAgent, easy21, lambdas, num_episodes=num_episodes, num_train=num_train)
# plot_error_vs_lambda(errors, lambdas, title="Sarsa(λ)")

# easy21 = Environment()
# mcQ = load_dump('mcQ.pickle')
# num_episodes = 1000
# num_train= 100
# lambdas = [0, 1]
# errors = mean_squared_error(mcQ, TDSarsaAgent, easy21, lambdas, num_episodes=num_episodes, num_train=num_train)
# plot_error_vs_episode(errors, lambdas, num_train=num_train, num_episodes=num_episodes, title="Sarsa(λ)")

