from rl_agent import *
from utils import *

class TDSarsaApproxAgent(RLAgent):

	def __init__(self, environment, N0=100, discount_factor=1, _lambda=1):
		super().__init__(environment, N0, discount_factor, _lambda)
		self.dealer_intervals = {(1,4), (4,7), (7,10)}
		self.player_intervals = {(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)}
		self.num_features = len(self.dealer_intervals)*len(self.player_intervals)*self.env.num_actions
		self.theta = np.random.randn(self.num_features) * 0.1

	def _init_feature_tenor(self):
		return np.zeros(self.num_features)

	def train(self, num_episodes=10):
		for e in range(num_episodes):
			E = self._init_feature_tenor()
			s = self.env.init_state()
			a = self._policy(s)
			while not s.is_terminal:
				next_s, r = self.env.step(s, Action(a))
				next_a = self._policy(next_s)
				phi = self._get_phi(s, a)
				q = self._get_Q(phi)
				td_delta = self._td_delta(next_s, next_a, r, q)
				E += phi
				self.theta += (self._get_alpha(s, a)*td_delta*E)
				E *= (self.discount_factor*self._lambda)
				s = next_s
				a = next_a

			if e % 10 == 0:
				print("\rEpisode {}/{}.".format(e, num_episodes), end="")
		
		self.Q = self._get_all_Q()
		return self.Q

	def _policy(self, s):
		if s.is_terminal: return None
		if random.random() < self._get_epsilon(s):
			return self._get_random_action()
		else:
			return self._get_greedy_action(s)

	def _get_epsilon(self, s):
		return 0.05

	def _get_random_action(self):
		return Action.HIT.value if random.random() <= 0.5 else Action.STICK.value

	def _get_greedy_action(self, s):
		return np.argmax([self._get_Q(self._get_phi(s, a.value)) for a in Action])

	def _get_phi(self, s, a):
		dealer_nonzero_features = np.where([i[0] <= s.dealer_sum <= i[1] for i in self.dealer_intervals])
		player_nonzero_features = np.where([i[0] <= s.player_sum <= i[1] for i in self.player_intervals])
		phi = np.zeros((len(self.dealer_intervals), len(self.player_intervals), self.env.num_actions), dtype=np.int)
		phi[dealer_nonzero_features, player_nonzero_features, a] = 1
		return phi.flatten()

	def _get_Q(self, phi):
		return np.dot(phi, self.theta)
	
	def _td_delta(self, next_s, next_a, r, q):
		if next_s.is_terminal:
			return r.value - q
		else:
			next_q = self._get_Q(self._get_phi(next_s, next_a))
			return (r.value + self.discount_factor*next_q) - q

	def _get_alpha(self, s, a):
		return 0.01

	def _get_all_Q(self):
		all_q = self._init_tenor()
		for d in range(1, self.env.max_dealer_sum + 1):
			for p in range(1, self.env.max_player_sum + 1):
				for a in Action:
					s = State(d, p)
					phi = self._get_phi(s, a.value)
					all_q[d, p, a.value] = self._get_Q(phi)
		return all_q

# 1. Plot value function
# easy21 = Environment()
# sarsa_approx_agent = TDSarsaApproxAgent(easy21)
# Q = sarsa_approx_agent.train(1000)
# V = sarsa_approx_agent.get_value_function(Q)
# plot_value_function(V)

# 2. Plot error vs episode
# easy21 = Environment()
# mcQ = load_dump('mcQ.pickle')
# num_episodes = 1000
# num_train=100
# lambdas = [0,1]
# errors = mean_squared_error(mcQ, TDSarsaApproxAgent, easy21, lambdas, num_episodes=num_episodes, num_train=num_train)
# plot_error_vs_episode(errors, lambdas, num_episodes=num_episodes, num_train=num_train, title="Sarsa(λ) Function Approximation")

