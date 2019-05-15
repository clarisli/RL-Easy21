from enum import Enum
from copy import copy
import random

class Color(Enum):
	RED = 0
	BLACK = 1

class Deck:
	def draw(self, color=None):
		return Card(color)

class Card:
	def __init__(self, color=None):
		self.value = self._random_value()
		if color:
			self.color = color
		else:
			self.color = self._random_color()

	def _random_color(self):
		random_value = random.random()
		if random_value < 1/3.0:
			return Color.RED
		return Color.BLACK

	def _random_value(self):
		return random.randint(1,10)

class Action(Enum):
	HIT = 0
	STICK = 1

class Reward(Enum):
	WIN = 1
	LOSE = -1
	NONE = 0

class State:
	def __init__(self, dealer_sum=0, player_sum=0, is_terminal=False):
		self.dealer_sum = dealer_sum
		self.player_sum = player_sum
		self.is_terminal = is_terminal

class Environment:

	def __init__(self):
		self.deck = Deck()
		self.max_dealer_sum = 10
		self.max_player_sum = 21
		self.num_actions = len(Action)

	def init_state(self):
		self.done = False
		dealer_value = self._draw_and_get_value(Color.BLACK)
		player_value = self._draw_and_get_value(Color.BLACK)
		return State(dealer_value, player_value)

	def _draw_and_get_value(self, color=None):
		card = self.deck.draw(color)
		multiplier = 1 if card.color == Color.BLACK else -1
		return multiplier * card.value

	def step(self, state, action):
		next_state, reward = None, Reward.NONE
		if action == Action.HIT:
			next_state, reward = self._player_hit(state)
		else:
			next_state, reward = self._dealer_play(state)
		return next_state, reward

	def _player_hit(self, state):
		next_state = copy(state)
		next_state.player_sum += self._draw_and_get_value()
		next_state.is_terminal = self._is_bust(next_state.player_sum)
		reward = Reward.LOSE if next_state.is_terminal else Reward.NONE
		return next_state, reward

	def _is_bust(self, cards_sum):
		return cards_sum < 1 or cards_sum > 21

	def _dealer_play(self, state):
		next_state = copy(state)
		action = None
		while not next_state.is_terminal and action != Action.STICK:
			action = self._dealer_next_action(next_state.dealer_sum)
			if action == Action.HIT:
				next_state.dealer_sum += self._draw_and_get_value()
			next_state.is_terminal = self._is_bust(next_state.dealer_sum)	
		reward = Reward.WIN if next_state.is_terminal else self._get_reward_dealer_stick(next_state)
		next_state.is_terminal = True
		return next_state, reward
	
	def _dealer_next_action(self, dealer_sum):
		return Action.HIT if dealer_sum < 17 else Action.STICK

	def _get_reward_dealer_stick(self, state):
		if state.player_sum > state.dealer_sum:
			return Reward.WIN
		elif state.player_sum < state.dealer_sum:
			return Reward.LOSE
		else:
			return Reward.NONE


			
