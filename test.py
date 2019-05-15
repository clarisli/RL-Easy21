import unittest
from environment import *


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.environment = Environment()

    def test_init_state(self):
        state = self.environment.init_state()
        self.assertTrue(1 <= state.player_sum <= 10, "Wrong value for player's first card")
        self.assertTrue(1 <= state.dealer_sum <= 10, "Wrong value for dealer's first card")

    def test_step_player_bust(self):
        state = State(player_sum=35, dealer_sum=1)
        state, reward = self.environment.step(state, Action.HIT)
        self.assertEqual(reward.value, -1, "Wrong reward value when player bust")

    def test_step_dealer_bust(self):
        state = State(player_sum=1, dealer_sum=23)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertEqual(reward.value, 1, "Wrong reward value when dealer bust")

    def test_step_dealer_stick(self):
        state = State(player_sum=1, dealer_sum=18)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertEqual(state.dealer_sum, 18, "Dealer did not stick on sum of 17 or greater")

    def test_step_dealer_hit(self):
        state = State(player_sum=1, dealer_sum=16)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertTrue(state.dealer_sum != 16, "Dealer did not hit on sum less than 17")

    def test_step_dealer_win(self):
        state = State(player_sum=1, dealer_sum=18)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertEqual(reward.value, -1, "Wrong reward when the dealer wins")

    def test_step_player_win(self):
        state = State(player_sum=20, dealer_sum=18)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertEqual(reward.value, 1, "Wrong reward when the player wins")

    def test_step_outcome_draw(self):
        state = State(player_sum=18, dealer_sum=18)
        state, reward = self.environment.step(state, Action.STICK)
        self.assertEqual(reward.value, 0, "Wrong reward when there's draw")



if __name__ == '__main__':
    unittest.main()