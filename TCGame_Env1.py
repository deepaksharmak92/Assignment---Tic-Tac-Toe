from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        win_pattern = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        
        i=curr_state.copy()
        if (i[0]+i[1]+i[2]==15) or (i[3]+i[4]+i[5]==15) or (i[6]+i[7]+i[8]==15) or (i[0]+i[3]+i[6]==15) or (i[1]+i[4]+i[7]==15) or (i[2]+i[5]+i[8]==15) or (i[0]+i[4]+i[8]==15) or (i[2]+i[4]+i[6]==15):
            output=True
        else:
            output=False
        return output

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        obj= curr_state.copy()
        if (curr_action in self.action_space(curr_state)[0]) or (curr_action in self.action_space(curr_state)[1]):
            obj[curr_action[0]] = curr_action[1]
        
        return obj


    def step(self, curr_state, curr_action):
        
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        obj1= curr_state.copy()
        intermediate_state = self.state_transition(obj1, curr_action)
        if self.is_terminal(intermediate_state)[1] == 'Resume':
            e_curr_action = random.choice(list(self.action_space(intermediate_state)[1]))
            env_state = self.state_transition(intermediate_state, e_curr_action)
            
            if self.is_terminal(env_state)[1] == 'Resume':
                return (env_state, -1, False)
            elif self.is_terminal(env_state)[1] == 'Tie':
                return (env_state, 0, True)
            elif self.is_terminal(env_state)[1] == 'Win':
                return (env_state, -10, True)
        
        elif self.is_terminal(intermediate_state)[1] == 'Tie':
            return (intermediate_state, 0, True)
        elif self.is_terminal(intermediate_state)[1] == 'Win':
            return (intermediate_state, 10, True)
                              

    def reset(self):
        return self.state
