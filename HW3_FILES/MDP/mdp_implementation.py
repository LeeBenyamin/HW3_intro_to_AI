import math
from copy import deepcopy
import numpy as np

# Returns a list of all valid states (i.e. not 'WALL') from the mdp.
# Includes terminal states
def get_states(mdp):
    return [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.

    # ====== YOUR CODE: ======
    gamma = mdp.gamma
    while True:
        delta = 0
        U = deepcopy(U_init)

        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
        for i, j in valid_states:
            if (i, j) in mdp.terminal_states:
                U_init[i][j] = float(mdp.board[i][j])
            else:
                max_utility = None
                curr_state = (i, j)
                for action in mdp.actions.keys():
                    p = mdp.transition_function[action]
                    R = []
                    for action in mdp.actions.keys():
                        next_state = mdp.step(curr_state, action)
                        R.append(U[next_state[0]][next_state[1]])
                    curr_utility = sum((p * r for p, r in zip(p, R)))

                    if max_utility is None or max_utility < curr_utility:
                        max_utility = curr_utility

                U_init[i][j] = float(mdp.board[i][j]) + gamma * max_utility

            if abs(U_init[i][j] - U[i][j]) > delta:
                delta = U_init[i][j] - U[i][j]

        if delta < ((epsilon * (1 - mdp.gamma)) / mdp.gamma):
            return U

    # ========================


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(U)
    for i, j in get_states(mdp):
        if not (i, j) in mdp.terminal_states:
            max_utility = None
            max_action = None
            for action in mdp.actions.keys():
                curr_state = (i, j)
                p = mdp.transition_function[action]
                R = []
                for action in mdp.actions.keys():
                    next_state = mdp.step(curr_state, action)
                    R.append(U[next_state[0]][next_state[1]])

                curr_utility = sum((p * r for p, r in zip(p, R)))

                if max_utility is None or max_utility < curr_utility:
                    max_utility = curr_utility
                    max_action = action

            policy[i][j] = max_action
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
