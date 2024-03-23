import math
from copy import deepcopy
import numpy as np

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
            if (i, j) in mdp.terminal_states or mdp.board[i][j] == 'WALL':
                U_init[i][j] = float(mdp.board[i][j])
            else:
                max_utility = None
                curr_state = (i, j)
                for action in mdp.actions.keys():
                    p = mdp.transition_function[action]
                    utility_list = []
                    for action in mdp.actions.keys():
                        next_state = mdp.step(curr_state, action)
                        utility_list.append(U[next_state[0]][next_state[1]])
                    curr_utility = sum(np.array(p) * np.array(utility_list))

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
    valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col)]
    for i, j in valid_states:
        if (i, j) in mdp.terminal_states or mdp.board[i][j] == 'WALL':
            policy[i][j] = None
            continue
        else:
            max_utility = None
            max_action = None
            for action in mdp.actions.keys():
                curr_state = (i, j)
                p = mdp.transition_function[action]
                utility_list = []
                for action in mdp.actions.keys():
                    next_state = mdp.step(curr_state, action)
                    utility_list.append(U[next_state[0]][next_state[1]])
                curr_utility = sum(np.array(p) * np.array(utility_list))

                if max_utility is None or max_utility < curr_utility:
                    max_utility = curr_utility
                    max_action = action

            policy[i][j] = max_action
    return policy
    # ========================

def idx_to_state(idx, mdp):
    row = idx // mdp.num_col
    col = idx % mdp.num_col
    return row, col


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    # ====== YOUR CODE: ======
    num_states = mdp.num_row * mdp.num_col
    actions = list(mdp.actions)

    p_mat = np.zeros((num_states, num_states))
    r_mat = np.zeros(num_states)
    for s_idx in range(num_states):
        s = idx_to_state(s_idx, mdp)

        if (s in mdp.terminal_states) or (mdp.board[s[0]][s[1]] == 'WALL'):
            continue

        r_mat[s_idx] = float(mdp.board[s[0]][s[1]])

        for next_idx in range(num_states):
            p = 0
            next_s = idx_to_state(next_idx, mdp)
            for action_idx, action in enumerate(actions):
                if mdp.step(s, action) == next_s:
                    curr_policy = policy[s[0]][s[1]]
                    p += mdp.transition_function[curr_policy][action_idx]
            p_mat[s_idx, next_idx] = p

    x = np.linalg.inv(np.identity(num_states) + (-mdp.gamma) * p_mat)
    x = np.dot(x, r_mat)

    return x.reshape(mdp.num_row, mdp.num_col)
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
