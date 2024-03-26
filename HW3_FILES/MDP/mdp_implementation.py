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
            if (i, j) in mdp.terminal_states:
                U_init[i][j] = float(mdp.board[i][j])
            else:
                max_utility = None
                curr_state = (i, j)
                for action in mdp.actions.keys():
                    p = mdp.transition_function[action]
                    utility_list = []
                    for next_state in [mdp.step(curr_state, a) for a in mdp.actions.keys()]:
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
    valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if (i,j) not in mdp.terminal_states
                    or mdp.board[i][j] != 'WALL']
    for curr_state in valid_states:
        max_utility = None
        max_action = None
        for action in mdp.actions.keys():
            p = mdp.transition_function[action]
            utility_list = []
            for next_state in [mdp.step(curr_state, a) for a in mdp.actions.keys()]:
                utility_list.append(U[next_state[0]][next_state[1]])

            curr_utility = sum(np.array(p) * np.array(utility_list))

            if max_utility is None or max_utility < curr_utility:
                max_utility = curr_utility
                max_action = action

            i, j = curr_state
            policy[i][j] = max_action

    none_policy_state = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if (i, j) in mdp.terminal_states
                         or mdp.board[i][j] == 'WALL']
    for i, j in none_policy_state:
        policy[i][j] = None
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
        i, j = idx_to_state(s_idx, mdp)
        if ((i, j) in mdp.terminal_states) or (mdp.board[i][j] == 'WALL'):
            continue

        r_mat[s_idx] = float(mdp.board[i][j])

        for next_idx in range(num_states):
            p = 0
            next_s = idx_to_state(next_idx, mdp)

            for action_idx, action in enumerate(actions):
                if mdp.step((i,j), action) == next_s:
                    curr_policy = policy[i][j]
                    p += mdp.transition_function[curr_policy][action_idx]

            p_mat[s_idx, next_idx] = p

    x = np.linalg.inv(np.identity(num_states) + (-mdp.gamma) * p_mat)
    x = np.dot(x, r_mat)

    return x.reshape(mdp.num_row, mdp.num_col)
    # ========================


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(policy_init)
    changed = True
    while changed:
        utility = policy_evaluation(mdp, policy)
        changed = False
        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if (i, j) not in mdp.terminal_states or mdp.board[i][j] != 'WALL']
        for i, j in valid_states:
            max_utility, max_action = None, None
            for act in list(mdp.actions):
                probability = mdp.transition_function[act]
                list_utility = [p * utility[i][j] for p in probability]
                curr_utility = sum(list_utility)

                if max_utility is None or max_utility < curr_utility:
                    max_utility = curr_utility
                    max_action = act

            if policy[i][j] != max_action:
                policy[i][j] = max_action
                changed = True

    none_policy_state = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if (i, j) in mdp.terminal_states
                         or mdp.board[i][j] == 'WALL']
    for i, j in none_policy_state:
        policy[i][j] = None

    return policy
    # ========================
