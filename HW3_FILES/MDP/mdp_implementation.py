import math
import sys
from copy import deepcopy
import numpy as np
import mdp # TODO: remove this import


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.

    # ====== YOUR CODE: ======
    gamma = mdp.gamma
    U = U_tag = deepcopy(U_init)
    delta = sys.maxsize
    while delta >= ((epsilon * (1 - mdp.gamma)) / mdp.gamma):
        delta = 0
        U = deepcopy(U_tag)

        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
        for i, j in valid_states:
            if (i, j) in mdp.terminal_states:
                U_tag[i][j] = float(mdp.board[i][j])
            else:
                max_utility = None
                curr_state = (i, j)
                for action in mdp.actions.keys():
                    p = mdp.transition_function[action]
                    utility_list = []
                    for next_state in [mdp.step(curr_state, a) for a in mdp.actions.keys()]:
                        next_i, next_j = next_state[0], next_state[1]
                        utility_list.append(U[next_i][next_j])
                    curr_utility = sum(np.array(p) * np.array(utility_list))

                    if max_utility is None or max_utility < curr_utility:
                        max_utility = curr_utility

                U_tag[i][j] = float(mdp.board[i][j]) + gamma * max_utility

            delta = max(delta, abs(U_tag[i][j] - U[i][j]))
        # # TODO: remove this print it is for debugging:
        # mdp.print_utility(U_tag)

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
    U = np.dot(x, r_mat)

    return U.reshape(mdp.num_row, mdp.num_col)
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
            if (i, j) in mdp.terminal_states:
                policy[i][j] = None
            else:
                max_utility, max_action = None, None
                for act in list(mdp.actions):
                    list_utility = []
                    probability = mdp.transition_function[act]
                    for next_state in [mdp.step((i,j), a) for a in mdp.actions.keys()]:
                        next_i, next_j = next_state[0], next_state[1]
                        list_utility.append(utility[next_i][next_j])
                    curr_utility = sum(np.array(probability) * np.array(list_utility))

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

"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
