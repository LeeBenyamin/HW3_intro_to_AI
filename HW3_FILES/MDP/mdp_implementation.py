import math
import sys
from copy import deepcopy
import numpy as np
import mdp # TODO: remove this import

actions_keys = ['UP', 'DOWN', 'RIGHT', 'LEFT']

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
                    # curr_utility = 0
                    # for index, probability in enumerate(mdp.transition_function[action]):
                    #     next_state = mdp.step((i, j), actions_keys[index])
                    #     next_i, next_j = next_state[0], next_state[1]
                    #     curr_utility += probability * U[next_i][next_j]

                    if not max_utility or max_utility < curr_utility:
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
    all_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col)]
    valid_states = [(i, j) for (i, j) in all_states if mdp.board[i][j] != 'WALL' and (i, j) not in mdp.terminal_states]
    for curr_state in valid_states:
        i, j = curr_state
        max_utility = None
        max_action = None
        for action in mdp.actions.keys():
            p = mdp.transition_function[action]
            utility_list = []
            for next_state in [mdp.step(curr_state, a) for a in mdp.actions.keys()]:
                utility_list.append(U[next_state[0]][next_state[1]])

            curr_utility = float(mdp.board[i][j]) + mdp.gamma * sum(np.array(p) * np.array(utility_list))

            if max_utility is None or max_utility < curr_utility:
                max_utility = curr_utility
                max_action = action

        policy[i][j] = max_action

    none_policy_states = [(i, j) for (i, j) in all_states if (i, j) not in valid_states]
    for i, j in none_policy_states:
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

        if mdp.board[i][j] == 'WALL':
            continue
        r_mat[s_idx] = float(mdp.board[i][j])

        if (i, j) in mdp.terminal_states:
            continue

        for next_idx in range(num_states):
            p = 0
            next_s = idx_to_state(next_idx, mdp)

            for action_idx, action in enumerate(actions):
                if mdp.step((i, j), action) == next_s:
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
    all_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col)]
    valid_states = [(i, j) for (i, j) in all_states if (i, j) not in mdp.terminal_states or mdp.board[i][j] != 'WALL']
    changed = True
    while changed:
        utility = policy_evaluation(mdp, policy)
        changed = False
        for i, j in valid_states:
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

    none_policy_states = [(i, j) for (i,j) in all_states if (i, j) not in valid_states]
    for i, j in none_policy_states:
        policy[i][j] = None

    return policy
    # ========================

"""For this functions, you can import what ever you want """

from termcolor import colored

def evaluate_state(mdp, U, state, epsilon=1e-4):
    i, j = state
    if mdp.board[i][j] and mdp.board[i][j] != 'WALL':
        R = float(mdp.board[i][j])
    else:
        return None, None
    max_utility = 0
    utility_actions = {}
    max_actions = [actions_keys[0]]  # default option
    if state not in mdp.terminal_states:
        i, j = state
        for act in mdp.actions.keys():
            curr_utility = 0
            for index, probability in enumerate(mdp.transition_function[act]):
                next_state = mdp.step((i, j), actions_keys[index])
                next_i, next_j = next_state[0], next_state[1]
                curr_utility += probability * U[next_i][next_j]
            utility_actions[act] = curr_utility
        max_utility = max(utility_actions.values())
        max_actions = [act for act in mdp.actions.keys() if utility_actions[act] >= max_utility - epsilon]
    return max_utility, max_actions, R


def print_policy2(mdp, policy):
    actions_chars = {
        "UP": "↑",
        "DOWN": "↓",
        "RIGHT": "→",
        "LEFT": "←",
        "WALL": "█",
        None: ""
    }
    res = ""
    for r in range(mdp.num_row):
        res += "|"
        for c in range(mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                val = mdp.board[r][c]
            else:
                if type(policy[r][c]) is str:
                    policy_chars = actions_chars.get(policy[r][c], policy[r][c])
                else:
                    policy_chars = [actions_chars.get(action, action) for action in policy[r][c]]
                val = ''.join(policy_chars)
            if (r, c) in mdp.terminal_states:
                res += " " + colored(val[:5].ljust(5), 'red') + " |"  # format
            elif mdp.board[r][c] == 'WALL':
                res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
            else:
                res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)

def print_policy_ours(mdp, policy):
    cell_r_size = 2
    cell_c_size = 2

    actions_chars = {
        "UP": "↑",
        "DOWN": "↓",
        "RIGHT": "→",
        "LEFT": "←",
        "WALL": "█",
        None: ""
    }
    table_size = 1 + mdp.num_col * (cell_c_size + 1) + 1

    res = ""

    for r in range(mdp.num_row):
        if r == 0:
            res += f"┌{'┬'.join(['─' * cell_c_size] * mdp.num_col)}┐\n"
        else:
            res += f"├{'┼'.join(['─' * cell_c_size] * mdp.num_col)}┤\n"
        for _ in range(cell_r_size):
            res += f"│{'│'.join([' ' * cell_c_size] * mdp.num_col)}│\n"
        if r == mdp.num_row - 1:
            res += f"└{'┴'.join(['─' * cell_c_size] * mdp.num_col)}┘\n"

    res = list(res)

    for i, row in enumerate(policy):
        for j, actions in enumerate(row):
            if actions == 0:
                actions = [mdp.board[i][j].lower(), "WALL"]
            if (i, j) in mdp.terminal_states:
                # actions = ["1", mdp.board[i][j]]
                actions = [mdp.board[i][j]]
            for action in actions:
                chars_to_curr_board = actions_chars.get(action, action)
                for curr_char in chars_to_curr_board:
                    row = 1 + i * (cell_r_size + 1)
                    col = 1 + j * (cell_c_size + 1)

                    res[row * table_size + col] = curr_char

    print("".join(res))

def get_all_policies(mdp, U, prev_policy=None):  # You can add more input parameters as needed
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #
    # ====== YOUR CODE: ======
    num_policies = 1
    unchanged = True
    policy = deepcopy(U)
    valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if (i, j) not in mdp.terminal_states
                    and mdp.board[i][j] != 'WALL']
    for s in valid_states:
        i, j = s
        _, utility_actions, R = evaluate_state(mdp, U, s)
        if utility_actions is not None:
            # policy[i][j] = [action for action in utility_actions.keys() if round(utility_actions[action], 2) == round(U[i][j], 2)]
            policy[i][j] = utility_actions
            if prev_policy:
                prev_acts = prev_policy[i][j]
                if utility_actions != prev_acts:
                    unchanged = False
            num_policies *= len(utility_actions)
    # Visualize or print the policies
    # (You can implement visualization based on your preference)
# TODO print policy for terminal states and wall
    print_policy2(mdp, policy)
    return num_policies, policy, unchanged
    # ========================

def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # Given the mdp
    # print / displays the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    init_u = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

    rewards = []
    states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
    for state in states:
        i, j = state
        rewards.append(float(mdp.board[i][j]))

    if mdp.gamma == 1:
        max_reward = 0
    else:
        max_reward = max(rewards)
    min_reward = min(rewards)

    mdp_array = np.array(mdp.board)
    policies_list = []
    prev_policies = None
    reward = min_reward
    r_res = 0.005
    all_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col)]

    while reward <= max_reward:
        reward = round(reward, 3)
        current_mdp = deepcopy(mdp)
        current_mdp_array = np.copy(mdp_array)
        for state in all_states:
            if state not in mdp.terminal_states and mdp_array[state] != 'WALL':
                current_mdp_array[state] = str(reward)
        current_mdp.board = current_mdp_array.tolist()

        U = value_iteration(current_mdp, U_init=init_u)
        _, policies, unchanged = get_all_policies(current_mdp, U, prev_policies)

        if not unchanged or not policies_list:
            policies_list.append([reward + r_res, policies])

        prev_policies = policies
        reward += r_res

    range_list = []
    for idx in range(len(policies_list)):
        reward, policies = policies_list[idx]
        print_policy2(mdp, policies)

        string_for_range = ""
        if idx > 0:
            current_range = policies_list[idx - 1][0]
            string_for_range += "{:.3f} <= R(s)".format(current_range)
            range_list.append(current_range)

        else:
            string_for_range += " -5 <= R(s) "

        if idx < len(policies_list) - 1:
            if string_for_range:
                string_for_range += " < "
            string_for_range += "{:.3f}".format(reward)

        else:
            string_for_range += " < 5 "

        if string_for_range:
            print(string_for_range)
    return range_list
    # ========================
