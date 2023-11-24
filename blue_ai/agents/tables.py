import itertools
import numpy as np
import networkx as nx


total_Q_writes = 0


class StateActionTable:
    """
    A table of values that can be looked up by state and action.
    Example usage:
        Q = StateActionTable(default_value=10)
        Q['s1', 'a1'] = 5.2
        Q['s1', 'a2'] = 2
        print(Q['s2', 'a5'])  # prints 10
    """

    def __init__(self, default_value, table: dict = None):
        self.default_value = default_value
        self.table = table or dict()

    def __getitem__(self, item):
        return self.table.get(item[0], dict()).get(item[1], self.default_value)

    def __setitem__(self, key, value):
        global total_Q_writes
        total_Q_writes += 1
        self.table.setdefault(key[0], dict())[key[1]] = value

    def contains(self, state, action):
        return state in self.table and action in self.table[state]

    def get_action_values(self, state, actions=None):
        """
        get dict of action values for the specified state
        :param state: state to get action values for
        :param actions: list of actions of interest. If None, will consider only actions that exist in the table for
                        the given state
        :return: dictionary of action -> value
        """
        if actions is None:
            actions = self.table[state].keys()
        return {action: self[state, action] for action in actions}

    def get_best_action(self, state, actions=None):
        """
        get best action to take for a specified state
        :param state: state to get action values for
        :param actions: list of actions of interest
        :return: the best action (key with the highest value)
        """
        if actions is None:
            actions = self.table[state].keys()
        action_values = self.get_action_values(state=state, actions=actions)
        return max(action_values, key=action_values.get)

    def get_best_value(self, state, actions=None):
        """
        get best value available from state, across the given actions
        :param state: state to consider
        :param actions: actions to maximize across
        :return: max value
        """
        return max(self.get_action_values(state=state, actions=actions).values())

    def get_all_states(self):
        return self.table.keys()

    def get_best_values_dict(self, actions=None):
        return {
            state: self.get_best_value(state, actions)
            for state in self.table
        }


class TTable:
    """
    A table of values that can be looked up by state, action, and next state
    can infer both successor and predecessor states for a given state
    Example usage:
        T = TTable()
        T['s1', 'a1', 's2'] += 1
        T['s1', 'a2', 's3'] += 1
        T['s1', 'a1', 's4'] += 1
        T.get_states_accessible_from('s1')  # returns ['s2', 's4', 's3']
        T.get_states_with_access_to('s3')  # returns ['s1']
    """

    def __init__(self):
        self.forward_map = dict()
        self.backward_map = dict()

    def __getitem__(self, item):
        return self.forward_map.get(item[0], dict()).get(item[1], dict()).get(item[2], 0)

    def __setitem__(self, key, value):
        self.forward_map.setdefault(key[0], dict()).setdefault(key[1], dict())[key[2]] = value
        self.backward_map.setdefault(key[2], dict()).setdefault(key[1], dict())[key[0]] = value

    def get_states_accessible_from(self, state) -> set:
        """
        returns states that can be accessed from the specified state (inferred from table entries)
        :param state: state to infer successor states for
        :return: list of states
        """
        return set(itertools.chain(*[x.keys() for x in self.forward_map.get(state, {}).values()]))

    def get_states_accessible_via_action(self, state, action) -> set:
        """
        returns states that can be accessed from the specified state taking a specific action (inferred from table entries)
        :param state: state to infer successor states for
        :param action: current action being taken
        :return: list of states
        """
        return set(self.forward_map.get(state, {}).get(action, {}).keys())

    def get_states_with_access_to(self, state):
        """
        returns states that have access to the specified state (inferred from table entries)
        :param state: state to infer predecessor states for
        :return: set of states
        """
        return set(itertools.chain(*[x.keys() for x in self.backward_map.get(state, {}).values()]))

    def get_random_transition_probabilities_from(self, state):
        """
        returns a probability distribution over next states after executing a random action from 'state'
        :param state: starting state
        :return: dict of new_state -> probability
        """
        if state not in self.forward_map:
            return dict()

        p = dict()

        action_dict = self.forward_map[state]
        for new_state_counts in action_dict.values():
            total_count = sum(new_state_counts.values())
            for new_state in new_state_counts:
                p[new_state] = p.get(new_state, 0) + new_state_counts[new_state] / total_count / len(action_dict)

        return p

    def get_state_actions_with_access_to(self, state):
        """
        returns state-action tuples that have given access to the specified state (inferred from table entries)
        :param state: state to infer predecessor states for
        :return: generator of state-action tuples
        """
        try:
            return itertools.chain(
                *[itertools.product(s1, (act, )) for act, s1 in self.backward_map[state].items()]
            )
        except KeyError:
            return []

    def get_all_states(self):
        """
        :return: all the states stored in the table
        """
        return list(set(self.forward_map).union(set(self.backward_map)))

    def get_state_probabilities_from_state_action(self, state, action, planT=None, as_dict=False):
        """
        returns the states available and their respective probabilities from a state, action pair
        :param state: the current state
        :param action: the action being taken
        :return: the states available to access from a state, action pair and the respective probabilities
        """

        def get_state_counts(mapping, state, action):
            try:
                items = mapping[state][action].items()
                states_available, state_counts = map(list, zip(*items))
            except KeyError:
                states_available, state_counts = [], []
            return states_available, state_counts

        if planT is not None:
            states_available, state_counts = get_state_counts(self.forward_map, state, action)
            states_available2, state_counts2 = get_state_counts(planT.forward_map, state, action)

            for i in range(len(states_available)):
                for j in range(len(states_available2)):
                    if states_available[i] == states_available2[j]:
                        state_counts[i] += state_counts2[j]
                        # break
                    else:
                        states_available.append(states_available2[j])
                        state_counts.append(state_counts2[j])
        else:
            states_available, state_counts = list(map(list, zip(*list(self.forward_map[state][action].items()))))

        total_counts = sum(state_counts)

        state_probabilities = [count / total_counts for count in state_counts]

        if as_dict:
            return dict(zip(states_available, state_probabilities))
        return states_available, state_probabilities

    def generate_random_walk(self, length, actions):
        """
        returns a list of nodes that were generated as a random walk through the table
        :param length: the length of the random walk
        :return: a list of nodes in a random walk
        """
        random_walk = []
        all_states = self.get_all_states()
        current = all_states[np.random.choice(len(all_states))]
        random_walk.append(current)

        for _ in range(length - 1):
            action = np.random.choice(actions)
            try:
                states_available, state_probabilities = self.get_state_probabilities_from_state_action(current, action)
                current = states_available[np.random.choice(len(states_available), p=state_probabilities)]
                random_walk.append(current)
            except KeyError:
                break

        return random_walk

    def get_known_actions_from_state(self, state):
        return list(self.forward_map.get(state, []))

    def to_random_transition_graph(self, pos_calculator: callable = None):
        """
        convert a T table (from a model-based RL agent) to a directed graph suitable for random walk generation
        :param T: the TTable object
        :param pos_calculator: optional callable that returns (x,y) coordinates for plotting given a state
        :return a networkx DiGraph object with a node for each state in the T table, and directed edges with weights
        representing the probability that a random action will cause that transition
        """

        G = nx.DiGraph()
        all_states = self.get_all_states()

        for state in all_states:
            for state2, weight in self.get_random_transition_probabilities_from(state).items():
                G.add_edge(state, state2, weight=weight)

        if callable(pos_calculator):
            for state in all_states:
                G.nodes[state]['pos'] = pos_calculator(state)

        return G

    def to_simple_graph(self, pos_calculator: callable = None):
        """
        convert a T table (from a model-based RL agent) to a directed graph
        :param T: the TTable object
        :param pos_calculator: optional callable that returns (x,y) coordinates for plotting given a state
        :return a networkx DiGraph object with a node for each state in the T table, and directed edges showing if a
            transition between the states is possible
        """
        G = nx.DiGraph()

        for state in self.forward_map:
            for successor in self.get_states_accessible_from(state):
                G.add_edge(state, successor)

        return G


if __name__ == '__main__':
    # use StateActionTable class for Q, R, C tables
    R = StateActionTable(default_value=10)
    R['s1', 'a1'] = 5.2
    R['s1', 'a2'] += 2
    print('R entry for s1, a1: ', R['s1', 'a1'])
    print('R entry for s2, a2: ', R['s1', 'a2'])
    print('R entries for s1, actions a1-a3: ', R.get_action_values(state='s1', actions=['a1', 'a2', 'a3']))
    print('Best action choice (max):', R.get_best_action(state='s1', actions=['a1', 'a2', 'a3']))

    # use TTable class for the T table
    T = TTable()
    T['s1', 'a1', 's2'] += 1
    T['s1', 'a2', 's3'] += 1
    T['s1', 'a1', 's4'] += 1
    T['s3', 'a1', 's2'] += 1
    T['s2', 'a3', 's3'] += 1
    T['s4', 'a3', 's2'] += 1
    print('random walk:', T.generate_random_walk(3, ['a1', 'a2', 'a3']))
    print('states accessible from s1: ', T.get_states_accessible_from('s1'))
    print('state-actions with access to s3: ', [x for x in T.get_states_with_access_to('s2')])
