import blue_ai.agents.tables as tables
import random
import sys
import pickle


class UpdatablePriorityQueue(dict):

    def insert(self, item, priority):
        """insert item with priority. If item already exists, update its priority to max of existing & new"""
        self[item] = max(self.get(item, priority), priority)

    def pop(self, operation=max, item=None):
        """pop item from queue. If item not specified, pop highest-priority item"""
        key_to_pop = item or operation(self, key=self.get)
        super().pop(key_to_pop)
        return key_to_pop

    def peek(self, operation=max):
        """see the item that will be popped next"""
        return operation(self, key=self.get)

    def is_empty(self):
        return len(self) == 0


class MBRL:
    """
    A tabular model-based reinforcement learner that uses prioritized sweeping to efficiency update
    Q values after each action.
    """
    def __init__(
        self,
        actions,
        epsilon=0.1,
        discount_factor=0.9,
        theta_threshold=0,
        max_value_iterations=sys.maxsize,
        q_default=100,
        r_default=0,
        c_default=0,
        t_table=None,
        c_table=None,
        r_table=None,
        q_table=None
    ):
        """
        Creates a model based reinforcement learner with the parameters specified
        :param actions: sequence of actions available, or a callable that gets the actions for a state
        :param epsilon: exploration factor
        :param discount_factor: discount factor
        :param theta_threshold: temporal differences greater than this will be added to the priority queue for updating
        :param max_value_iterations: max value calculations per step
        :param q_default: default q value
        :param r_default: default r value
        :param c_default: starting count for transition counts
        """
        self.actions = actions if callable(actions) else lambda state=None: actions
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.theta_threshold = theta_threshold
        self.max_value_iterations = max_value_iterations
        self.q_default = q_default
        self.r_default = r_default
        self.c_default = c_default

        # Create and initialize StateActionTables, TTable, and PQueue.
        self.Q = q_table if q_table is not None else tables.StateActionTable(default_value=self.q_default)
        self.R = r_table if r_table is not None else tables.StateActionTable(default_value=self.r_default)
        self.C = c_table if c_table is not None else tables.StateActionTable(default_value=self.c_default)
        self.T = t_table if t_table is not None else tables.TTable()
        self.PQueue = UpdatablePriorityQueue()

    def select_action(self, state):
        """
        Chooses an action from a specific state uses e-greedy exploration.
        """
        actions = self.actions(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            return self.Q.get_best_action(state=state, actions=actions)

    def update(self, state, action, new_state, reward, done=False):
        """
        Updates the Q, T, C, and R tables appropriately by conducting prioritized sweeping
        in order to only update the important states first.
        """
        # update T, C, and R tables
        self.T[state, action, new_state] += 1
        self.C[state, action] += 1
        self.R[state, action] += (reward - self.R[state, action]) / self.C[state, action]

        Vmax = 0 if done else max(self.Q.get_action_values(state=new_state, actions=self.actions(new_state)).values())

        priority = abs(reward + self.discount_factor * Vmax - self.Q[state, action])

        if priority > self.theta_threshold:
            self.PQueue.insert((state, action), priority)

        self.process_priority_queue(self.max_value_iterations)

    def process_priority_queue(self, n):
        for _ in range(n):
            if self.PQueue.is_empty():
                break

            state, act = self.PQueue.pop()

            self.Q[state, act] = self.R[state, act]

            # Loop for all (state, action) pairs predicted to lead to S:
            for state_prime in self.T.get_states_accessible_via_action(state, act):
                s_prime_values = self.Q.get_action_values(state=state_prime, actions=self.actions(state_prime)).values()
                Vmax = max(s_prime_values) if len(s_prime_values) > 0 else 0
                self.Q[state, act] += self.discount_factor * (self.T[state, act, state_prime] / self.C[state, act]) * Vmax

            for s_bar, act_from_sbar_to_s in self.T.get_state_actions_with_access_to(state):
                predicted_reward = self.R[s_bar, act_from_sbar_to_s]

                # Set priority for (sbar, act) pair
                priority = abs(predicted_reward + self.discount_factor * \
                    max(self.Q.get_action_values(state=state, actions=self.actions(state)).values()) - self.Q[s_bar, act_from_sbar_to_s])

                if priority > self.theta_threshold:
                    self.PQueue.insert((s_bar, act_from_sbar_to_s), priority)

    def get_dist_over_next_states(self, state, action) -> dict:
        """
        returns the states available and their respective probabilities from a state, action pair
        :param state: the current state
        :param action: the action being taken
        :return: the states accessible from the state-action pair and the respective probabilities
        """
        total = self.C[state, action]
        return {k: v / total for k, v in self.T.forward_map[state][action].items()}

    def get_random_walk(self, length):
        """
        return a list of nodes representing a random walk through the MDP
        :param length: length of the walk
        :return: list of states encountered
        """

        walk = [random.choice(list(self.T.forward_map))]
        for _ in range(length):
            actions = self.T.get_known_actions_from_state(walk[-1])
            # todo: should random walks consider rewards somehow? They respect boundaries, but should they respect
            # undesireable transitions too? if so, could have a weighted selection of action. softmax maybe
            if len(actions) == 0:
                break
            action = random.choice(actions)
            next_states = self.get_dist_over_next_states(walk[-1], action)
            if len(next_states) == 0:
                break
            walk.append(random.choices(list(next_states), weights=list(next_states.values()), k=1)[0])
        return walk

    def get_q_table_influenced_walk(self, length, temp, start=None):
        """
        return a list of nodes representing a q table influenced walk through the MDP
        :param length: length of the walk
        :param temp: temperature of the softmax
        :return: list of states encountered
        """
        import numpy as np
        import time
        def softmax(x):
            x = x - x.max()
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        walk = [start or random.choice(list(self.T.forward_map))]
        for _ in range(length):
            actions = self.T.get_known_actions_from_state(walk[-1])
            # todo: should random walks consider rewards somehow? They respect boundaries, but should they respect
            # undesireable transitions too? if so, could have a weighted selection of action. softmax maybe
            if len(actions) == 0:
                break
            action_values = self.Q.get_action_values(state=walk[-1], actions=actions)
            weights = softmax(np.array(list(action_values.values())) / temp)
            # pick a random weight based on the softmax
            action = random.choices(list(action_values.keys()), weights=weights)[0]
            next_states = self.get_dist_over_next_states(walk[-1], action)
            if len(next_states) == 0:
                break
            walk.append(random.choices(list(next_states), weights=list(next_states.values()), k=1)[0])
        return walk

    def random_walks(self, min_length, max_length, n, start=None):
        return [
            #self.get_random_walk(length=random.randint(min_length, max_length))
            self.get_q_table_influenced_walk(length=random.randint(min_length, max_length), temp=0.01, start=start)
            for _ in range(n)
        ]

    def save(self, filename):
        actions_backup = self.actions
        self.actions = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        self.actions = actions_backup

    @classmethod
    def load(cls, filename, actions):
        with open(filename, 'rb') as f:
            agent = pickle.load(f)
        agent.actions = actions if callable(actions) else lambda state=None: actions
        return agent
