import random
import operator


class Turkey(object):
    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.3):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Turkey
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the turkey
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the turkey is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning turkey can be altered,
        update these parameters when necessary.
        """

        if self.testing:    #  1. No random choice when testing
            self.epsilon = 0
        else:   #  2. Update parameters when learning
            if self.epsilon > 0.:
                self.epsilon -= 0.01

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the turkey.
        """

        #  3. Return turkey's current state
        return self.maze.sense_turkey()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        #  4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if state not in self.Qtable.keys():
            self.Qtable[state] = {'u': 0., 'd': 0., 'l': 0., 'r': 0.}

    def choose_action(self):
        """
        Return an action according to given rules
        """

        def is_random_exploration():

            #  5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            if random.random() < self.epsilon:
                return True
            else:
                return False

        final_action = ''
        if self.learning:
            if is_random_exploration():
                #  6. Return random choose aciton
                final_action = self.valid_actions[random.randint(0, 3)]
            else:
                #  7. Return action with highest q value
                final_action = max(
                    self.Qtable[self.state].items(),
                    key=operator.itemgetter(1))[0]
        elif self.testing:
            #  7. choose action with highest q value
            final_action = max(
                self.Qtable[self.state].items(),
                key=operator.itemgetter(1))[0]
        else:
            #  6. Return random choose aciton
            final_action = self.valid_actions[random.randint(0, 3)]

        return final_action

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            #  8. When learning, update the q table according
            # to the given rules

            # The reward the turkey get after doing the action
            reward_t1 = r

            # Code Review1: Replace the following code
            '''
            # Get the key & value which enables the max Q value in State(t+1)
            max_Q_St1_info = max(self.Qtable[next_state].items(), key=operator.itemgetter(1))

            # The action which enables the max Q value in State(t+1)
            max_Q_St1_action = max_Q_St1_info[0]

            # The max Q value in State(t+1)
            max_Q_St1 = max_Q_St1_info[1]
            '''
            max_Q_St1 = max(self.Qtable[next_state].values())

            Q_old = self.Qtable[self.state][action]
            Q_new = reward_t1 + self.gamma * max_Q_St1

            self.Qtable[self.state][action] = (
                1 - self.alpha) * Q_old + self.alpha * Q_new

    def update(self):
        """
        Describle the procedure what to do when update the turkey.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state()  # Get the current state
        self.create_Qtable_line(self.state)  # For the state, create q table line

        action = self.choose_action()  # choose action for this state
        reward = self.maze.move_turkey(action)  # move turkey for given action

        next_state = self.sense_state()  # get next state
        self.create_Qtable_line(
            next_state)  # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state)  # update q table
            self.update_parameter()  # update parameters

        return action, reward
