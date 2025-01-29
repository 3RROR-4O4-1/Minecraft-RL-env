import numpy as np
from AI_NEURO_EVOLUTION.NN.model import create_model, mutate_weights


class NeuroevolutionAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = create_model(state_size, action_size)
        self.fitness = 0

    def act(self, state):
        # Simple feedforward network
        state = np.array(state).reshape(1, -1)
        weights = self.model['weights']
        bias = self.model['bias']

        # Neural network computation
        output = np.dot(state, weights) + bias
        return output[0]  # Return raw actions

    def mutate(self, mutation_rate=0.1):
        # Apply random mutations to weights
        self.model['weights'] = mutate_weights(self.model['weights'], mutation_rate)
        self.model['bias'] = mutate_weights(self.model['bias'], mutation_rate / 2)

    def clone(self):
        cloned = NeuroevolutionAgent(self.state_size, self.action_size)
        cloned.model = {
            'weights': np.copy(self.model['weights']),
            'bias': np.copy(self.model['bias'])
        }
        return cloned