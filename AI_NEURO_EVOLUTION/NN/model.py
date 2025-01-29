import numpy as np

def create_model(input_size, output_size, hidden_size=8):
    return {
        'weights': np.random.randn(input_size, hidden_size) * 0.1,
        'bias': np.random.randn(hidden_size, output_size) * 0.1
    }

def mutate_weights(weights, mutation_rate):
    mutation = np.random.normal(scale=mutation_rate, size=weights.shape)
    return weights + mutation

def action_mapping(raw_actions):
    """Map neural network output to bot controls"""
    return {
        'forward': raw_actions[0] > 0,
        'back': raw_actions[1] > 0,
        'left': raw_actions[2] > 0,
        'right': raw_actions[3] > 0,
        'jump': raw_actions[4] > 0.5
    }