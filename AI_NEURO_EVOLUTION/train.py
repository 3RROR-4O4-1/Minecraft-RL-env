import random
import numpy as np
from AI_NEURO_EVOLUTION.environment import ParkourEnv
from AI_NEURO_EVOLUTION.NN.agent import NeuroevolutionAgent
from AI_NEURO_EVOLUTION.NN.model import action_mapping



def train_neurev(pop_size=5, generations=50, mutation_rate=0.2, elite_nb=4):
    def evaluate_agent(agent, env, max_steps=1000):
        state = env.get_state()
        total_reward = 0

        for _ in range(max_steps):
            # Get agent action
            raw_actions = agent.act(state)
            actions = action_mapping(raw_actions)

            # Execute actions in Minecraft
            env.bot.setControlState('forward', actions['forward'])
            env.bot.setControlState('back', actions['back'])
            env.bot.setControlState('left', actions['left'])
            env.bot.setControlState('right', actions['right'])
            if actions['jump']:
                env.bot.setControlState('jump', True)
                env.bot.setControlState('jump', False)

            # Update state and reward
            state = env.get_state()
            total_reward += env.total_reward

            if env.bot.entity.position.y < 0:  # Fell off
                break

        return env.calculate_fitness()
    
    # Initialize population
    population = [NeuroevolutionAgent(state_size=9, action_size=5)
                  for _ in range(pop_size)]

    for generation in range(generations):
        # Evaluate population
        fitness_scores = []
        for agent in population:
            username = f"bot_{random.randint(1, 9999)}"
            env = ParkourEnv(username)
            fitness = evaluate_agent(agent, env)
            fitness_scores.append(fitness)
            env.bot.quit()

        # Selection (elitism + tournament)
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population),
                                           key=lambda pair: pair[0], reverse=True)]
        next_gen = sorted_pop[:elite_nb]

        # Fill rest with mutated offspring
        while len(next_gen) < pop_size:
            parent = random.choice(sorted_pop[:elite_nb * 2])
            child = parent.clone()
            child.mutate(mutation_rate)
            next_gen.append(child)

        population = next_gen
        print(f"Generation {generation} - Best Fitness: {max(fitness_scores):.2f}")

