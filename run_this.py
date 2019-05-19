#derived from the neat-python example

from __future__ import print_function
import neat
import gym
import numpy as np
import visualize


env = gym.make('BipedalWalker-v2')
#env = gym.make('CartPole-v1')

def pick_action(obs, net):
    action = net.activate(obs)
    #action = 0 if action[0] < 0.5 else 1
    return action

def run_env(net, render=False):
    sum_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action = pick_action(obs,net)
        if render:
            env.render()
        obs, reward, done, info = env.step(action)
        sum_reward+=reward
        if done:
            break

    return sum_reward



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        avg_reward = [run_env(net) for i in range(3)]
        genome.fitness = np.average(avg_reward)


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-164')

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
run_env(winner_net, render=True)

visualize.draw_net(config, winner, True)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

