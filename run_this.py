#derived from the neat-python example

from __future__ import print_function
import neat
import gym
import numpy as np
import visualize
import os
import multiprocessing

cpu_count = multiprocessing.cpu_count()

env = gym.make('BipedalWalkerHardcore-v2')
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
    assert type(sum_reward) is not type(None), "Something bad happened"
    return sum_reward



def eval_genome(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    avg_reward = [run_env(net) for i in range(3)]
    return np.average(avg_reward)



def run (config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-164')

    # Run until a solution is found.
    pe = neat.ParallelEvaluator(cpu_count, eval_genome)
    print("Starting evaluation on", cpu_count, "CPUs")
    winner = p.run(pe.evaluate, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    run_env(winner_net, render=True)

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
