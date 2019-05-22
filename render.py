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

def run_env_cont(net, render=True):
    while True:
        run_env(net=net, render=render)



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        avg_reward = [run_env(net) for i in range(3)]
        genome.fitness = np.average(avg_reward)



p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-664')
config = p.config

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Show output of the most fit genome against training data.
print('\nOutput:')
winner = p.run(eval_genomes,3)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

visualize.draw_net(config, winner, True)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

run_env_cont(winner_net, render=True)



