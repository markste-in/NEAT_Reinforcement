#derived from the neat-python example

from __future__ import print_function


import neat
import gym
import numpy as np
import visualize
import os
import multiprocessing
import pickle

#Used for parameter optimization
import hyperopt
from hyperopt import fmin, tpe, space_eval
from hyperopt import hp , Trials, STATUS_OK
import time

#To create a mixed search algorithm
from hyperopt import anneal, rand, tpe, mix
from functools import partial

cpu_count = multiprocessing.cpu_count()

#env = gym.make('BipedalWalkerHardcore-v2')
#env = gym.make('CartPole-v1')
#env = gym.make('LunarLander-v2')
env = gym.make('Humanoid-v3')

def softmax(arr):
    return np.exp(arr)/np.sum(np.exp(arr))

def pick_action(obs, net):
    action = net.activate(obs)
    #action = 0 if action[0] < 0.5 else 1
    #return np.clip(np.around(action),0,1).astype(int)
    #return np.argmax(softmax(action))
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
    avg_reward = [run_env(net) for i in range(1)]
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
    winner = p.run(pe.evaluate,1000)
    pickle.dump({"population":p,
                 "best_genome":winner,
                 "config":config,
                 "stats":stats
                 },
                open("best_results.p", "wb"))

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    run_env(winner_net, render=True)

space = {'fitness_criterion': hp.choice('fitness_criterion',['max','mean','min']),
         'activation_default': hp.choice('activation_default',['relu','elu','sigmoid','sin','tanh','random']),
         'activation_mutate_rate' : hp.uniform('activation_mutate_rate',0.,1.),
         'bias_mutate_rate' : hp.uniform('bias_mutate_rate',0.,1.),
         'bias_mutate_power' : hp.uniform('bias_mutate_power',0.,1.), \
         'compatibility_disjoint_coefficient' :hp.uniform('compatibility_disjoint_coefficient',0.,2.),\
         'compatibility_weight_coefficient' :hp.uniform('compatibility_weight_coefficient',0.,2.),\
         'num_hidden' : hp.uniform('num_hidden',0,100), \
         'enabled_mutate_rate':hp.uniform('enabled_mutate_rate',0.,1.),\
         'response_mutate_rate':hp.uniform('response_mutate_rate',0.,1.),\
         'response_mutate_power':hp.uniform('response_mutate_power',0.,1.),\
         'response_replace_rate':hp.uniform('response_replace_rate',0.,1.),\
         'weight_mutate_rate':hp.uniform('weight_mutate_rate',0.,1.),\
         'weight_mutate_power':hp.uniform('weight_mutate_power',0.,1.),\
         'weight_replace_rate':hp.uniform('weight_replace_rate',0.,1.),\
         'species_elitism' : hp.uniform('species_elitism',0,1), \
         'species_fitness_func': hp.choice('species_fitness_func',['max','mean','min']), \
         'compatibility_threshold':hp.uniform('compatibility_threshold',0.,10.)
        }

mix_algo = partial(mix.suggest, p_suggest=[
    (0.05, rand.suggest),
    (0.75, tpe.suggest),
    (0.20, anneal.suggest)])

def load_trials():
    try:
        trials = pickle.load( open( "hyperopt_trials.pickle", "rb" ) )
        print("Trials file found with", len(trials),"entries")
        if len(trials) == 0:
            trials = Trials()
    except:
        trials = Trials()
        print("No trials found. Create new trials file")

    return trials
def save_trials(trials):
    print("Save", len(trials), "runs")
    pickle.dump(trials,open('hyperopt_trials.pickle', 'wb'))

def objective(X):
    space, config_file = X
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)

    config.fitness_criterion = space['fitness_criterion']
    config.genome_config.activation_default = space['activation_default']
    config.genome_config.activation_mutate_rate = space['activation_mutate_rate']
    config.genome_config.bias_mutate_rate = space['bias_mutate_rate']
    config.genome_config.bias_mutate_power = space['bias_mutate_power']
    config.genome_config.compatibility_disjoint_coefficient = space['compatibility_disjoint_coefficient']
    config.genome_config.compatibility_weight_coefficient = space['compatibility_weight_coefficient']
    config.genome_config.num_hidden = int(space['num_hidden'])
    config.genome_config.enabled_mutate_rate = space['enabled_mutate_rate']
    config.genome_config.response_mutate_rate = space['response_mutate_rate']
    config.genome_config.response_mutate_power = space['response_mutate_power']
    config.genome_config.response_replace_rate = space['response_replace_rate']
    config.genome_config.weight_mutate_rate = space['weight_mutate_rate']
    config.genome_config.weight_mutate_power = space['weight_mutate_power']
    config.genome_config.weight_replace_rate = space['weight_replace_rate']
    config.stagnation_config.species_elitism = int(space['species_elitism'])
    config.stagnation_config.species_fitness_func = space['species_fitness_func']
    config.species_set_config.compatibility_threshold = space['compatibility_threshold']

    # Create the population, which is the top-level object for a NEAT run.



    p = neat.Population(config)


    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    pe = neat.ParallelEvaluator(cpu_count, eval_genome)
    print("Starting evaluation on", cpu_count, "CPUs")
    winner = p.run(pe.evaluate,20)



    loss = -1. * winner.fitness
    return {
        'loss' : loss,
        'status' : STATUS_OK,
        'eval_time': time.time(),
        'reward' : winner.fitness
    }


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')

    trials = load_trials()
    best = []
    print(hyperopt.pyll.stochastic.sample(space))
    try:

        while True:
            max_trials = len(trials) + 1
            best.append(fmin(objective, [space, config_path], algo=mix_algo, max_evals=max_trials, trials=trials))
            if max_trials % 3 == 0:
                print("saving trials...")
                save_trials(trials)


    except KeyboardInterrupt:
        print("Analysis done...")
        save_trials(trials)
        best = best[-1]
        print(best)


    #if you just want to run the config file uncomment this and comment out everything above in "main"
    #run(config_path)
