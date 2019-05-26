[terminal]: pics/terminal.png
[570gens]: pics/570gens.png
[bipedalgif]: pics/bipedalwalker.GIF

# Parallelized NEAT Reinforcement Learning with OpenAI Gym
Testing a scaled implementation of the NEAT algorithm (NeuroEvolution of Augmenting Topologies) on some [OpenAI gym](https://gym.openai.com/) environments

The example is one of the simplest possible implementation using a modified version of the [neat-python](https://github.com/markste-in/neat-python) library and the Humanoid-v3 environment.

If you prefer to use the original version of neat-python you have to delete the following line out of the config file (which will repopulate a species with the best genome after total extinction)
```python
reset_with_best       = 1
```

Then the example can simply be run by typing 
```python
python run_this.py
```
![run in terminal][terminal]


After 570 generations
![570 generations later][570gens]

![bipedalanimation][bipedalgif]