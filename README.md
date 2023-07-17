# Miner-Gym: a framework using reinforcement learning for path planning and navigation in underground mines

## Installation
A conda environment can be installed from the provided `environment.yml` file. To install the conda environment with all dependencies:
```
conda env create -f environment.yml
```

## Basic Usage
The `play.py` script can be used to control the agent manually using the W, A, S, and D keys. For models trained using the Stable Baselines project, the `test.py` file can be used to quickly load a saved model and run a few test episodes.

## Training
Currently, this framework supports training models using stable-baselines3. To train a new model, use the helper functions in `trainer.py` to load the desired model (i.e. `make_a2c_model` for using A2C). If no name is specified, a new model is created. If the name of a previously saved model is given, the model is loaded to be trained for additional episodes. A `MineEnv` instance can be made based on preset layouts using the constructors in `envs.py`.