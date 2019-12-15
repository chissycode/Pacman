# Pacman

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install

```
python2.7.13
matplotlib
numpy
```

### Installing

A step by step series of examples that tell you how to get a development env running

```
python2 -m pip install matplotlib
```

And repeat

```
python2 -m pip install numpy
```

## Running the tests

Explain how to run tests for the game and how to set trainable parameters

### Basic Setup
run games by 100 times and be quiet(don't display realtime game)
```
python pacman.py -n 100 -q
```

### Tree Search Model

run multimax tree search model for pacman and ghost, set random model for opponent player

```
python pacman.py
```

run multimax tree search model for pacman and ghost, set greedy model for opponent player

```
python pacman.py -o GreedyAgent 
```

### Q-Learning Model
run games by 150 times and set up pacman trained by Q-Learning which is trained 100 times.
```
python pacman.py -p ApproximateQAgent -n 150 -x 100 -q
```

## Authors

* **Ang Li** - *Initial work, Tree Search, Opponent baseline* 
* **Qian Chen** - *Q-Learning Model*
* See also the list of [contributors](http://ai.berkeley.edu/project_overview.html) who participated in this project.

