# personal_a2c_neural_network

To train, you can add optional command line arguments (order matters):

`python3 train.py`
This trains the model WITH NO incremental .model saves, at interval 1000, with default game "SpaceInvadersNoFrameskip-v4" and default folder 'models/'.

`python3 train.py -e PongNoFrameskip-v4 -i 4000 -s models/`
This trains the model WITH incremental .model saves, at an interval specified by the integer after -i. This will save each .model with the generation update number after it, into the file path of 'models/'. This runs the enviornment "Pong".

To play the model, you can use the following commands:

`python3 play.py`
This will start the simulation and attempt to find the save file in default folder 'models/', without a generation number appened to it.

`python3 play.py -e PongNoFrameskip-v4 -v 1515000 -l models`
This starts the simulation with the specified model generation number, and environment.
