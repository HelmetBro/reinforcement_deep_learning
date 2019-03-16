# personal_a2c_neural_network

To train, you can add optional command line arguments (order matters):

`python3 train.py`
This trains the model WITH NO incremental .model saves, at interval 1000, with default game "SpaceInvadersNoFrameskip-v4".

`python3 train.py -e StarGunnerNoFrameskip-v4 -i 1000`
This trains the model WITH incremental .model saves, at an interval specified by the integer after -s. This will save each .model with the generation update number after it. This runs the enviornment "StarGunner".

To play the model, you can use the following commands:

`python3 play.py`
This will start the simulation and attempt to find the save file without a generation number appened to it.

`python3 play.py -e StarGunnerNoFrameskip-v4 -i 6000`
This starts the simulation with the specified model generation number, and environment.
