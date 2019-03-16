* Three algorithms have been implemented: a2c, ddqn, and ppo. 
* All three models were run, tested, and trained based off of 
* our personal implementations of the algorithm.

project/
	|--> a2c
	|--> ddqn
	|--> ppo
	|--> A2Cgifs [gifs taken from best trained model in a2c]
	|--> DDQNgifs [gifs taken from best trained model in ddqn]

	+ a2c +

		a2c.py           
			--> contains the model which hosts the working relationship between the actor and critic

		policy.py  
			--> contains the policy and convolutional neural networks used by the actor and critic

		play.py
			--> runs the network depending on the command line arguments (see README.md)
		
		train.py
			--> trains the network depending on the command line arguments (see README.md)

		atari_wrappers.py  
			--> wrapper file used by gym retro to take care of game data buffers and frame stacking optimization.

		subproc_vec_env.py
			--> wrapper file for gym retro to take care of enviornment manipulation and simulation.
			
		gifs/
			--> folder to store gifs generated from play.py [necessary]

		models/  
			--> folder to store models generated from train.py [necessary]

		README.md     
			--> instructions on how to use run and train models with various environments

	+ ddqn +

	+ ppo +