* Three algorithms have been implemented: a2c, ddqn, and mcpg. 
* All three models were run, tested, and trained based off of 
* our personal implementations of the algorithm.

```

project/
	|--> a2c
	|--> ddqn
	|--> mcpg
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
	
		ddqn.py
			--> contains the ddqn model which uses temporal difference learning. Also contains the main training loop for ddqn.
		
		play.py
			--> renders the results of a pretrained ddqn model on the specified game
			
		wrappers.py
			--> contains pytorch optimized wrappers which filter the frame data from the environment before it is passed to the ddqn model
			
		models/
			--> folder to periodically store models while training from ddqn.py [necessary]
			
		plots/
			--> folder to periodically store plots while training from ddqn.py [necessary]

	+ mcpg +
	
		mcpg.py	
			--> Contains the Monto Carlo Policy Gradient
		
		trained_pong.py 
			--> Runs a trained model for Pong.
		
		policy_1550000_Final 
			--> Contains the trained weights for pong that can be loaded in to play games.

		project.ipynb 
			--> for displaying pong games using a trained learner

```
