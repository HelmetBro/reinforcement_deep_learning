import time
import joblib
import numpy as np
import tensorflow as tf
import os

INIT_SEED = 0

# For this implementation, i used these as a reference
# A2C: https://github.com/raillab/a2c
# A2C: https://github.com/takuseno/a2c
# BLOG: https://openai.com/blog/baselines-acktr-a2c/
# BLOG: https://danieltakeshi.github.io/2018/06/28/a2c-a3c/

# formula stuff, so not that major
def reward_reduction(rewards, stops, gamma):
    reward_update = 0
    reduc = []
    for reward, done in zip(rewards[::-1], stops[::-1]):
        reward_update = reward + gamma * reward_update * (1. - done) 
        reduc.append(reward_update)
    return reduc[::-1]

def get_vars(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()

class Model():

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(1e9)):

        nbatch = nenvs * nsteps

 		#enable GPU usage and parallelize threads, and create a session
        config = tf.ConfigProto(intra_op_parallelism_threads=nenvs, inter_op_parallelism_threads=nenvs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        init_start = tf.placeholder(tf.int32, [nbatch])
        start_G = tf.placeholder(tf.float32, [nbatch])
        init_reward = tf.placeholder(tf.float32, [nbatch])
        reward_L = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        pol_grad_loss = tf.reduce_mean(start_G * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=init_start))
        value_grad_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(train_model.vf), init_reward) / 2.0)
        
        # ++++ this calculates entropy
        first = train_model.pi - tf.reduce_max(train_model.pi, 1, keep_dims=True)
        exp_first = tf.exp(first)
        summation = tf.reduce_sum(exp_first, 1, keep_dims=True)
        entropy = tf.reduce_mean(tf.reduce_sum( (exp_first / summation) * (tf.log(summation) - first), 1))
        # ++++ 

        # alpha * d [action_loss(state, action)] * (predicted value of value function) / d(theta)
        loss = pol_grad_loss - entropy * ent_coef + value_grad_loss * vf_coef
        params = get_vars("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=reward_L, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        # NOTE: methods were influenced by a reference

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)
        self.load = load

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)
        self.save = save

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            td_map = {train_model.X: obs, init_start: actions, start_G: advs, init_reward: rewards, reward_L: lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            p_grad, v_grad, p_base, _ = sess.run( [pol_grad_loss, value_grad_loss, entropy, _train], td_map)
            return p_grad, v_grad, p_base
        self.train = train

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.value = step_model.value
        self.step = step_model.step
        self.initial_state = step_model.initial_state

        tf.global_variables_initializer().run(session=sess)


class Runner():

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.batch_ob_shape = (env.num_envs * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((env.num_envs, nh, nw, nc * nstack), dtype=np.uint8)
        self.nc = nc
        self.update_obs(env.reset())
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.stops = [False for _ in range(env.num_envs)]

    #taken from an implementation of the atari_wrapper class, helps performance
    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self):

    	# NOTE: some was influenced by a reference

        oENV, rewardsENV, actionsENV, valuesENV, stoppingENV = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            
            actions, values, states = self.model.step(self.obs, self.states, self.stops)
            oENV.append(np.copy(self.obs))
            actionsENV.append(actions)
            valuesENV.append(values)
            stoppingENV.append(self.stops)
            
            obs, rewards, stops, _ = self.env.step(actions)
            self.states = states
            self.stops = stops
            for n, done in enumerate(stops):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(obs)
            rewardsENV.append(rewards)

        stoppingENV.append(self.stops)

        # this is what brandon was talking about, putting it into a np array stores it explicitly
        # later look for implementation that doesn't do this
        oENV = np.asarray(oENV, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        rewardsENV = np.asarray(rewardsENV, dtype=np.float32).swapaxes(1, 0)
        actionsENV = np.asarray(actionsENV, dtype=np.int32).swapaxes(1, 0)
        valuesENV = np.asarray(valuesENV, dtype=np.float32).swapaxes(1, 0)
        stoppingENV = np.asarray(stoppingENV, dtype=np.bool).swapaxes(1, 0)
        mb_masks = stoppingENV[:, :-1]
        stoppingENV = stoppingENV[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.stops).tolist()
        
        #enumerate and get tuples 
        for n, (rewards, stops, value) in enumerate(zip(rewardsENV, stoppingENV, last_values)):
            rewards = rewards.tolist()
            stops = stops.tolist()
            if stops[-1] == 0:
                rewards = reward_reduction(rewards + [value], stops + [0], self.gamma)[:-1]
            else:
                rewards = reward_reduction(rewards, stops, self.gamma)
            rewardsENV[n] = rewards
        
        return oENV, mb_states, rewardsENV.flatten(), stoppingENV[:, :-1].flatten(), actionsENV.flatten(), valuesENV.flatten()


def learn(policy, env, new_session=True,  nsteps=5, nstack=4, total_timesteps=int(1e9),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, interval_saves=False, INTERVAL=1000,
          PATH='models'):

    #seeds
    tf.reset_default_graph()
    tf.set_random_seed(INIT_SEED)
    np.random.seed(INIT_SEED)

    model = Model(policy=policy, 
                  ob_space=env.observation_space, 
                  ac_space=env.action_space, 
                  nenvs=env.num_envs,
                  nsteps=nsteps, 
                  nstack=nstack,
                  ent_coef=ent_coef, 
                  vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr,
                  alpha=alpha, 
                  epsilon=epsilon, 
                  total_timesteps=total_timesteps)

    # load from previous (OPTIONAL)
    # file_path = os.path.join('models', ... + '.model')
    # model.load(file_path)

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    start_time = time.time()

    print('+ training +')
    batch_num = env.num_envs * nsteps
    for update in range(1, (total_timesteps // batch_num) + 1):
        
        obs, states, rewards, masks, actions, vals = runner.run()
        model.train(obs, states, rewards, masks, actions, vals)

        #update if it's time
        if update % INTERVAL == 0:
            print('GENERATION PASS:', update, 'TIME:', time.time() - start_time)

            if(interval_saves):
                file_path = os.path.join(PATH, env.env_id + "-" + str(update) + '.model')
            else:
                file_path = os.path.join(PATH, env.env_id + '.model')

            print(file_path)

            model.save(file_path)

    env.close()
    model.save(file_path)