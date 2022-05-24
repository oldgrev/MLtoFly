#https://stable-baselines3.readthedocs.io/en/master/guide/imitation.html
# Behavioral Cloning
# customized to leverage telemetry

from rlenv import flyEnv
import pandas as pd
import numpy as np
import time
import os

#dont think this worked but leaving...

env = flyEnv()
env.reset()

models_dir = f"E:/ml/models/{int(time.time())}/"
logdir = f"E:/ml/logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
#for step in range(count_row):
    #if(step == 0):
    #    pass
    #else:
        #time_step = step
        #print("DEBUG:", step)
        #step_action = np.array([df.iloc[step-1].collective,df.iloc[step-1].bank,df.iloc[step-1].pitch1,df.iloc[step-1].rudder]).astype(np.float32)
        #print("DEBUG: action",step_action,step) 
        #obs, reward, done, info = env.step(step_action)
        #print('DEBUG: reward',reward)

#https://github.com/HumanCompatibleAI/imitation/blob/master/examples/1_train_bc.ipynb
#implements based on this example



from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
#model = PPO('MlpPolicy', env=env, verbose=2, tensorboard_log=logdir)

model = PPO.load("2200",env=env,tensorboard_log=logdir)
TIMESTEPS = 200
EPISODES = 5
iters = 0
while True:
	iters += 1
	#model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name=f"PPO")
	#model.learn(n_eval_episodes=EPISODES, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")

##import gym
##from stable_baselines3.common.evaluation import evaluate_policy
##from imitation.data import rollout         #imitation algorithms don't work on windows because it doesn't have jaxlib. CBF manually compiling.
##from imitation.data.wrappers import RolloutInfoWrapper         #imitation algorithms don't work on windows because it doesn't have jaxlib. CBF manually compiling.
##from stable_baselines3.common.vec_env import DummyVecEnv


#expert = PPO(
#    policy=MlpPolicy,
#    env=env,
#   seed=0,
#   batch_size=64,
#    ent_coef=0.0,
#    learning_rate=0.0003,
#    n_epochs=1,
#    n_steps=64,
#    tensorboard_log=logdir,
#)



#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
#model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
#model.save(f"{models_dir}/{TIMESTEPS*iters}")

#expert.load("expert")

#TIMESTEPS = 10000
#iters = 0
#while True:
    #iters += 1
    #model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    #model.save(f"{models_dir}/{TIMESTEPS*iters}")
    #expert.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
##
##rollouts = rollout.rollout(
##    expert,
##    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
##    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
##)
##transitions = rollout.flatten_trajectories(rollouts)
##
##expert.save("expert")

########I have no idea if everything below is needed or not#############
########At this point I have the expert model????#########################
########
########There's other BC algorithms I could use https://github.com/HumanCompatibleAI/imitation/tree/master/examples
########
########The DAgger algorithm is an extension of behavior cloning. In behavior cloning, the training trajectories are recorded directly from an expert.
########In DAgger, the learner generates the trajectories but an experts corrects the actions with the optimal actions in each of the visited states. 
########This ensures that the state distribution of the training data matches that of the learner's current policy.
########
########That could be useful if I set the environment to have random input values, but using the expert's transitions?????
########


##print(
##    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
##After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
##The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
##"""
##)

##from imitation.algorithms import bc   #imitation algorithms don't work on windows because it doesn't have jaxlib

##c_trainer = bc.BC(
##    observation_space=env.observation_space,
##    action_space=env.action_space,
##    demonstrations=transitions,
##)

#before and after scores will be identical
##reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
##print(f"Reward before training: {reward_before_training}")

##bc_trainer.train(n_epochs=1)
##reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
##print(f"Reward after training: {reward_after_training}")

##bc_trainer.policy.save("bc_trainer")