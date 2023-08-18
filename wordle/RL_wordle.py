import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

env = gym.make("wordle-v0")

model = PPO(MlpPolicy, env, verbose=1, seed=0, n_epochs=1, learning_rate=0.00001)
model.learn(total_timesteps=1e5, n_eval_episodes=50)
model.save(f"wordle")

#env.plot_reward(env.everyTimesRewardlst)
env.plot_reward(env.ratelst)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
#
# env.close()