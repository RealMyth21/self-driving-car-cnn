import gymnasium as gym
import gym_ppo
import carla
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
import signal


def make_env(rank, params):
  def _init():
      env = gym.make('ppo-v0', params=params)
      def close_env():
        env.clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
        print(f"Environment {rank} closed")
      
      # Ensure cleanup on exit
      signal.signal(signal.SIGINT, lambda *args: close_env())
      signal.signal(signal.SIGTERM, lambda *args: close_env())

      return env
  return _init

class CustomCombinedExtractor(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict):
      super().__init__(observation_space, features_dim=1)
      
      image_space = observation_space['semantic_image']
      image_shape = image_space.shape
      
      # CNN for image processing
      self.cnn = nn.Sequential(
          nn.Conv2d(image_shape[2], 32, kernel_size=4, stride=2, padding=0),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten()
      )
      
      # Calculate CNN output dimension
      with torch.no_grad():
          n_flatten = self.cnn(torch.zeros(1, image_shape[2], image_shape[0], image_shape[1])).shape[1]
      
      # Network for scalar inputs
      self.scalar_net = nn.Sequential(
          nn.Linear(2, 32),
          nn.LayerNorm(32),
          nn.ReLU(),
          nn.Linear(32, 32),
          nn.LayerNorm(32),
          nn.ReLU()
      )
      
      # Combined features network
      self.combined_net = nn.Sequential(
          nn.Linear(n_flatten + 32, 256),
          nn.LayerNorm(256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.LayerNorm(128),
          nn.ReLU()
      )
      
      self._features_dim = 128

  def forward(self, observations):
      image = observations['semantic_image'].permute(0, 3, 1, 2)
      image = image.float() / 255.0
      cnn_features = self.cnn(image)
      
      scalar_input = torch.cat([
          observations['waypoint_distance'],
          observations['waypoint_angle']
      ], dim=1)
      scalar_features = self.scalar_net(scalar_input)
      
      # Combine features
      combined = torch.cat([cnn_features, scalar_features], dim=1)
      return self.combined_net(combined)
  
def train_agent(params, total_timesteps=500000):
  num_envs = 2
  env = SubprocVecEnv([make_env(i, params) for i in range(num_envs)])
  
  # Normalization
  env = VecNormalize(
      env,
      norm_obs=True,
      norm_reward=True,
      clip_obs=10.,
      clip_reward=10.,
      gamma=0.99,
      epsilon=1e-8
  )
  
  policy_kwargs = dict(
      features_extractor_class=CustomCombinedExtractor,
      net_arch=dict(pi=[256, 128], vf=[256, 128]),
      activation_fn=nn.ReLU,
      normalize_images=False
  )
  
  model = PPO(
      "MultiInputPolicy",
      env,
      verbose=1,
      learning_rate=lambda remaining_progress: 3e-4 * remaining_progress,
      n_steps=2048,
      batch_size=64,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      clip_range_vf=0.2,
      ent_coef=0.3,
      vf_coef=0.5,
      max_grad_norm=0.5,
      target_kl=0.03  ,
      tensorboard_log="./carla_tensorboard/",
      device='auto'
  )

  
  
  model.learn(
      total_timesteps=total_timesteps,
      log_interval=1,
      progress_bar=True
  )
  
  return model


def main():
  # Parameters
  params = {
    'target_speed': 8,
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'dt': 0.01,
    'ego_vehicle_filter': 'vehicle.ford.mustang',
    'steering_limits': [-1, 1],
    'velocity_limits': [-1, 1],
    'port': 2000,
    'town': 'Town10HD',
    'max_time_episode': 5000,
    'max_waypt': 12,
    'd_behind': 12,
    'out_lane_thres': 2.0,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'spec_dist': 10,
    'obs_size_x': 256,
    'obs_size_y': 256,
    'show_sensor': False,
    'wayp_sampling_resolution': 10,
    'wayp_angle_bias': 5,
  }




  model = train_agent(params=params, total_timesteps=300000)
  model.save("carla_ppo_agent_final")


  env = gym.make('ppo-v0', params=params)
  model = PPO.load("carla_ppo_agent_final", env=env, device="cuda")
  obs, _ = env.reset()
  while True:
    steer = model.predict(obs)[0]
    obs, reward, _, done, _ = env.step(steer)
    if done:
      obs, _ = env.reset()



if __name__ == '__main__':
  main()

