import sys
import time
sys.path.append('/Users/seominseok/content/highway-env')
sys.path.append('/Users/seominseok/content/rl-agents')

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import logging
from pprint import pprint

import highway_env
highway_env.register_highway_envs()

from rl_agents.trainer.evaluation  import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

logger = logging.getLogger(__name__)

TRAIN = False
NUM_EPISODES = 5000

from os import chdir
chdir('/Users/seominseok/content/rl-agents/scripts')

env_config = 'configs/IntersectionMergeEnv/env.json'
agent_config = 'configs/IntersectionMergeEnv/agents/DQNAgent/ego_attention_2h.json'

if __name__ == '__main__':
    start_time = time.time()

    env = load_environment(env_config)
    agent = load_agent(agent_config, env)

    if TRAIN:
        evaluation = Evaluation(env, agent, num_episodes=NUM_EPISODES, display_env=False, display_agent=False)
        evaluation.train()
    
    else:
        agent.load('C:/Users/Seominseok/content/rl-agents/scripts/out/IntersectionMergeEnv/DQNAgent/run_20231012-163046_22216/checkpoint-best.tar')
        evaluation = Evaluation(env, agent, num_episodes=NUM_EPISODES, display_env=False, display_agent=True)

        env = evaluation.env
        print('env configuration')
        pprint(env.default_config())
        env = RecordVideo(env, video_folder="/Users/seominseok/content/videos", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": 15})

        for videos in range(10):
            done = truncated = False
            obs, info = env.reset()
            while not(done or truncated):
                action = evaluation.agent.act(obs)
                osb, reward, done, truncated, info = env.step(action)
                env.render()
        env.close()

    end_time = time.time()

    # 초를 시, 분, 초로 변환
    hours, remainder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"프로그램 실행 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
