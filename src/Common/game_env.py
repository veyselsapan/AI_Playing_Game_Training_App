# src/Common/game_env.py

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import cv2
import numpy as np

def create_env(environment_name='Breakout-v4', n_envs=4, n_stack=4, seed=0):
    env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    env.metadata['render_fps'] = 30
    env = VecFrameStack(env, n_stack=n_stack)
    return env

def create_eval_env(environment_name='Breakout-v4', n_envs=1, n_stack=4, seed=0):
    eval_env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    eval_env.metadata['render_fps'] = 30
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    return eval_env

def record_env(env, model, video_path, video_fps=30, recording_time=60):
    env.reset()
    obs = env.reset()
    done = np.array([False] * env.num_envs)
    step = 0
    max_steps = recording_time * video_fps

    frame = env.render(mode='rgb_array')
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if done.any():
            obs = env.reset()

        step += 1

    out.release()
    env.close()

def create_single_env(environment_name='Breakout-v4', n_envs=1, n_stack=1, seed=0):
    env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    env.metadata['render_fps'] = 30
    if n_stack > 0:
        env = VecFrameStack(env, n_stack=n_stack)
    return env

def record_fsm_env(env, fsm_agent, video_path, video_fps=30, recording_time=60):
    env.reset()
    state = env.reset()
    done = False
    step = 0
    max_steps = recording_time * video_fps

    frame = env.render(mode='rgb_array')
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))

    while step < max_steps:
        action = fsm_agent.act(frame)
        state, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if done:
            state = env.reset()

        step += 1

    out.release()
    env.close()
