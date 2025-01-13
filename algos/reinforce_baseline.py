# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "MountainCarContinuous-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_learning_rate: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 1.
    """the lambda for the general advantage estimation (should be 1. for REINFORCE)"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    value_coef: float = 0.5
    """coefficient of the value function loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)
    
    @torch.no_grad()
    def get_value_no_backprop(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_space, action_space):
        self.observation_dims = observation_space.shape
        self.action_dims = action_space.shape
        self.observations = torch.zeros((num_steps, num_envs) + self.observation_dims).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + self.action_dims).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.logprobs = torch.zeros_like(self.rewards).to(device)
        self.dones = torch.zeros_like(self.rewards).to(device)
        self.values = torch.zeros_like(self.rewards).to(device)
        self.advantages = torch.zeros_like(self.rewards).to(device)
        self.returns = torch.zeros_like(self.rewards).to(device)
    
    def get_minibatch_iterator(self, minibatch_size):
        return DataLoader(
            dataset=TensorDataset(
                self.observations.reshape((-1,) + self.observation_dims),
                self.actions.reshape((-1,) + self.action_dims),
                self.logprobs.reshape(-1),
                self.values.reshape(-1),
                self.returns.reshape(-1),
                self.advantages.reshape(-1)
            ),
            batch_size=minibatch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    buffer = RolloutBuffer(num_steps=args.num_steps, num_envs=args.num_envs, 
                           observation_space=envs.single_observation_space, action_space=envs.single_action_space)

    global_step = 0
    start_time = time.time()
    observation, _ = envs.reset(seed=args.seed)
    observation = torch.Tensor(observation).to(device)
    done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        
        if args.anneal_learning_rate:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            current_learning_rate = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = current_learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            buffer.observations[step] = observation
            buffer.dones[step] = done

            # action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(observation)
                buffer.values[step] = value.flatten()
            buffer.actions[step] = action
            buffer.logprobs[step] = logprob

            # execute the game and log data.
            next_observation, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
            observation, done = torch.Tensor(next_observation).to(device), torch.Tensor(done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("episode/mean_return", info["episode"]["r"], global_step)
                        writer.add_scalar("episode/mean_length", info["episode"]["l"], global_step)
            elif "episode" in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r'].mean().mean()}")
                writer.add_scalar("episode/mean_return", infos["episode"]["r"].mean(), global_step)
                writer.add_scalar("episode/mean_length", infos["episode"]["l"].mean(), global_step)

        # GENERALIZED ADVANTAGE ESTIMATION (GAE)
        last_value = agent.get_value_no_backprop(observation).reshape(1, -1)
        last_advantage = 0

        for t in reversed(range(args.num_steps)):

            # bootstrap value if not done
            if t == args.num_steps - 1:
                not_done = 1.0 - done
                next_value = last_value * not_done
            else:
                not_done = 1.0 - buffer.dones[t + 1]
                next_value = buffer.values[t + 1] * not_done

            delta = buffer.rewards[t] + args.gamma * next_value - buffer.values[t]

            buffer.advantages[t] = last_advantage = delta + args.gamma * args.gae_lambda * not_done * last_advantage

        buffer.returns = buffer.advantages + buffer.values

        # Optimizing the policy and value networks
        clip_fractions = []
        for epoch in range(args.update_epochs):
            for (
                observations_minibatch,
                actions_minibatch,
                log_probs_minibatch,
                values_minibatch,
                returns_minibatch,
                advantages_minibatch,
            ) in buffer.get_minibatch_iterator(args.minibatch_size):

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(observations_minibatch, actions_minibatch)
                logratio = newlogprob - log_probs_minibatch
                ratio = logratio.exp()

                # Policy loss
                pg_loss = (-returns_minibatch * ratio).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns_minibatch) ** 2).mean()

                loss = pg_loss + v_loss * args.value_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = buffer.values.cpu().numpy(), buffer.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/value_loss", v_loss.item(), global_step)
        writer.add_scalar("train/policy_loss", loss.item(), global_step)
        writer.add_scalar("train/clip_fraction", np.mean(clip_fractions), global_step)
        writer.add_scalar("train/explained_variance", explained_var, global_step)
        writer.add_scalar("train/steps_per_second", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from .evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/mean_episode_return", episodic_return, idx)

    envs.close()
    writer.close()
