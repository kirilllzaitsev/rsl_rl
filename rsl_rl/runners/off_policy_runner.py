# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import statistics
import time
import typing as t
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO, DayDreamer
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from torch.utils.tensorboard import SummaryWriter


class OffPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        # for now, the dreamer config is embedded in the daydreamer.py
        self.alg_cfg = train_cfg["algorithm"] if "algorithm" in train_cfg else {}
        self.policy_cfg = train_cfg["policy"] if "policy" in train_cfg else {}
        self.device = device
        self.env = env
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: DayDreamer = alg_class(device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # get first portion of proprioceptive data
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor.train()
        self.alg.critic.train()
        self.alg.rssm.train()
        self.alg.encoder.train()
        self.alg.decoder.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        train_metrics = defaultdict(list)

        # initial collection
        interaction_info = self.alg.environment_interaction(
            self.env,
            self.num_steps_per_env,
            num_envs=self.env.num_envs,
        )

        rewbuffer.extend(interaction_info["rewbuffer"])
        lenbuffer.extend(interaction_info["lenbuffer"])
        ep_infos.extend(interaction_info["ep_infos"])

        obs = self.alg.buffer.observation
        rewards = self.alg.buffer.reward
        actions = self.alg.buffer.action

        if self.log_dir is not None:
            print(f"{Path(self.log_dir).name=}")
        # main loop
        for it in range(self.current_learning_iteration, tot_iter):

            start = time.time()
            iter_metrics = self.alg.update()
            for k, v in iter_metrics.items():
                for loss_name, loss_values in v.items():
                    train_metrics[loss_name].append(np.mean(loss_values))
            stop = time.time()
            learn_time = stop - start
            # print(f"{learn_time=}")

            # tmp: without new samples, prove that it can overfit to what's in the buffer
            if True or it == self.current_learning_iteration:
                start = time.time()
                interaction_info = self.alg.environment_interaction(
                    self.env, self.num_steps_per_env, num_envs=self.env.num_envs
                )
                rewbuffer.extend(interaction_info["rewbuffer"])
                lenbuffer.extend(interaction_info["lenbuffer"])
                ep_infos.extend(interaction_info["ep_infos"])
                stop = time.time()
                collection_time = stop - start
                # print(f"{collection_time=}")

            self.log(locals(), train_metrics=train_metrics)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            print(f"{Path(self.log_dir).name=}")
            self.save(
                os.path.join(
                    self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
                )
            )

    def log(self, locs, width=80, pad=35, train_metrics=None):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        train_metrics = {} if train_metrics is None else train_metrics

        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if self.log_dir is not None:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if self.log_dir is not None:
            self.writer.add_scalar(
                "Loss/learning_rate", self.alg.learning_rate, locs["it"]
            )
            # self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar(
                "Perf/collection time", locs["collection_time"], locs["it"]
            )
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    statistics.mean(locs["lenbuffer"]),
                    locs["it"],
                )
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
        if len(locs["lenbuffer"]) > 0:
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""

        for k, v in train_metrics.items():
            log_string += f"""{f'Mean {k}:':>{pad}} {statistics.mean(v):.4f}\n"""

        for k, v in train_metrics.items():
            prefix = "Train" if "loss" not in k.lower() else "Loss"
            if self.log_dir is not None:
                self.writer.add_scalar(f"{prefix}/{k}", statistics.mean(v), locs["it"])

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "transition_model_state_dict": self.alg.rssm.transition_model.state_dict(),
                "reward_predictor_state_dict": self.alg.reward_predictor.state_dict(),
                "recurrent_state_model_state_dict": self.alg.rssm.recurrent_model.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "decoder_state_dict": self.alg.decoder.state_dict(),
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "model_optimizer_state_dict": self.alg.model_optimizer.state_dict(),
                "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.rssm.transition_model.load_state_dict(
            loaded_dict["transition_model_state_dict"]
        )
        self.alg.reward_predictor.load_state_dict(
            loaded_dict["reward_predictor_state_dict"]
        )
        self.alg.rssm.recurrent_model.load_state_dict(
            loaded_dict["recurrent_state_model_state_dict"]
        )
        self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
        self.alg.decoder.load_state_dict(loaded_dict["decoder_state_dict"])
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
        if load_optimizer:
            self.alg.model_optimizer.load_state_dict(
                loaded_dict["model_optimizer_state_dict"]
            )
            self.alg.actor_optimizer.load_state_dict(
                loaded_dict["actor_optimizer_state_dict"]
            )
            self.alg.critic_optimizer.load_state_dict(
                loaded_dict["critic_optimizer_state_dict"]
            )
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    @torch.no_grad()
    def do_inference(self, obs, num_actions, device=None):
        device = device if device is not None else self.device
        batch_size = obs.shape[0]
        if not hasattr(self, "did_one_iter"):
            self.did_one_iter = True
            # TODO: do init fresh once or at every subsequent inference step?
            self.prev_action = torch.zeros(batch_size, num_actions).to(device)
            _, self.prev_deterministic = self.alg.rssm.recurrent_model_input_init(
                batch_size
            )

        embedded_observation = self.alg.encoder(obs.to(device))
        _, posterior = self.alg.rssm.representation_model(
            embedded_observation, self.prev_deterministic
        )

        deterministic = self.alg.rssm.recurrent_model(
            posterior, self.prev_action, self.prev_deterministic
        )
        embedded_observation = embedded_observation.reshape(batch_size, -1)
        _, posterior = self.alg.rssm.representation_model(
            embedded_observation, deterministic
        )
        action = self.alg.actor(posterior, deterministic).detach()

        self.prev_deterministic = deterministic
        self.prev_action = action

        return action

    def get_inference_policy(self, device=None):
        self.alg.actor.eval()
        if device is not None:
            self.alg.actor.to(device)
        return self.do_inference
