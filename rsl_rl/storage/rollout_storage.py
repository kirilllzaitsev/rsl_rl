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

import argparse

import numpy as np
import torch
from attrdict import AttrDict
from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *privileged_obs_shape,
                device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        num_mini_batches=4, len(self.observations)=24
        batch_size=512*24=12288
        => mini_batch_size=12288//4=3072
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (
                    obs_batch,
                    critic_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        None,
                        None,
                    ),
                    None,
                )

    def mini_batch_generator_dreamer(self, num_mini_batches, num_epochs=8):
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in self.mini_batch_generator(num_mini_batches, num_epochs):
            yield (
                obs_batch.reshape(self.num_transitions_per_env, -1, *self.obs_shape),
                critic_obs_batch.reshape(
                    self.num_transitions_per_env,
                    -1,
                    *(
                        self.privileged_obs_shape
                        if self.privileged_observations is not None
                        else self.obs_shape
                    )
                ),
                actions_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                target_values_batch.reshape(self.num_transitions_per_env, -1, 1),
                advantages_batch.reshape(self.num_transitions_per_env, -1, 1),
                returns_batch.reshape(self.num_transitions_per_env, -1, 1),
                old_actions_log_prob_batch.reshape(self.num_transitions_per_env, -1, 1),
                old_mu_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                old_sigma_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                hid_states_batch,
                masks_batch,
            )

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.privileged_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[
                    :, first_traj:last_traj
                ]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj

    def reccurent_mini_batch_generator_dreamer(self, num_mini_batches, num_epochs=8):
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in self.reccurent_mini_batch_generator(num_mini_batches, num_epochs):
            yield (
                obs_batch.reshape(self.num_transitions_per_env, -1, *self.obs_shape),
                critic_obs_batch.reshape(
                    self.num_transitions_per_env,
                    -1,
                    *(
                        self.privileged_obs_shape
                        if self.privileged_observations is not None
                        else self.obs_shape
                    )
                ),
                actions_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                target_values_batch.reshape(self.num_transitions_per_env, -1, 1),
                advantages_batch.reshape(self.num_transitions_per_env, -1, 1),
                returns_batch.reshape(self.num_transitions_per_env, -1, 1),
                old_actions_log_prob_batch.reshape(self.num_transitions_per_env, -1, 1),
                old_mu_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                old_sigma_batch.reshape(
                    self.num_transitions_per_env, -1, *self.actions_shape
                ),
                hid_states_batch,
                masks_batch,
            )


class ReplayBuffer(object):
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        observation_shape,
        action_size,
        device,
    ):
        self.device = device
        self.capacity = num_transitions_per_env

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.observation_shape = observation_shape
        self.observation = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, *observation_shape),
            device=self.device,
        )
        self.next_observation = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, *observation_shape),
            device=self.device,
        )
        self.action = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, action_size),
            device=self.device,
        )
        self.reward = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.done = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, 1), device=self.device
        )

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done):
        self.observation[self.buffer_index] = observation
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward.unsqueeze(-1)
        self.next_observation[self.buffer_index] = next_observation
        self.done[self.buffer_index] = done.unsqueeze(-1)

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    # TODO: should be sampled at random from the buffer (which by itself contains more than the specified number of transitions. it should represent the whole history or at least approximate it)
    def sample(self, num_mini_batches):
        # just a copy of the PPO mini_batch_generator
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observation.flatten(0, 1)
        next_observations = self.next_observation.flatten(0, 1)
        actions = self.action.flatten(0, 1)
        rewards = self.reward.flatten(0, 1)
        dones = self.done.flatten(0, 1)

        i = 0
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        batch_idx = indices[start:end]

        obs_batch = observations[batch_idx]
        next_obs_batch = next_observations[batch_idx]
        actions_batch = actions[batch_idx]
        rewards_batch = rewards[batch_idx]
        dones_batch = dones[batch_idx]

        batch = {
            "observation": obs_batch,
            "action": actions_batch,
            "reward": rewards_batch,
            "next_observation": next_obs_batch,
            "done": dones_batch,
        }
        batch = AttrDict(batch)

        return batch
