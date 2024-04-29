import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.models.rssm import RSSM
from dreamerv2.training.trainer import Trainer as DreamerTrainer
from dreamerv2.utils.algorithm import compute_return
from dreamerv2.utils.module import FreezeParameters, get_parameters
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class DreamerConfig:
    obs_shape: tuple = (1, 48)
    action_size: int = 12
    rssm_type: str = "continuous"
    rssm_info: dict = {
        "deter_size": 256,
        "stoch_size": 32,
        "min_std": 0.01,
    }  # ?
    embedding_size: int = 64
    rssm_node_size: int = 256  # ?
    actor: dict = {
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "one_hot",
    }
    expl: dict = {
        "train_noise": 0.3,
        "eval_noise": 0.0,
        "expl_min": 0.1,
        "expl_decay": 1000,
        "expl_type": "epsilon_greedy",
    }
    reward: dict = {
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "normal",
    }
    critic: dict = {
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "normal",
    }
    discount: dict = {
        "use": True,
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "binary",
    }
    pixel: bool = False
    obs_encoder: dict = {
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "normal",
    }
    obs_decoder: dict = {
        "layers": 2,
        "node_size": 256,
        "activation": nn.ReLU,
        "dist": "normal",
    }

    actor_grad: str = "reinforce"
    actor_entropy_scale: float = 0.01
    lambda_: float = 0.95
    grad_clip_norm: float = 100.0

    lr: dict = {
        "model": 1e-3,
        "actor": 1e-4,
        "critic": 1e-4,
    }

    horizon: int = 10
    discount_: float = 0.99


class DayDreamer:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # config = MinAtarConfig(
        #     env=env_name,
        #     obs_shape=obs_shape,
        #     action_size=action_size,
        #     obs_dtype=obs_dtype,
        #     action_dtype=action_dtype,
        #     seq_len=seq_len,
        #     batch_size=batch_size,
        #     model_dir=model_dir,
        # )
        self.dreamer_config = DreamerConfig()
        self._model_initialize(self.dreamer_config)
        self._optim_initialize(self.dreamer_config)
        self.kl_info = dict(
            use_kl_balance=True,
            kl_balance_scale=0.5,
            use_free_nats=True,
            free_nats=3,
        )
        self.loss_scale = {"kl": 1.0, "discount": 0.1}
        self.batch_size = None
        self.seq_len = None

    def _model_initialize(self, config: DreamerConfig):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info["deter_size"]
        if config.rssm_type == "continuous":
            stoch_size = config.rssm_info["stoch_size"]
        elif config.rssm_type == "discrete":
            category_size = config.rssm_info["category_size"]
            class_size = config.rssm_info["class_size"]
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        self.RSSM = RSSM(
            action_size,
            rssm_node_size,
            embedding_size,
            self.device,
            config.rssm_type,
            config.rssm_info,
        ).to(self.device)
        # actor model. not sure about it being discrete
        self.ActionModel = DiscreteActionModel(
            action_size,
            deter_size,
            stoch_size,
            embedding_size,
            config.actor,
            config.expl,
        ).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(
            self.device
        )
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(
            self.device
        )
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(
            self.device
        )
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())

        if config.discount["use"]:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(
                self.device
            )
        if config.pixel:
            self.ObsEncoder = ObsEncoder(
                obs_shape, embedding_size, config.obs_encoder
            ).to(self.device)
            self.ObsDecoder = ObsDecoder(
                obs_shape, modelstate_size, config.obs_decoder
            ).to(self.device)
        else:
            self.ObsEncoder = DenseModel(
                (embedding_size,), int(np.prod(obs_shape)), config.obs_encoder
            ).to(self.device)
            self.ObsDecoder = DenseModel(
                obs_shape, modelstate_size, config.obs_decoder
            ).to(self.device)

    def representation_loss(self, obs, actions, rewards, nonterms):

        embed = self.ObsEncoder(obs)  # t to t+seq_len
        embed = embed.sample()

        batch_size = obs.shape[1]
        if self.batch_size is None:
            self.batch_size = batch_size
        prev_rssm_state = self.RSSM._init_rssm_state(batch_size)

        seq_len = obs.shape[0]
        if self.seq_len is None:
            self.seq_len = seq_len
        prior, posterior = self.RSSM.rollout_observation(
            seq_len, embed, actions, nonterms, prev_rssm_state
        )

        post_modelstate = self.RSSM.get_model_state(posterior)  # t to t+seq_len
        obs_dist = self.ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        reward_dist = self.RewardDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        pcont_dist = self.DiscountModel(post_modelstate[:-1])  # t to t+seq_len-1

        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = (
            self.loss_scale["kl"] * div
            + reward_loss
            + obs_loss
            + self.loss_scale["discount"] * pcont_loss
        )
        return (
            model_loss,
            div,
            obs_loss,
            reward_loss,
            pcont_loss,
            prior_dist,
            post_dist,
            posterior,
        )

    def _optim_initialize(self, config):
        model_lr = config.lr["model"]
        actor_lr = config.lr["actor"]
        value_lr = config.lr["critic"]
        self.world_list = [
            self.ObsEncoder,
            self.RSSM,
            self.RewardDecoder,
            self.ObsDecoder,
            self.DiscountModel,
        ]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _actor_loss(
        self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
    ):

        lambda_returns = compute_return(
            imag_reward[:-1],
            imag_value[:-1],
            discount_arr[:-1],
            bootstrap=imag_value[-1],
            lambda_=self.dreamer_config.lambda_,
        )

        if self.dreamer_config.actor_grad == "reinforce":
            advantage = (lambda_returns - imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.dreamer_config.actor_grad == "dynamics":
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(
            torch.mean(
                discount
                * (objective + self.dreamer_config.actor_entropy_scale * policy_entropy),
                dim=1,
            )
        )
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates)
        value_loss = -torch.mean(
            value_discount * value_dist.log_prob(value_target).unsqueeze(-1)
        )
        return value_loss

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info["use_kl_balance"]:
            alpha = self.kl_info["kl_balance_scale"]
            kl_lhs = torch.mean(
                torch.distributions.kl.kl_divergence(
                    self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist
                )
            )
            kl_rhs = torch.mean(
                torch.distributions.kl.kl_divergence(
                    post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))
                )
            )
            if self.kl_info["use_free_nats"]:
                free_nats = self.kl_info["free_nats"]
                kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs

        else:
            kl_loss = torch.mean(
                torch.distributions.kl.kl_divergence(post_dist, prior_dist)
            )
            if self.kl_info["use_free_nats"]:
                free_nats = self.kl_info["free_nats"]
                kl_loss = torch.max(
                    kl_loss, kl_loss.new_full(kl_loss.size(), free_nats)
                )
        return prior_dist, post_dist, kl_loss

    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, train_metrics) -> dict:
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator_dreamer(
                self.num_mini_batches, self.num_learning_epochs
            )
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
        ) in generator:
            obs = obs_batch
            rewards = returns_batch
            actions = actions_batch
            terms = torch.zeros((obs.shape[0], obs.shape[1], 1)).to(self.device)
            # obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            actions = torch.tensor(actions, dtype=torch.float32).to(
                self.device
            )  # t-1, t+seq_len-1
            rewards = (
                torch.tensor(rewards, dtype=torch.float32).to(self.device)
            )  # t-1 to t+seq_len-1
            nonterms = (
                torch.tensor(1 - terms, dtype=torch.float32)
                .to(self.device)
                
            )  # t-1 to t+seq_len-1

            (
                model_loss,
                kl_loss,
                obs_loss,
                reward_loss,
                pcont_loss,
                prior_dist,
                post_dist,
                posterior,
            ) = self.representation_loss(obs, actions, rewards, nonterms)

            self.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(
                get_parameters(self.world_list), self.dreamer_config.grad_clip_norm
            )
            self.model_optimizer.step()

            actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                get_parameters(self.actor_list), self.dreamer_config.grad_clip_norm
            )
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                get_parameters(self.value_list), self.dreamer_config.grad_clip_norm
            )

            self.actor_optimizer.step()
            self.value_optimizer.step()

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info["mean_targ"])
            min_targ.append(target_info["min_targ"])
            max_targ.append(target_info["max_targ"])
            std_targ.append(target_info["std_targ"])

        num_updates = self.num_learning_epochs * self.num_mini_batches
        train_metrics["model_loss"] = np.mean(model_l)
        train_metrics["kl_loss"] = np.mean(kl_l)
        train_metrics["reward_loss"] = np.mean(reward_l)
        train_metrics["obs_loss"] = np.mean(obs_l)
        train_metrics["value_loss"] = np.mean(value_l)
        train_metrics["actor_loss"] = np.mean(actor_l)
        train_metrics["prior_entropy"] = np.mean(prior_ent_l)
        train_metrics["posterior_entropy"] = np.mean(post_ent_l)
        train_metrics["pcont_loss"] = np.mean(pcont_l)
        train_metrics["mean_targ"] = np.mean(mean_targ)
        train_metrics["min_targ"] = np.mean(min_targ)
        train_metrics["max_targ"] = np.mean(max_targ)
        train_metrics["std_targ"] = np.mean(std_targ)

        self.storage.clear()

        return train_metrics

    def actorcritc_loss(self, posterior):
        assert self.batch_size is not None, "batch_size is not set"
        assert self.seq_len is not None, "seq_len is not set"

        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(
                self.RSSM.rssm_seq_to_batch(
                    posterior, self.batch_size, self.seq_len - 1
                )
            )

        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = (
                self.RSSM.rollout_imagination(
                    self.dreamer_config.horizon, self.ActionModel, batched_posterior
                )
            )

        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(
            self.world_list
            + self.value_list
            + [self.TargetValueModel]
            + [self.DiscountModel]
        ):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.dreamer_config.discount_ * torch.round(
                discount_dist.base_dist.probs
            )  # mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self._actor_loss(
            imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
        )
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item()
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            "min_targ": min_targ,
            "max_targ": max_targ,
            "std_targ": std_targ,
            "mean_targ": mean_targ,
        }

        return actor_loss, value_loss, target_info

    def update_old(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
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
        ) in generator:

            self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
