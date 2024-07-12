import argparse
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from attrdict import AttrDict
from dreamer.modules.decoder import Decoder
from dreamer.modules.encoder import Encoder
from dreamer.modules.model import RSSM, ContinueModel, RewardModel
from dreamer.utils.utils import (
    DynamicInfos,
    compute_lambda_values,
    create_normal_dist,
    load_config,
)
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.models.rssm import RSSM as Dreamerv2RSSM
from dreamerv2.training.trainer import Trainer as DreamerTrainer
from dreamerv2.utils.algorithm import compute_return
from dreamerv2.utils.module import FreezeParameters, get_parameters
from rsl_rl.algorithms.dreamer.actor import Actor
from rsl_rl.algorithms.dreamer.critic import Critic
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.rollout_storage import ReplayBuffer
from tqdm.auto import tqdm

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DreamerConfig:
    obs_shape: tuple = (1, 48)
    action_size: int = 12
    pixel: bool = False
    action_repeat: int = 1

    buffer_capacity: int = int(
        1e5
    )  # the more the better, but have to scale network capacities accordingly
    action_dtype: np.dtype = np.float32

    # rssm_type: str = "continuous"
    # rssm_info: dict = {
    #     "deter_size": 256,
    #     "stoch_size": 32,
    #     "min_std": 0.01,
    # }  # ?
    # rssm_type: str = "discrete"
    rssm_info: dict = {
        "deter_size": 200,
        "stoch_size": 50,
        "class_size": 20,
        "category_size": 20,
        "min_std": 0.1,
    }
    embedding_size: int = 200
    # rssm_node_size: int = 128

    grad_clip_norm: float = 100.0
    grad_norm_type: int = 2
    discount_: float = 0.99
    lambda_: float = 0.95
    # horizon: int = 10

    # lr: dict = {"model": 2e-4, "actor": 4e-5, "critic": 1e-4}
    loss_scale: dict = {"kl": 1, "reward": 1.0, "discount": 5.0}
    kl: dict = {
        # "use_kl_balance": True,
        # "kl_balance_scale": 0.8,
        # "use_free_nats": False,
        # "free_nats": 0.0,
        "use_free_nats": True,
        "free_nats": 3.0,
    }

    # ?
    # use_slow_target: float = True
    # slow_target_update: int = 100
    # slow_target_fraction: float = 1.00

    actor: dict = {
        "layers": 2,  # same as node_size but the effect is less pronounced
        "node_size": 256,  # higher -> positive impact on actor loss, negative impact on value loss
        "dist": "one_hot",  # not used
        "min_std": 1e-4,
        "init_std": 3.0,  # important. has to be high with tanh transform
        "mean_scale": 1.0,  # not important
        "activation": nn.ELU,
    }
    # expl: dict = {
    #     "train_noise": 0.4,
    #     "eval_noise": 0.0,
    #     "expl_min": 0.05,
    #     "expl_decay": 7000.0,
    #     "expl_type": "epsilon_greedy",
    # }
    critic: dict = {
        "layers": 2,  # ? higher -> negative impact on both actor and value loss. need to increase their capacities as well (but why higher node_size helps)?!
        "node_size": 256,  # higher -> positive impact on both actor and value loss
        "dist": "normal",
        "activation": nn.ELU,
    }
    # actor_grad: str = "reinforce"
    # actor_grad_mix: int = 0.0
    # actor_entropy_scale: float = 1e-3

    obs_encoder: dict = {
        "layers": 2,
        "node_size": 256,
        "dist": None,
        "activation": nn.ELU,
        # "kernel": 2,
        # "depth": 16,
    }
    obs_decoder: dict = {
        "layers": 2,
        "node_size": 256,
        "dist": "normal",
        "activation": nn.ELU,
        # "kernel": 2,
        # "depth": 16,
    }
    reward: dict = {
        "layers": 2,
        "node_size": 400,
        "dist": "normal",
        "activation": nn.ELU,
    }
    # continue_: dict = {
    #     "layers": 2,
    #     "node_size": 128,
    #     "dist": "normal",
    #     "activation": nn.ELU,
    # }
    # discount: dict = {
    #     "layers": 2,
    #     "node_size": 128,
    #     "dist": "binary",
    #     "activation": nn.ELU,
    #     "use": True,
    # }

    use_continue_flag: bool = False  # NN to predict end of episode
    # learning rates are crucial for convergence and overall performance
    model_learning_rate: float = 0.0008
    actor_learning_rate: float = 0.0001
    critic_learning_rate: float = 0.0003

    collect_interval: int = 30
    horizon_length: int = 10


class DayDreamer:

    def __init__(
        self,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device="cpu",
    ):

        self.device = device

        self.learning_rate = learning_rate

        # PPO components
        self.storage = None  # initialized later

        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        self.dreamer_config = DreamerConfig()
        self.batch_size = None
        self.seq_len = None
        self.kl_info = self.dreamer_config.kl

        # following SimpleDreamer
        self.action_size = self.dreamer_config.action_size
        self.discrete_action_bool = False
        self.num_total_episodes = 0

        # creates latent of an observation
        self.encoder = DenseModel(
            (self.dreamer_config.embedding_size,),
            int(np.prod(self.dreamer_config.obs_shape)),
            self.dreamer_config.obs_encoder,
        ).to(self.device)
        # modelstate_size = stoch_size + deter_size
        self.modelstate_size = (
            self.dreamer_config.rssm_info["stoch_size"]
            + self.dreamer_config.rssm_info["deter_size"]
        )
        # reconstructs observation from latent
        self.decoder = DenseModel(
            self.dreamer_config.obs_shape,
            self.modelstate_size,
            self.dreamer_config.obs_decoder,
        ).to(self.device)

        self.rssm = RSSM(
            self.action_size,
            stochastic_size=self.dreamer_config.rssm_info["stoch_size"],
            deterministic_size=self.dreamer_config.rssm_info["deter_size"],
            device=self.device,
            recurrent_model_config={
                "hidden_size": 200,
                "activation": "ELU",
            },
            transition_model_config={
                "hidden_size": 200,
                "num_layers": 2,
                "activation": "ELU",
                "min_std": 0.1,
            },
            representation_model_config={
                "embedded_state_size": self.dreamer_config.embedding_size,
                "hidden_size": 200,
                "num_layers": 2,
                "activation": "ELU",
                "min_std": 0.1,
            },
        ).to(self.device)
        self.reward_predictor = RewardModel(
            stochastic_size=self.dreamer_config.rssm_info["stoch_size"],
            deterministic_size=self.dreamer_config.rssm_info["deter_size"],
            hidden_size=self.dreamer_config.reward["node_size"],
            num_layers=self.dreamer_config.reward["layers"],
            activation=self.dreamer_config.reward["activation"],
        ).to(self.device)
        if self.dreamer_config.use_continue_flag:
            self.continue_predictor = ContinueModel(
                stochastic_size=self.dreamer_config.rssm_info["stoch_size"],
                deterministic_size=self.dreamer_config.rssm_info["deter_size"],
                hidden_size=self.dreamer_config.continue_["node_size"],
                num_layers=self.dreamer_config.continue_["layers"],
                activation=self.dreamer_config.continue_["activation"],
            ).to(self.device)
        self.actor = Actor(
            self.discrete_action_bool,
            self.action_size,
            stochastic_size=self.dreamer_config.rssm_info["stoch_size"],
            deterministic_size=self.dreamer_config.rssm_info["deter_size"],
            hidden_size=self.dreamer_config.actor["node_size"],
            num_layers=self.dreamer_config.actor["layers"],
            activation=self.dreamer_config.actor["activation"],
            mean_scale=self.dreamer_config.actor["mean_scale"],
            init_std=self.dreamer_config.actor["init_std"],
            min_std=self.dreamer_config.actor["min_std"],
        ).to(self.device)
        self.critic = Critic(
            stochastic_size=self.dreamer_config.rssm_info["stoch_size"],
            deterministic_size=self.dreamer_config.rssm_info["deter_size"],
            hidden_size=self.dreamer_config.critic["node_size"],
            num_layers=self.dreamer_config.critic["layers"],
            activation=self.dreamer_config.critic["activation"],
        ).to(self.device)

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.continue_criterion = nn.BCELoss()

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.dreamer_config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.dreamer_config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.dreamer_config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.dreamer_config.critic_learning_rate,
        )

        self.stochastic_size = self.dreamer_config.rssm_info["stoch_size"]
        self.deterministic_size = self.dreamer_config.rssm_info["deter_size"]
        self.horizon_length = self.dreamer_config.horizon_length

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # actor_obs_shape vs critic_obs_shape?
        self.num_transitions_per_env = num_transitions_per_env
        self.buffer = ReplayBuffer(
            capacity=self.dreamer_config.buffer_capacity,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            observation_size=int(np.prod(self.dreamer_config.obs_shape)),
            action_size=self.action_size,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        # returns actions
        raise NotImplementedError

    def process_env_step(self, rewards, dones, infos):
        # add data to buffer
        raise NotImplementedError

    def update(self) -> dict:
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        dynamic learning + behavior learning
        """
        behavior_losses = defaultdict(list)
        dynamic_losses = defaultdict(list)
        for collect_interval in tqdm(range(self.dreamer_config.collect_interval)):
            # TODO: check if this is correct
            data = self.buffer.sample(num_consecutive_trajs=1, envs_per_batch=-1)
            dynamic_learning_res = self.dynamic_learning(data)
            posteriors = dynamic_learning_res["posteriors"]
            deterministics = dynamic_learning_res["deterministics"]
            dynamic_losses_batch = dynamic_learning_res["losses"]
            behavior_losses_batch = self.behavior_learning(posteriors, deterministics)
            for k, v in dynamic_losses_batch.items():
                dynamic_losses[k].append(v)
            for k, v in behavior_losses_batch.items():
                behavior_losses[k].append(v)
        return {
            "dynamic_loss": dynamic_losses,
            "behavior_loss": behavior_losses,
        }

    def dynamic_learning(self, data):
        """
        data.action: (batch, seq_len, action_size)
        """
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        # self.num_transitions_per_env != seq_len which could be arbitrarily short/long
        # start from 1 because we are already given the first observation
        for t in range(1, self.num_transitions_per_env):
            if t == 1:
                # get distribution of actions
                action = data.action[:, t - 1]
                # print diff stats
                logger.debug(
                    f"action: {action.mean()}, {action.std()} {action.min()}, {action.max()}"
                )
                # same for observation
                obs = data.observation[:, t]
                logger.debug(
                    f"obs: {obs.mean()}, {obs.std()} {obs.min()}, {obs.max()}"
                )
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        losses = self._model_update(data, infos)
        # return infos.posteriors.detach(), infos.deterministics.detach()
        return {
            "posteriors": infos.posteriors.detach(),
            "deterministics": infos.deterministics.detach(),
            "losses": losses,
        }

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            torch.cat(
                (
                    posterior_info.deterministics,
                    posterior_info.posteriors,
                ),
                dim=-1,
            )
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.dreamer_config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.dreamer_config.kl["free_nats"]).to(self.device),
            kl_divergence_loss,
        )
        model_loss = (
            self.dreamer_config.loss_scale["kl"] * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.dreamer_config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()

        # for i in range(len(self.model_params)):
        #     logger.debug(f"{self.model_params[i].grad.norm()}")
        logger.debug(f"encoder")
        for i, p in enumerate(self.encoder.parameters()):
            # print name and norm
            logger.debug(f"{i} : {p.grad.norm()}")
        logger.debug(f"decoder")
        for i, p in enumerate(self.decoder.parameters()):
            # print name and norm
            logger.debug(f"{i} : {p.grad.norm()}")
        counter = 0
        for i, p in enumerate(self.rssm.transition_model.parameters()):
            if i == 0:
                logger.debug(f"transition_model")
            logger.debug(f"{i} : {p.grad.norm()}")
            counter += 1
        for i, p in enumerate(self.rssm.representation_model.parameters()):
            if i == 0:
                logger.debug(f"representation_model")
            logger.debug(f"{i} : {p.grad.norm()}")
            counter += 1
        for i, p in enumerate(self.rssm.representation_model.parameters()):
            if i == 0:
                logger.debug(f"representation_model")
            logger.debug(f"{i} : {p.grad.norm()}")
            counter += 1
        # logger.debug(f"{counter=} layers in three rssm models")
        logger.debug(f"reward")
        for i, p in enumerate(self.reward_predictor.parameters()):
            # print name and norm
            logger.debug(f"{i} : {p.grad.norm()}")
        if self.dreamer_config.use_continue_flag:
            logger.debug(f"continue")
            for i, p in enumerate(self.continue_predictor.parameters()):
                # print name and norm
                logger.debug(f"{i} : {p.grad.norm()}")

        nn.utils.clip_grad_norm_(
            self.model_params,
            self.dreamer_config.grad_clip_norm,
            norm_type=self.dreamer_config.grad_norm_type,
        )
        self.model_optimizer.step()
        losses = {
            "reconstruction_observation_loss": reconstruction_observation_loss.mean().item(),
            "reward_loss": reward_loss.mean().item(),
            "kl_divergence_loss": kl_divergence_loss.item(),
            "model_loss": model_loss.item(),
        }
        return losses

    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.stochastic_size)
        deterministic = deterministics.reshape(-1, self.deterministic_size)

        # continue_predictor reinit
        for t in range(self.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        losses = self._agent_update(self.behavior_learning_infos.get_stacked())
        return losses

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.dreamer_config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.dreamer_config.discount_ * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.horizon_length,
            self.device,
            self.dreamer_config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        logger.debug(f"actor")
        for i, p in enumerate(self.actor.parameters()):
            # print name and norm
            logger.debug(f"{i} : {p.grad.norm()}")
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.dreamer_config.grad_clip_norm,
            norm_type=self.dreamer_config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()

        logger.debug(f"critic")
        for i, p in enumerate(self.critic.parameters()):
            # print name and norm
            logger.debug(f"{i} : {p.grad.norm()}")

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.dreamer_config.grad_clip_norm,
            norm_type=self.dreamer_config.grad_norm_type,
        )
        self.critic_optimizer.step()

        losses = {"actor_loss": actor_loss.item(), "value_loss": value_loss.item()}
        return losses

    @torch.no_grad()
    def environment_interaction(self, env, num_steps_per_env, num_envs, train=True):
        # five full episodes in the original dreamer
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float)
        cur_episode_length = torch.zeros(num_envs, dtype=torch.float)
        score_lst = []

        observation, _ = env.reset()
        embedded_observation = self.encoder(observation.to(self.device))
        batch_size = embedded_observation.shape[0]
        posterior, deterministic = self.rssm.recurrent_model_input_init(batch_size)
        action = torch.zeros(batch_size, self.action_size).to(self.device)

        for t in range(num_steps_per_env):

            deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
            embedded_observation = embedded_observation.reshape(batch_size, -1)
            _, posterior = self.rssm.representation_model(
                embedded_observation, deterministic
            )
            action = self.actor(posterior, deterministic).detach()

            if self.discrete_action_bool:
                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()

            else:
                buffer_action = action
                env_action = buffer_action

            next_observation, privileged_next_observation, reward, done, info = (
                env.step(env_action)
            )
            if train:
                self.buffer.add(
                    observation, buffer_action, reward, next_observation, done
                )
            score_lst.append(reward.mean().item())
            embedded_observation = self.encoder(next_observation.to(self.device))
            observation = next_observation

            # Book keeping
            if "episode" in info:
                ep_infos.append(info["episode"])
            cur_reward_sum += reward.cpu()
            cur_episode_length += 1
            new_ids = (done > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            score = np.mean(score_lst)
            self.num_total_episodes += 1
            logger.debug(f"episode {self.num_total_episodes} score: {score}")

        return {
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
            "ep_infos": ep_infos,
            "score": score,
        }

    def update_dreamer_v1(self, train_metrics) -> dict:
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        dynamic learning + behavior learning
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
            generator = self.storage.reccurent_mini_batch_generator_dreamer(
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
            obs = obs.to(self.device)  # t, t+seq_len
            actions = actions.to(self.device)  # t-1, t+seq_len-1
            rewards = rewards.to(self.device)  # t-1 to t+seq_len-1
            nonterms = 1 - terms

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