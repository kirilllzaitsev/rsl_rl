import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from attrdict import AttrDict
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.modules.decoder import Decoder
from dreamer.modules.encoder import Encoder
from dreamer.modules.model import RSSM, ContinueModel, RewardModel
from dreamer.utils.utils import load_config
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.models.rssm import RSSM as Dreamerv2RSSM
from dreamerv2.training.trainer import Trainer as DreamerTrainer
from dreamerv2.utils.algorithm import compute_return
from dreamerv2.utils.module import FreezeParameters, get_parameters
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.rollout_storage import ReplayBuffer


class DreamerConfig:
    obs_shape: tuple = (1, 48)
    action_size: int = 12
    pixel: bool = False
    action_repeat: int = 1

    capacity: int = int(1e6)
    action_dtype: np.dtype = np.float32

    seq_len: int = 50  # ?
    batch_size: int = 50  # ?

    # rssm_type: str = "continuous"
    # rssm_info: dict = {
    #     "deter_size": 256,
    #     "stoch_size": 32,
    #     "min_std": 0.01,
    # }  # ?
    rssm_type: str = "discrete"
    rssm_info: dict = {
        "deter_size": 128,
        "stoch_size": 20,
        "class_size": 20,
        "category_size": 20,
        "min_std": 0.1,
    }
    embedding_size: int = 128
    rssm_node_size: int = 128

    grad_clip_norm: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10

    lr: dict = {"model": 2e-4, "actor": 4e-5, "critic": 1e-4}
    loss_scale: dict = {"kl": 0.1, "reward": 1.0, "discount": 5.0}
    kl: dict = {
        "use_kl_balance": True,
        "kl_balance_scale": 0.8,
        "use_free_nats": False,
        "free_nats": 0.0,
    }

    # ?
    # use_slow_target: float = True
    # slow_target_update: int = 100
    # slow_target_fraction: float = 1.00

    actor: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": "one_hot",
        "min_std": 1e-4,
        "init_std": 5,
        "mean_scale": 5,
        "activation": nn.ELU,
    }
    expl: dict = {
        "train_noise": 0.4,
        "eval_noise": 0.0,
        "expl_min": 0.05,
        "expl_decay": 7000.0,
        "expl_type": "epsilon_greedy",
    }
    critic: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": "normal",
        "activation": nn.ELU,
    }
    actor_grad: str = "reinforce"
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    obs_encoder: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": None,
        "activation": nn.ELU,
        "kernel": 3,
        "depth": 16,
    }
    obs_decoder: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": "normal",
        "activation": nn.ELU,
        "kernel": 3,
        "depth": 16,
    }
    reward: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": "normal",
        "activation": nn.ELU,
    }
    discount: dict = {
        "layers": 2,
        "node_size": 64,
        "dist": "binary",
        "activation": nn.ELU,
        "use": True,
    }


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

        self.config = argparse.Namespace(
            **dict(
                collect_interval=100,
                batch_size=50,
                batch_length=50,
            )
        )

        self.discrete_action_bool = False
        config_path = "/media/master/wext/msc_studies/fourth_semester/robot_learning/project/related_work/SimpleDreamer/dreamer/configs/leggedgym-quadruped-walk.yml"
        with open(config_path) as f:
            self.config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
        self.num_total_episodes = 0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # actor_obs_shape vs critic_obs_shape?
        assert len(action_shape) == 1
        self.action_size = action_shape[0]
        self.buffer = ReplayBuffer(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            observation_shape=actor_obs_shape,
            action_size=self.action_size,
            device=self.device,
        )
        observation_shape = actor_obs_shape
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
        self.decoder = DenseModel(
            self.dreamer_config.obs_shape,
            self.modelstate_size,
            self.dreamer_config.obs_decoder,
        ).to(self.device)
        self.rssm = RSSM(self.action_size, self.config).to(self.device)
        self.reward_predictor = RewardModel(self.config).to(self.device)
        self.continue_predictor = ContinueModel(self.config).to(self.device)
        self.actor = Actor(self.discrete_action_bool, self.action_size, self.config).to(
            self.device
        )
        self.critic = Critic(self.config).to(self.device)

    def act(self, obs, critic_obs):
        # returns actions
        raise NotImplementedError

    def process_env_step(self, rewards, dones, infos):
        # add data to buffer
        raise NotImplementedError

    def update(self, train_metrics) -> dict:
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        dynamic learning + behavior learning
        """
        # self.num_steps_per_env
        for collect_interval in range(self.config.collect_interval):
            data = self.buffer.sample(self.config.batch_size, self.config.batch_length)
        raise NotImplementedError

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.RSSM.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.RSSM.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
            return evaluate_score

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
