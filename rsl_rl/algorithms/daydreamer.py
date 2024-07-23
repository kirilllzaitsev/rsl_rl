import copy
import logging
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from dreamer.modules.model import RSSM, ContinueModel, RewardModel
from dreamer.utils.utils import DynamicInfos, compute_lambda_values, create_normal_dist
from dreamerv2.models.dense import DenseModel
from dreamerv2.utils.module import FreezeParameters, get_parameters
from rsl_rl.algorithms.dreamer.actor import Actor
from rsl_rl.algorithms.dreamer.critic import Critic
from rsl_rl.storage.rollout_storage import ReplayBuffer
from tqdm.auto import tqdm

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DreamerConfig:
    obs_shape: tuple = (1, 48)
    action_size: int = 12
    pixel: bool = False

    buffer_capacity: int = int(
        1e5
    )  # the more the better, but have to scale network capacities accordingly
    action_dtype: np.dtype = np.float32

    rssm_info: dict = {
        "deter_size": 400,
        "stoch_size": 50,
        "min_std": 0.1,
    }
    embedding_size: int = 512

    grad_clip_norm: float = 100.0
    grad_norm_type: int = 2
    discount_: float = 0.997
    lambda_: float = 0.95
    horizon_length: int = 15

    loss_scale: dict = {"kl": 1.0, "reward": 1.0, "discount": 1.0}
    kl: dict = {
        # "use_free_nats": False,
        # "free_nats": 0.0,
        "use_free_nats": True,
        "free_nats": 1.0,
    }
    actor: dict = {
        "layers": 3,  # same as node_size but the effect is less pronounced
        "node_size": 512,  # higher -> positive impact on actor loss, negative impact on value loss
        # "dist": "one_hot",  # not used
        "min_std": 1e-3,
        "init_std": 1.0,  # important. has to be high with tanh transform
        "mean_scale": 1.0,  # not important
        "activation": nn.ELU,
    }
    critic: dict = {
        "layers": 3,  # ? higher -> negative impact on both actor and value loss. need to increase their capacities as well (but why higher node_size helps)?!
        "node_size": 512,  # higher -> positive impact on both actor and value loss
        "dist": "normal",
        "activation": nn.ELU,
    }

    obs_encoder: dict = {
        "layers": 3,
        "node_size": 512,
        "dist": None,
        "activation": nn.ELU,
    }
    obs_decoder: dict = {
        "layers": 4,
        "node_size": 512,
        "dist": "normal",
        "activation": nn.ELU,
    }
    reward: dict = {
        "layers": 3,
        "node_size": 512,
        "dist": "normal",
        "activation": nn.ELU,
    }
    recurrent_model_config = {
        "hidden_size": 512,
        "activation": "ELU",
    }
    transition_model_config = {
        "hidden_size": 512,
        "num_layers": 3,
        "activation": "ELU",
        "min_std": rssm_info["min_std"],
    }
    representation_model_config = {
        "embedded_state_size": embedding_size,
        "hidden_size": 512,
        "num_layers": 3,
        "activation": "ELU",
        "min_std": rssm_info["min_std"],
    }

    use_continue_flag: bool = False  # NN to predict end of episode
    # learning rates are crucial for convergence and overall performance
    model_learning_rate: float = 3e-4
    actor_learning_rate: float = 8e-5
    critic_learning_rate: float = 8e-5

    collect_interval: int = 10


class DayDreamer:

    def __init__(
        self,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device="cpu",
    ):

        self.device = device

        # initialized later
        self.storage = None
        self.batch_size = None
        self.seq_len = None

        self.dreamer_config = DreamerConfig()
        self.kl_info = self.dreamer_config.kl

        # following SimpleDreamer
        self.action_size = self.dreamer_config.action_size
        self.discrete_action_bool = False
        self.num_total_episodes = 0

        self.encoder = DenseModel(
            (self.dreamer_config.embedding_size,),
            int(np.prod(self.dreamer_config.obs_shape)),
            self.dreamer_config.obs_encoder,
        ).to(self.device)
        self.modelstate_size = (
            self.dreamer_config.rssm_info["stoch_size"]
            + self.dreamer_config.rssm_info["deter_size"]
        )
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
            recurrent_model_config=self.dreamer_config.recurrent_model_config,
            transition_model_config=self.dreamer_config.transition_model_config,
            representation_model_config=self.dreamer_config.representation_model_config,
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
        self.model_modules = [
            self.encoder,
            self.decoder,
            self.rssm,
            self.reward_predictor,
        ]
        if self.dreamer_config.use_continue_flag:
            self.model_modules.append(self.continue_predictor)
        self.model_params = get_parameters(self.model_modules)
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

        # print all model parameters
        print(
            f"self.encoder.parameters={sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}"
        )
        print(
            f"self.decoder.parameters={sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}"
        )
        print(
            f"self.rssm.parameters={sum(p.numel() for p in self.rssm.parameters() if p.requires_grad)}"
        )
        print(
            f"self.reward_predictor.parameters={sum(p.numel() for p in self.reward_predictor.parameters() if p.requires_grad)}"
        )
        if self.dreamer_config.use_continue_flag:
            print(
                f"self.continue_predictor.parameters={sum(p.numel() for p in self.continue_predictor.parameters() if p.requires_grad)}"
            )
        print(
            f"self.actor.parameters={sum(p.numel() for p in self.actor.parameters() if p.requires_grad)}"
        )
        print(
            f"self.critic.parameters={sum(p.numel() for p in self.critic.parameters() if p.requires_grad)}"
        )

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
        # start from 1 because the first observation is given
        for t in range(1, self.num_transitions_per_env):
            # debug
            if t == 1:
                # get distribution of actions
                action = data.action[:, t - 1]
                # print diff stats
                logger.debug(
                    f"action: {action.mean()}, {action.std()} {action.min()}, {action.max()}"
                )
                # same for observation
                obs = data.observation[:, t]
                logger.debug(f"obs: {obs.mean()}, {obs.std()} {obs.min()}, {obs.max()}")
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
        # reconstruction of observation
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

        # reward loss
        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        # KL(prior || posterior)
        # in dreamerv2, these are categorical distributions
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

        # total loss
        model_loss = (
            self.dreamer_config.loss_scale["kl"] * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )

        # continue model
        if self.dreamer_config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()

        self.log_module_grads(self.encoder, "encoder")
        self.log_module_grads(self.decoder, "decoder")
        self.log_module_grads(self.rssm.transition_model, "transition_model")
        self.log_module_grads(self.rssm.representation_model, "representation_model")
        self.log_module_grads(self.reward_predictor, "reward_predictor")
        if self.dreamer_config.use_continue_flag:
            self.log_module_grads(self.continue_predictor, "continue_predictor")

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
        with FreezeParameters(self.model_modules, enabled=False):
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
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        with FreezeParameters(self.model_modules, enabled=False):
            predicted_rewards = self.reward_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
            if self.dreamer_config.use_continue_flag:
                continues = self.continue_predictor(
                    behavior_learning_infos.priors,
                    behavior_learning_infos.deterministics,
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

        last_observation = env.get_observations()
        embedded_observation = self.encoder(last_observation.to(self.device))
        observation = copy.deepcopy(last_observation)
        batch_size = embedded_observation.shape[0]
        posterior, deterministic = self.rssm.recurrent_model_input_init(batch_size)
        action = torch.zeros(batch_size, self.action_size).to(self.device)

        count_dones = 0
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

            count_dones += len(new_ids)

            if train:
                score = np.mean(score_lst)
                self.num_total_episodes += 1
                logger.debug(f"episode {self.num_total_episodes} score: {score}")
        print(f"share of dones: {round(count_dones/ num_steps_per_env, 2)}")

        if not train:
            evaluate_score = np.mean(score_lst)
            logger.debug("evaluate score : ", evaluate_score)
            logger.debug("test score", evaluate_score, self.num_total_episodes)
            return evaluate_score
        return {
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
            "ep_infos": ep_infos,
            "last_observation": observation,
        }

    def log_module_grads(self, module, name):
        logger.debug(name)
        for i, p in enumerate(module.parameters()):
            if p.grad is None:
                continue
            logger.debug(f"{i} : {p.grad.norm()}")
