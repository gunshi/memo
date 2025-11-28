from collections import defaultdict
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional

import gym.spaces as spaces
import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)
from habitat_baselines.utils.common import get_action_space_info
from habitat_baselines.utils.timing import g_timer
from torch.nn.utils.rnn import pad_sequence

ATT_MASK_K = "att_mask"
HIDDEN_WINDOW_K = "hidden_window"
START_HIDDEN_WINDOW_K = "start_hidden_window"
START_ATT_MASK_K = "start_att_mask"
FETCH_BEFORE_COUNTS_K = "fetch_before_counts"


def transpose_stack_pad_dicts(dicts_i):
    res = {}
    for k in dicts_i[0].keys():
        if isinstance(dicts_i[0][k], dict):
            res[k] = transpose_stack_pad_dicts([d[k] for d in dicts_i])
        else:
            res[k] = pad_sequence(
                [d[k] for d in dicts_i], batch_first=True, padding_value=0.0
            )

    return res


def np_dtype_to_torch_dtype(dtype):
    assert hasattr(torch, dtype.name)
    return getattr(torch, dtype.name)

@baseline_registry.register_storage
class ACTransformerRolloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        is_double_buffered: bool = False,
        device: bool = "cpu",
        separate_rollout_and_policy: bool = False,
        on_cpu: bool = False,
        dtype = torch.float16,
        freeze_visual_feats = False,
        acting_context: Optional[int] = None,
        is_memory: bool = False,
        summary_length: Optional[int] = None,
        segment_length: Optional[int] = None,
        avoid_memory: bool = False,
        calc_cross_eps_returns: bool = False,
    ):
        self.dst_device = device
        self.separate_rollout_and_policy = separate_rollout_and_policy or on_cpu
        if separate_rollout_and_policy or on_cpu:
            if on_cpu:
                device = "cpu"
            else:
                if isinstance(device, str):
                    if ":" in device:
                        _d, _i = device.split(":")
                        _i = int(_i)
                    else:
                        _d = device
                        _i = torch.cuda.current_device()
                else:
                    _d, _i = device.type, device.index
                    if _i is None:
                        _i = torch.cuda.current_device()
                
                _i += torch.distributed.get_world_size()
                
                device = torch.device(_d, _i)

        self._dtype = dtype
        self.context_length = actor_critic.context_len
        self.memory_length = actor_critic.memory_size
        self.is_banded = actor_critic.banded_attention
        self.add_context_loss = actor_critic.add_context_loss
        self.acting_context = acting_context
        self.is_memory = is_memory
        self.avoid_memory = avoid_memory
        self.calc_cross_eps_returns = calc_cross_eps_returns

        if self.acting_context is not None:
            assert not self.is_banded

        self.is_first_update = True
        self.old_context_length = 0
        self._frozen_visual = freeze_visual_feats

        self.og_numsteps = numsteps
        numsteps += self.context_length

        action_shape, discrete_actions = get_action_space_info(action_space)

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        self.vis_keys = {
                k
                for k in observation_space.spaces
                if len(observation_space.spaces[k].shape) == 3
            }
        if 'head_depth' in self.vis_keys:
            self.vis_keys.remove('head_depth') 

        print('VIS KEYS: ', self.vis_keys)

        if PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces:
            to_remove_sensor = self.vis_keys
        else:
            to_remove_sensor = []
        print('TO REMOVE SENSOR: ', to_remove_sensor)

        for sensor in observation_space.spaces:
            if sensor in to_remove_sensor:
                continue

            dtype = observation_space.spaces[sensor].dtype
            if self._dtype in (torch.bfloat16, torch.float16) and "float" in dtype.name:
                dtype = np.dtype("float16")
            dtype = np_dtype_to_torch_dtype(dtype)

            shape = observation_space.spaces[sensor].shape

            self.buffers["observations"][sensor] = torch.zeros(
                    (
                        num_envs * 
                        (numsteps + 1),
                        *shape
                    ),
                    dtype=dtype,
                    device=device,
                    pin_memory=(device == "cpu")
                ).view(
                    num_envs, 
                    (numsteps + 1),
                    *shape
                )

        if is_memory:
            self._recurrent_hidden_states_shape = (
                actor_critic.num_recurrent_layers,
                actor_critic.memory_size,
                num_envs,
                actor_critic.recurrent_hidden_size,
            )
        else:
            self._recurrent_hidden_states_shape = (
                actor_critic.num_recurrent_layers,
                2,
                num_envs,
                actor_critic.num_heads,
                numsteps + 1,
                actor_critic.recurrent_hidden_size//actor_critic.num_heads,
            )

        self.summary_length = summary_length
        self.segment_length = segment_length

        if summary_length is not None:
            print('SUMMARY VECTOR LEN: ', (int(self.context_length/self.og_numsteps)+1))
            self.summary_vectors_shape = (
                 actor_critic.num_recurrent_layers,
                 2,
                 num_envs,
                 actor_critic.num_heads,
                 summary_length*(int(numsteps/self.segment_length)),
                 actor_critic.recurrent_hidden_size//actor_critic.num_heads,
             )

            self.init_summary_vectors()


        self.buffers["rewards"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))
        self.buffers["value_preds"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))
        self.buffers["returns"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))

        self.buffers["action_log_probs"] = torch.zeros(
            (num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )

        if action_shape is None:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            (num_envs, (numsteps + 1), *action_shape), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )
        self.buffers["prev_actions"] = torch.zeros(
            (num_envs, (numsteps + 1), *action_shape), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].to(self._dtype)
            self.buffers["prev_actions"] = self.buffers["prev_actions"].to(self._dtype)

        self.buffers["masks"] = torch.zeros(
            (num_envs, (numsteps + 1), 1), dtype=torch.bool, device=device, pin_memory=(device == "cpu")
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.num_steps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]
        self.current_summary_idxs = [0 for _ in range(self._nbuffers)]

        # The default device to torch is the CPU, so everything is on the CPU.
        self.device = torch.device(device)
        self.init_recurrent_hidden_states()
        self.init_summary_vectors()

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    @property
    def current_summary_idx(self) -> int:
        assert all(
            s == self.current_summary_idxs[0]
            for s in self.current_summary_idxs
        )
        return self.current_summary_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))
        if self._recurrent_hidden_states is not None:
            self._recurrent_hidden_states = self._recurrent_hidden_states.to(device)
        if self.summary_vectors is not None:
            self.summary_vectors = self.summary_vectors.to(device)
        self.device = device

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1
        # print('CURRENT ROLL STEP: ', self.current_rollout_step_idxs[buffer_index])
        if self.summary_length is not None and self.current_rollout_step_idxs[buffer_index]%self.segment_length == 0:
            self.current_summary_idxs[buffer_index] += self.summary_length
            # print('CURRENT SUMMARY STEP: ', self.current_summary_idxs[buffer_index])

    @g_timer.avg_time("rollout_storage.insert", level=1)
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):

        if not self.is_double_buffered:
            assert buffer_index == 0

        if self._frozen_visual and next_observations is not None:
            next_observations = {
                k: v for k, v in next_observations.items() if k not in self.vis_keys
            }

        next_step = dict(
            observations=next_observations,
            prev_actions=actions,
            masks=next_masks,
        )


        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index] + 1),
                next_step,
                strict=False,
            )

        if next_recurrent_hidden_states is not None:
            if self.is_memory:
                self._recurrent_hidden_states[:, :, env_slice] = next_recurrent_hidden_states['past_key_values']
            else:
                self._recurrent_hidden_states[:, :, env_slice, :, self.current_rollout_step_idxs[buffer_index] + 1] = next_recurrent_hidden_states['past_key_values']

            if next_recurrent_hidden_states.get("softprompt") is not None:
                self.summary_vectors[:, :, env_slice, :, self.current_summary_idxs[buffer_index]: self.current_summary_idxs[buffer_index]+self.summary_length] = next_recurrent_hidden_states["softprompt"]

        if len(current_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index]),
                current_step,
                strict=False,
            )

    def after_update(self):
        if self.context_length > 0:
            self.old_context_length = min(
                self.context_length, self.current_rollout_step_idx
            )
            self.old_summary_vecs_length = min(
                (self.context_length//self.segment_length)*self.summary_length, self.current_summary_idx
            )
            self.buffers[:, 0 : self.old_context_length + 1] = deepcopy(
                self.buffers[
                    :,
                    self.current_rollout_step_idx
                    - self.old_context_length : self.current_rollout_step_idx
                    + 1,
                ]
            )

        else:
            self.old_context_length = 0
            self.buffers[:, 0] = self.buffers[:, self.current_rollout_step_idx]
            self.old_summary_vecs_length = 0


        # self.init_recurrent_hidden_states()

        self.current_rollout_step_idxs = [
            self.old_context_length for _ in self.current_rollout_step_idxs
        ]
        self.current_summary_idxs = [
            self.old_summary_vecs_length for _ in self.current_summary_idxs
        ]
        self.is_first_update = False

    @g_timer.avg_time("rollout_storage.compute_returns", level=1)
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][
                :, self.current_rollout_step_idx
            ] = next_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                if self.calc_cross_eps_returns:
                    # use dummy masks that will compute cross-episode returns
                    use_masks = torch.ones_like(self.buffers["masks"][:, step + 1])
                else:
                    use_masks = self.buffers["masks"][:, step + 1]
                delta = (
                    self.buffers["rewards"][:, step]
                    + gamma
                    * self.buffers["value_preds"][:, step + 1]
                    * use_masks
                    - self.buffers["value_preds"][:, step]
                )
                gae = (
                    delta
                    + gamma * tau * gae * use_masks
                )
                self.buffers["returns"][:, step] = (  # type: ignore
                    gae + self.buffers["value_preds"][:, step]  # type: ignore
                )

        else:
            self.buffers["returns"][
                :, self.current_rollout_step_idx
            ] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                if self.calc_cross_eps_returns:
                    # use dummy masks that will compute cross-episode returns
                    use_masks = torch.ones_like(self.buffers["masks"][:, step + 1])
                else:
                    use_masks = self.buffers["masks"][:, step + 1]
                self.buffers["returns"][:, step] = (
                    gamma
                    * self.buffers["returns"][:, step + 1]
                    * use_masks
                    + self.buffers["rewards"][:, step]
                )

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def insert_first_observations(self, batch, reset_memory=False):
        if self._frozen_visual:
            batch = {k: v for k, v in batch.items() if k not in self.vis_keys}
        self.buffers["masks"][:, 0] = 0
        self.buffers["observations"][:, 0] = batch
        if (self.is_memory or self.avoid_memory) and reset_memory:
            self.init_recurrent_hidden_states()
            self.init_summary_vectors()

    def get_current_step(self, env_slice, buffer_index):
        batch = self.buffers[
            env_slice,
            self.current_rollout_step_idxs[buffer_index],
        ]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idxs[buffer_index]
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idxs[buffer_index] - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idxs[buffer_index] else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
                :, :,
                env_slice,
            ]
        else:
            batch["recurrent_hidden_states"] = TensorDict(
                {"past_key_values" : self._recurrent_hidden_states[
                        :, :,
                        env_slice,
                        :,
                        start_idx + 1 : self.current_rollout_step_idxs[buffer_index] + 1,
                    ],
                "softprompt" : None if self.current_summary_idxs[buffer_index]==0 else self.summary_vectors[
                        :, :,
                        env_slice,
                        :,
                        :self.current_summary_idxs[buffer_index],
                    ]

            })

        batch["masks"] = self.buffers[
            env_slice,
            start_idx : self.current_rollout_step_idxs[buffer_index] + 1,
        ]["masks"]

        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch

    def get_last_step(self):
        batch = self.buffers[:, self.current_rollout_step_idx]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idx
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states
        else:            
            batch["recurrent_hidden_states"] = TensorDict(
                {
                    "past_key_values": self._recurrent_hidden_states[
                        :, :,
                        :,
                        :,
                        start_idx + 1 : self.current_rollout_step_idx + 1,
                    ],
                    "softprompt": None if self.current_summary_idx==0 else self.summary_vectors[
                        :, :,
                        :,
                        :,
                        :self.current_summary_idx,
                    ]
                }
            )

        batch["masks"] = self.buffers[
            :,
            start_idx : self.current_rollout_step_idx + 1,
        ]["masks"]

        # del self._recurrent_hidden_states
        # self._recurrent_hidden_states = None
        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch

    def get_context_step(self, env_id=None, n_steps=None):
        n_steps = self.old_context_length if n_steps is None else n_steps

        if self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0


        if env_id is None:
            batch = self.buffers[:, start_idx:n_steps]
        else:
            batch = self.buffers[env_id:env_id+1, start_idx:n_steps]
        dims = batch["masks"].shape[:2]
        batch.map_in_place(lambda v: v.flatten(0, 1))
        batch["rnn_build_seq_info"] = TensorDict(
            {
                "dims": torch.from_numpy(
                    np.asarray(dims)
                ),
                "is_first": torch.tensor(True),
                "old_context_length": torch.tensor(self.old_context_length)
            }
        )
        if self.is_memory:
            if env_id is not None:
                batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, env_id:env_id+1]
            else:
                batch["recurrent_hidden_states"] = TensorDict(
                    {
                        "recurrent_hidden_states": self._recurrent_hidden_states_train,
                        "softprompts": self.summary_vectors                                                                         ## is this correct?
                    }
                )

        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch
    
    def update_context_data(self, value_preds, action_log_probs, recurrent_hidden_states, env_id=None, n_steps=None):
        if self._recurrent_hidden_states is None:
            self.init_recurrent_hidden_states()
            self.init_summary_vectors()

        n_steps = self.old_context_length if n_steps is None else n_steps
        env_id = slice(env_id, env_id+1) if env_id is not None else slice(None)

        if self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
                start_summary_idx = max(0, self.current_summary_idx - (self.acting_context//self.segment_length)*self.summary_length)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx

        else:
            start_idx = 0
            start_summary_idx = 0

        if value_preds is not None:
            self.buffers["value_preds"][env_id, start_idx:n_steps] = value_preds

        if recurrent_hidden_states is not None:
            if self.is_memory:
                self._recurrent_hidden_states[:, :, env_id] = recurrent_hidden_states
                self._recurrent_hidden_states_train[:, :, env_id] = recurrent_hidden_states
            else:
                self._recurrent_hidden_states[:, :, env_id, :, start_idx+1:n_steps+1] = recurrent_hidden_states["past_key_values"] # .unsqueeze(2)    
                
                if recurrent_hidden_states.get("softprompt") is not None:
                    self.summary_vectors[:, :, env_id, :, start_summary_idx:start_summary_idx+self.current_summary_idx] = recurrent_hidden_states["softprompt"] # .unsqueeze(2)
                    
        if action_log_probs is not None:
            self.buffers["action_log_probs"][env_id, start_idx:n_steps] = action_log_probs


    def init_summary_vectors(self):
        with torch.inference_mode(False):
            self.summary_vectors = torch.zeros(
                self.summary_vectors_shape, device=self.dst_device, dtype=self._dtype
            )


    def init_recurrent_hidden_states(self):
        with torch.inference_mode(False):
            self._recurrent_hidden_states = torch.zeros(
                self._recurrent_hidden_states_shape, device=self.dst_device, dtype=self._dtype
            )
            if self.is_memory:
                self._recurrent_hidden_states_train = torch.zeros(
                    self._recurrent_hidden_states_shape, device=self.dst_device, dtype=self._dtype
                )

            # set to nan to avoid silent bugs
            # self._recurrent_hidden_states[:] = torch.nan

    def shuffle_episodes(self):
        assert not self.is_memory

        bgn_idxs = torch.nonzero(~self.buffers["masks"])
        bgn_idxs_dict = defaultdict(lambda: [0])
        for item in bgn_idxs:
            if item[1].item() < self.current_rollout_step_idx:
                bgn_idxs_dict[item[0].item()].append(item[1].item())
        bgn_idxs_dict = dict(bgn_idxs_dict)
        len_dict = defaultdict(list)
        for item in bgn_idxs_dict:
            for i in range(1, len(bgn_idxs_dict[item])):
                len_dict[item].append(bgn_idxs_dict[item][i] - bgn_idxs_dict[item][i-1])

        indexes = np.zeros((self._num_envs, self.num_steps+1), dtype="int") + np.arange(self.num_steps+1)[None]
        for i in range(self._num_envs):
            order = np.random.choice(len(len_dict[i]), size=len(len_dict[i]), replace=False)
            curser = 0
            for j in order:
                indexes[i, curser:curser+len_dict[i][j]] = range(bgn_idxs_dict[i][j], bgn_idxs_dict[i][j]+len_dict[i][j])
                curser += len_dict[i][j]

        for i in range(self._num_envs):
            self.buffers[i].map_in_place(lambda x: x[indexes[i]])
        # self.buffers.map_in_place(lambda x: x[np.arange(len(x))[:, None], indexes])

    def del_recurrent_hidden_states(self):
        del self._recurrent_hidden_states
        self._recurrent_hidden_states = None
        if self.is_memory:
            del self._recurrent_hidden_states_train
            self._recurrent_hidden_states_train = None

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
        percent_envs_update = None,
        slice_from = 0
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        # self.del_recurrent_hidden_states()
        num_environments = self.buffers["returns"].size(0)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        if advantages is not None:
           self.buffers["advantages"] = advantages

        # batches = self.buffers[np.random.permutation(num_environments), :self.current_rollout_step_idx]
        b_indexes = np.random.permutation(num_environments)
        if percent_envs_update is not None:
            num_envs_update = int(percent_envs_update*self._num_envs)
            b_indexes = b_indexes[:num_envs_update]

        batch_size = num_environments//num_mini_batch

        for inds in range(0, len(b_indexes), batch_size):
            # When batch size is 1, we don't need to use slicing because it, along with the steps slicing, creates
            # a copy of the data which is not necessary.
            if batch_size == 1:
                batch = self.buffers[b_indexes[inds], slice_from:self.current_rollout_step_idx]
                batch = batch.map(lambda x: x[None])
            else:
                batch = self.buffers[b_indexes[inds:inds+batch_size], slice_from:self.current_rollout_step_idx]

            if self.is_memory:
                if batch_size == 1:
                    batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, b_indexes[inds]:b_indexes[inds]+1]
                else:
                    batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, b_indexes[inds:inds+batch_size]]
            else:
                # batch["recurrent_hidden_states"] = torch.tensor([[]])
                batch["recurrent_hidden_states"] = TensorDict(
                    {
                        "recurrent_hidden_states": torch.tensor([[]]),
                        "softprompts": torch.tensor([[]]),
                    }
                )

            if self.separate_rollout_and_policy:
                batch.map_in_place(lambda v: v.flatten(0, 1).to(self.dst_device))
            else:
                batch.map_in_place(lambda v: v.flatten(0, 1))
            batch["rnn_build_seq_info"] = TensorDict(
                {
                    "dims": torch.from_numpy(
                        np.asarray([min(inds+batch_size, len(b_indexes)) - inds, self.current_rollout_step_idx-slice_from])
                    ),
                    "is_first": torch.tensor(self.is_first_update),
                    "old_context_length": torch.tensor(
                        0 #self.old_context_length
                    ),
                }
            )

            yield batch.to_tree()

    def save_memory(self, path):
        """
        Save all the memories and the associated key-value cache for the memory for later use.
        """
        breakpoint()
        memory_data = {
            "recurrent_hidden_states": self._recurrent_hidden_states,
            "summary_vectors": self.summary_vectors,
            "current_rollout_step_idxs": self.current_rollout_step_idxs,
            "current_summary_idxs": self.current_summary_idxs,
            "context_length": self.context_length,
            "segment_length": self.segment_length,
            "summary_length": self.summary_length,
        }
        with open(path, "wb") as f:
            pkl.dump(memory_data, f)
        


@baseline_registry.register_storage
class RMTTransformerRolloutStorage(ACTransformerRolloutStorage):
    def get_current_step(self, env_slice, buffer_index):
        batch = self.buffers[
            env_slice,
            self.current_rollout_step_idxs[buffer_index],
        ]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idxs[buffer_index]
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idxs[buffer_index] - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idxs[buffer_index] else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
                :, :,
                env_slice,
            ]
        else:
            batch["recurrent_hidden_states"] = TensorDict(
                {"past_key_values" : self._recurrent_hidden_states[
                        :, :,
                        env_slice,
                        :,
                        start_idx + 1 : self.current_rollout_step_idxs[buffer_index] + 1,
                    ],
                "softprompt" : None if self.current_summary_idxs[buffer_index]==0 else self.summary_vectors[
                        :, :,
                        env_slice,
                        :,
                        self.current_summary_idxs[buffer_index] - self.summary_length :self.current_summary_idxs[buffer_index],
                    ]

            })

        batch["masks"] = self.buffers[
            env_slice,
            start_idx : self.current_rollout_step_idxs[buffer_index] + 1,
        ]["masks"]

        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch

    def get_last_step(self):
        batch = self.buffers[:, self.current_rollout_step_idx]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idx
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states
        else:            
            batch["recurrent_hidden_states"] = TensorDict(
                {
                    "past_key_values": self._recurrent_hidden_states[
                        :, :,
                        :,
                        :,
                        start_idx + 1 : self.current_rollout_step_idx + 1,
                    ],
                    "softprompt": None if self.current_summary_idx==0 else self.summary_vectors[
                        :, :,
                        :,
                        :,
                        self.current_summary_idx-self.summary_length:self.current_summary_idx,
                    ]
                }
            )

        batch["masks"] = self.buffers[
            :,
            start_idx : self.current_rollout_step_idx + 1,
        ]["masks"]

        # del self._recurrent_hidden_states
        # self._recurrent_hidden_states = None
        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch
    
    def update_context_data(self, value_preds, action_log_probs, recurrent_hidden_states, env_id=None, n_steps=None):
        if self._recurrent_hidden_states is None:
            self.init_recurrent_hidden_states()
            self.init_summary_vectors()

        n_steps = self.old_context_length if n_steps is None else n_steps
        env_id = slice(env_id, env_id+1) if env_id is not None else slice(None)

        if self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
                start_summary_idx = max(0, self.current_summary_idx - (self.acting_context//self.segment_length)*self.summary_length)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx

        else:
            start_idx = 0
            start_summary_idx = 0

        if value_preds is not None:
            self.buffers["value_preds"][env_id, start_idx:n_steps] = value_preds

        if recurrent_hidden_states is not None:
            if self.is_memory:
                self._recurrent_hidden_states[:, :, env_id] = recurrent_hidden_states
                self._recurrent_hidden_states_train[:, :, env_id] = recurrent_hidden_states
            else:
                self._recurrent_hidden_states[:, :, env_id, :, start_idx+1:n_steps+1] = recurrent_hidden_states["past_key_values"] # .unsqueeze(2)    
                
                if recurrent_hidden_states.get("softprompt") is not None:
                    self.summary_vectors[:, :, env_id, :, start_summary_idx+self.current_summary_idx-self.summary_length:start_summary_idx+self.current_summary_idx] = recurrent_hidden_states["softprompt"] # .unsqueeze(2)
                    
        if action_log_probs is not None:
            self.buffers["action_log_probs"][env_id, start_idx:n_steps] = action_log_probs


@baseline_registry.register_storage
class MinimalTransformerRolloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        is_double_buffered: bool = False,
        device: bool = "cpu",
        separate_rollout_and_policy: bool = False,
        on_cpu: bool = False,
        dtype = torch.float16,
        freeze_visual_feats = False,
        acting_context: Optional[int] = None,
        is_memory: bool = False,
        summary_length = None,
        segment_length = None,
        avoid_memory = False,
        calc_cross_eps_returns = False,
    ):
        self.dst_device = device
        self.separate_rollout_and_policy = separate_rollout_and_policy or on_cpu
        if separate_rollout_and_policy or on_cpu:
            if on_cpu:
                device = "cpu"
            else:
                if isinstance(device, str):
                    if ":" in device:
                        _d, _i = device.split(":")
                        _i = int(_i)
                    else:
                        _d = device
                        _i = torch.cuda.current_device()
                else:
                    _d, _i = device.type, device.index
                    if _i is None:
                        _i = torch.cuda.current_device()
                
                _i += torch.distributed.get_world_size()
                
                device = torch.device(_d, _i)

        self._dtype = dtype
        self.context_length = actor_critic.context_len
        self.memory_length = actor_critic.memory_size
        self.is_banded = actor_critic.banded_attention
        self.add_context_loss = actor_critic.add_context_loss
        self.acting_context = acting_context
        self.is_memory = is_memory
        self.avoid_memory = avoid_memory
        self.calc_cross_eps_returns = calc_cross_eps_returns

        if self.acting_context is not None:
            assert not self.is_banded

        self.is_first_update = True
        self.old_context_length = 0
        self._frozen_visual = freeze_visual_feats

        numsteps += self.context_length

        action_shape, discrete_actions = get_action_space_info(action_space)

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        self.vis_keys = {
                k
                for k in observation_space.spaces
                if len(observation_space.spaces[k].shape) == 3
            }

        if 'head_depth' in self.vis_keys:
            self.vis_keys.remove('head_depth') 
            
        if PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces:
            to_remove_sensor = self.vis_keys
        else:
            to_remove_sensor = []

        for sensor in observation_space.spaces:
            if sensor in to_remove_sensor:
                continue

            dtype = observation_space.spaces[sensor].dtype
            if self._dtype in (torch.bfloat16, torch.float16) and "float" in dtype.name:
                dtype = np.dtype("float16")
            dtype = np_dtype_to_torch_dtype(dtype)

            shape = observation_space.spaces[sensor].shape

            self.buffers["observations"][sensor] = torch.zeros(
                    (
                        num_envs * 
                        (numsteps + 1),
                        *shape
                    ),
                    dtype=dtype,
                    device=device,
                    pin_memory=(device == "cpu")
                ).view(
                    num_envs, 
                    (numsteps + 1),
                    *shape
                )

        if is_memory:
            self._recurrent_hidden_states_shape = (
                actor_critic.num_recurrent_layers,
                actor_critic.memory_size,
                num_envs,
                actor_critic.recurrent_hidden_size,
            )
        else:
            self._recurrent_hidden_states_shape = (
                actor_critic.num_recurrent_layers,
                2,
                num_envs,
                actor_critic.num_heads,
                numsteps + 1,
                actor_critic.recurrent_hidden_size//actor_critic.num_heads,
            )

        self.buffers["rewards"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))
        self.buffers["value_preds"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))
        self.buffers["returns"] = torch.zeros((num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu"))

        self.buffers["action_log_probs"] = torch.zeros(
            (num_envs, (numsteps + 1), 1), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )

        if action_shape is None:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            (num_envs, (numsteps + 1), *action_shape), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )
        self.buffers["prev_actions"] = torch.zeros(
            (num_envs, (numsteps + 1), *action_shape), device=device, dtype=self._dtype, pin_memory=(device == "cpu")
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].to(self._dtype)
            self.buffers["prev_actions"] = self.buffers["prev_actions"].to(self._dtype)

        self.buffers["masks"] = torch.zeros(
            (num_envs, (numsteps + 1), 1), dtype=torch.bool, device=device, pin_memory=(device == "cpu")
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.num_steps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

        # The default device to torch is the CPU, so everything is on the CPU.
        self.device = torch.device(device)
        self.init_recurrent_hidden_states()

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))
        if self._recurrent_hidden_states is not None:
            self._recurrent_hidden_states = self._recurrent_hidden_states.to(device)
        self.device = device

    @g_timer.avg_time("rollout_storage.insert", level=1)
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        if self._frozen_visual and next_observations is not None:
            next_observations = {
                k: v for k, v in next_observations.items() if k not in self.vis_keys
            }

        next_step = dict(
            observations=next_observations,
            prev_actions=actions,
            masks=next_masks,
        )


        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index] + 1),
                next_step,
                strict=False,
            )

        if next_recurrent_hidden_states is not None:
            if self.is_memory:
                self._recurrent_hidden_states[:, :, env_slice] = next_recurrent_hidden_states
            else:
                self._recurrent_hidden_states[:, :, env_slice, :, self.current_rollout_step_idxs[buffer_index] + 1] = next_recurrent_hidden_states

        if len(current_step) > 0:
            self.buffers.set(
                (env_slice, self.current_rollout_step_idxs[buffer_index]),
                current_step,
                strict=False,
            )

    def after_update(self):
        if self.context_length > 0:
            self.old_context_length = min(
                self.context_length, self.current_rollout_step_idx
            )
            self.buffers[:, 0 : self.old_context_length + 1] = deepcopy(
                self.buffers[
                    :,
                    self.current_rollout_step_idx
                    - self.old_context_length : self.current_rollout_step_idx
                    + 1,
                ]
            )
        else:
            self.old_context_length = 0
            self.buffers[:, 0] = self.buffers[:, self.current_rollout_step_idx]

        # self.init_recurrent_hidden_states()

        self.current_rollout_step_idxs = [
            self.old_context_length for _ in self.current_rollout_step_idxs
        ]
        self.is_first_update = False

    @g_timer.avg_time("rollout_storage.compute_returns", level=1)
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][
                :, self.current_rollout_step_idx
            ] = next_value
            gae = 0.0

            for step in reversed(range(self.current_rollout_step_idx)):
                if self.calc_cross_eps_returns:
                    # use dummy masks that will compute cross-episode returns
                    use_masks = torch.ones_like(self.buffers["masks"][:, step + 1])
                else:
                    use_masks = self.buffers["masks"][:, step + 1]

                delta = (
                    self.buffers["rewards"][:, step]
                    + gamma
                    * self.buffers["value_preds"][:, step + 1]
                    * use_masks
                    - self.buffers["value_preds"][:, step]
                )
                gae = (
                    delta
                    + gamma * tau * gae * use_masks
                )
                self.buffers["returns"][:, step] = (  # type: ignore
                    gae + self.buffers["value_preds"][:, step]  # type: ignore
                )

        else:
            self.buffers["returns"][
                :, self.current_rollout_step_idx
            ] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                if self.calc_cross_eps_returns:
                    # use dummy masks that will compute cross-episode returns
                    use_masks = torch.ones_like(self.buffers["masks"][:, step + 1])
                else:
                    use_masks = self.buffers["masks"][:, step + 1]
                self.buffers["returns"][:, step] = (
                    gamma
                    * self.buffers["returns"][:, step + 1]
                    * use_masks
                    + self.buffers["rewards"][:, step]
                )

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def insert_first_observations(self, batch, reset_memory=False):
        if self._frozen_visual:
            batch = {k: v for k, v in batch.items() if k not in self.vis_keys}
        self.buffers["masks"][:, 0] = 0
        self.buffers["observations"][:, 0] = batch
        if (self.is_memory or self.avoid_memory) and reset_memory:
            self.init_recurrent_hidden_states()

    def get_current_step(self, env_slice, buffer_index):
        batch = self.buffers[
            env_slice,
            self.current_rollout_step_idxs[buffer_index],
        ]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idxs[buffer_index]
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idxs[buffer_index] - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idxs[buffer_index] else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
                :, :,
                env_slice,
            ]
        else:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
                :, :,
                env_slice,
                :,
                start_idx + 1 : self.current_rollout_step_idxs[buffer_index] + 1,
            ]
        batch["masks"] = self.buffers[
            env_slice,
            start_idx : self.current_rollout_step_idxs[buffer_index] + 1,
        ]["masks"]

        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch

    def get_last_step(self):
        batch = self.buffers[:, self.current_rollout_step_idx]

        if self.is_banded:
            start_idx = max(
                0,
                self.current_rollout_step_idx
                - self.context_length,
            )
        elif self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0

        if self.is_memory:
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states
        else:            
            batch["recurrent_hidden_states"] = self._recurrent_hidden_states[
                :, :, :, :,
                start_idx + 1 : self.current_rollout_step_idx + 1,
            ]
        batch["masks"] = self.buffers[
            :,
            start_idx : self.current_rollout_step_idx + 1,
        ]["masks"]

        # del self._recurrent_hidden_states
        # self._recurrent_hidden_states = None
        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch

    def get_context_step(self, env_id=None, n_steps=None):
        n_steps = self.old_context_length if n_steps is None else n_steps

        if self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0


        if env_id is None:
            batch = self.buffers[:, start_idx:n_steps]
        else:
            batch = self.buffers[env_id:env_id+1, start_idx:n_steps]
        dims = batch["masks"].shape[:2]
        batch.map_in_place(lambda v: v.flatten(0, 1))
        batch["rnn_build_seq_info"] = TensorDict(
            {
                "dims": torch.from_numpy(
                    np.asarray(dims)
                ),
                "is_first": torch.tensor(True),
                "old_context_length": torch.tensor(self.old_context_length)
            }
        )
        if self.is_memory:
            if env_id is not None:
                batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, env_id:env_id+1]
            else:
                batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train

        if self.separate_rollout_and_policy:
            batch = batch.map(lambda x: x.to(self.dst_device))
        return batch
    
    def update_context_data(self, value_preds, action_log_probs, recurrent_hidden_states, env_id=None, n_steps=None):
        if self._recurrent_hidden_states is None:
            self.init_recurrent_hidden_states()
            self.init_summary_vectors()

        n_steps = self.old_context_length if n_steps is None else n_steps
        env_id = slice(env_id, env_id+1) if env_id is not None else slice(None)

        if self.acting_context is not None:
            if self.acting_context >= 0:
                start_idx = max(0, self.current_rollout_step_idx - self.acting_context)
            else:
                start_idx = -self.acting_context
                start_idx = 0 if start_idx > self.current_rollout_step_idx else start_idx
        else:
            start_idx = 0

        if value_preds is not None:
            self.buffers["value_preds"][env_id, start_idx:n_steps] = value_preds

        if recurrent_hidden_states is not None:
            if self.is_memory:
                self._recurrent_hidden_states[:, :, env_id] = recurrent_hidden_states
                self._recurrent_hidden_states_train[:, :, env_id] = recurrent_hidden_states
            else:
                self._recurrent_hidden_states[:, :, env_id, :, start_idx+1:n_steps+1] = recurrent_hidden_states

        if action_log_probs is not None:
            self.buffers["action_log_probs"][env_id, start_idx:n_steps] = action_log_probs


    def init_recurrent_hidden_states(self):
        with torch.inference_mode(False):
            self._recurrent_hidden_states = torch.zeros(
                self._recurrent_hidden_states_shape, device=self.dst_device, dtype=self._dtype
            )
            if self.is_memory:
                self._recurrent_hidden_states_train = torch.zeros(
                    self._recurrent_hidden_states_shape, device=self.dst_device, dtype=self._dtype
                )

    def shuffle_episodes(self):
        assert not self.is_memory
        bgn_idxs = torch.nonzero(~self.buffers["masks"])
        bgn_idxs_dict = defaultdict(lambda: [0])
        for item in bgn_idxs:
            if item[1].item() < self.current_rollout_step_idx:
                bgn_idxs_dict[item[0].item()].append(item[1].item())
        bgn_idxs_dict = dict(bgn_idxs_dict)
        len_dict = defaultdict(list)
        for item in bgn_idxs_dict:
            for i in range(1, len(bgn_idxs_dict[item])):
                len_dict[item].append(bgn_idxs_dict[item][i] - bgn_idxs_dict[item][i-1])

        indexes = np.zeros((self._num_envs, self.num_steps+1), dtype="int") + np.arange(self.num_steps+1)[None]
        for i in range(self._num_envs):
            order = np.random.choice(len(len_dict[i]), size=len(len_dict[i]), replace=False)
            curser = 0
            for j in order:
                indexes[i, curser:curser+len_dict[i][j]] = range(bgn_idxs_dict[i][j], bgn_idxs_dict[i][j]+len_dict[i][j])
                curser += len_dict[i][j]
        # self.buffers.map_in_place(lambda x: x[np.arange(len(x))[:, None], indexes])
        # instead do this operation looping over the envs to avoid out of memory errors
        for i in range(self._num_envs):
            self.buffers[i].map_in_place(lambda x: x[indexes[i]])

    def del_recurrent_hidden_states(self):
        del self._recurrent_hidden_states
        self._recurrent_hidden_states = None
        if self.is_memory:
            del self._recurrent_hidden_states_train
            self._recurrent_hidden_states_train = None

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
        percent_envs_update = None,
        slice_from = 0
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        # self.del_recurrent_hidden_states()
        num_environments = self.buffers["returns"].size(0)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        if advantages is not None:
           self.buffers["advantages"] = advantages

        # batches = self.buffers[np.random.permutation(num_environments), :self.current_rollout_step_idx]
        b_indexes = np.random.permutation(num_environments)
        if percent_envs_update is not None:
            num_envs_update = int(percent_envs_update*self._num_envs)
            b_indexes = b_indexes[:num_envs_update]

        batch_size = num_environments//num_mini_batch

        for inds in range(0, len(b_indexes), batch_size):
            # When batch size is 1, we don't need to use slicing because it, along with the steps slicing, creates
            # a copy of the data which is not necessary.
            if batch_size == 1:
                batch = self.buffers[b_indexes[inds], slice_from:self.current_rollout_step_idx]
                batch = batch.map(lambda x: x[None])
            else:
                batch = self.buffers[b_indexes[inds:inds+batch_size], slice_from:self.current_rollout_step_idx]

            if self.is_memory:
                if batch_size == 1:
                    batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, b_indexes[inds]:b_indexes[inds]+1]
                else:
                    batch["recurrent_hidden_states"] = self._recurrent_hidden_states_train[:, :, b_indexes[inds:inds+batch_size]]
            else:
                batch["recurrent_hidden_states"] = torch.tensor([[]])

            if self.separate_rollout_and_policy:
                batch.map_in_place(lambda v: v.flatten(0, 1).to(self.dst_device))
            else:
                batch.map_in_place(lambda v: v.flatten(0, 1))
            batch["rnn_build_seq_info"] = TensorDict(
                {
                    "dims": torch.from_numpy(
                        np.asarray([min(inds+batch_size, len(b_indexes)) - inds, self.current_rollout_step_idx-slice_from])
                    ),
                    "is_first": torch.tensor(self.is_first_update),
                    "old_context_length": torch.tensor(
                        0 #self.old_context_length
                    ),
                }
            )

            yield batch.to_tree()


def flatten_trajs(
    advantages,
    buffers,
    context_len,
    current_rollout_step_idx,
    hidden_window,
    before_start_att_masks,
    inds,
    max_total_samples: Optional[int] = None,
) -> TensorDict:
    max_data_window_size = current_rollout_step_idx
    n_envs = advantages.shape[0]
    device = advantages.device
    curr_slice = (
        slice(0, max_data_window_size),
        inds,
    )

    batch = buffers[curr_slice]
    batch["advantages"] = advantages[curr_slice]

    # Find where episode starts (when masks = False).
    inserted_start = {}
    ep_starts = torch.nonzero(~batch["masks"])[:, :2]
    ep_starts_cpu = ep_starts.cpu().numpy().tolist()
    for batch_idx in range(len(inds)):
        if [0, batch_idx] not in ep_starts_cpu:
            inserted_start[(0, batch_idx)] = True
            ep_starts_cpu.insert(0, [0, batch_idx])
    eps = []
    ep_starts_cpu = np.array(ep_starts_cpu)

    # Track the next start episode.
    batch_to_next_ep_starts = {}
    for batch_idx in range(len(inds)):
        batch_ep_starts = ep_starts_cpu[ep_starts_cpu[:, 1] == batch_idx][:, 0]
        batch_to_next_ep_starts[batch_idx] = {}
        for i in range(len(batch_ep_starts) - 1):
            batch_to_next_ep_starts[batch_idx][
                batch_ep_starts[i]
            ] = batch_ep_starts[i + 1]

    num_samples = 0
    fetch_before_counts = []
    for step_idx, batch_idx in ep_starts_cpu:
        step_idx_end = min(step_idx + context_len, max_data_window_size)

        next_ep_starts = batch_to_next_ep_starts[batch_idx]

        if step_idx in next_ep_starts:
            ep_intersect = step_idx_end - next_ep_starts[step_idx]
        else:
            ep_intersect = 0

        add_batch = batch.map(lambda x: x[step_idx:step_idx_end, batch_idx])

        att_mask = torch.ones(
            # The att mask needs to have same window size as the data batch.
            (step_idx_end - step_idx, 1),
            dtype=torch.bool,
            device=device,
        )
        assert (
            att_mask.shape[0] == add_batch["masks"].shape[0]
        ), f"Got att mask of shape {att_mask.shape} and masks of shape {add_batch['masks'].shape}, trying to read from range {step_idx}-{step_idx_end}, when step size is {max_data_window_size} cur step idx {current_rollout_step_idx}, orig mask shape {batch['masks'].shape}"

        # Mask out steps that intersect with the next episode
        if ep_intersect > 0:
            att_mask[-ep_intersect:] = False

        # The trajectory should actually have some data.
        assert att_mask.sum() > 0, "Trajectory has no data"

        did_insert_start = inserted_start.get((step_idx, batch_idx), False)
        if did_insert_start:
            fetch_before_counts.append(ep_intersect)
        else:
            fetch_before_counts.append(0)

        # This is the max pad length
        num_samples += context_len

        add_batch["observations"][ATT_MASK_K] = att_mask
        add_batch["observations"][START_HIDDEN_WINDOW_K] = hidden_window[
            :, :, batch_idx, :, :context_len
        ]
        add_batch["observations"][START_ATT_MASK_K] = before_start_att_masks[
            batch_idx
        ].unsqueeze(-1)

        eps.append(add_batch)

        # Could the next seq addition put us over the max count threshold?
        if max_total_samples is not None and num_samples > (
            max_total_samples - context_len
        ):
            # Stop fetching trajectories.
            break

    assert (
        len(eps) > 0
    ), "Collected no episodes from rollout, ensure episode horizon is shorter than rollout length."
    ret_batch = transpose_stack_pad_dicts(eps)

    ret_batch["observations"][FETCH_BEFORE_COUNTS_K] = torch.tensor(
        fetch_before_counts, device=device
    )

    # Ensure every sample is accounted for in this data batch.
    num_samples = ret_batch["observations"][ATT_MASK_K].sum()
    expected_num_samples = len(inds) * max_data_window_size
    # This assert no longer works, because we might truncate the episode collection based on the `max_total_samples`.
    # assert (
    #     expected_num_samples == num_samples
    # ), f"Incorrect number of data samples, got {num_samples}, expected {expected_num_samples}, Ep starts {ep_starts_cpu} data window {max_data_window_size}"
    return TensorDict(ret_batch)
