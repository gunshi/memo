from functools import partial
import os
from collections import defaultdict

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
import pandas as pd

from habitat import VectorEnv
from torch import Tensor
from numpy import ndarray
from typing import Any, Dict, List, Tuple, Union
import pickle as pkl
import ipdb
from habitat_transformers.tasks.sensors import OBJ_MAPPING



class AC_Evaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """
    def get_key_from_episode_info(self, episode_info, ep_eval_count):
        return  "_".join([
            episode_info.scene_id,
            episode_info.episode_id,
            str(ep_eval_count.get(
                (
                    episode_info.scene_id,
                    episode_info.episode_id,
                ), 0
            ))
        ])


    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
        n_steps = 8200,
        n_demos = -1,
        max_n_eps = -1,
        suffix="",
        is_memory=False,
        fifo_eval=False,
    ):
        VIS_KEYS = set(["head_rgb", "third_rgb"])
        n_envs_visits = np.zeros(envs.num_envs)
        observations = envs.reset()
        next_episodes_info = envs.current_episodes()
        _last_save_value = 0

        selected_eps = set()
        n_envs = envs.num_envs

        for i in range(n_envs):
            _curr_eps = (next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)
            while (_curr_eps in selected_eps):
                envs.call_at(i, "after_update")
                observations[i] = envs.reset_at(i)[0]
                n_envs_visits[i] += 1
                _curr_eps = (envs.current_episodes()[i].scene_id, envs.current_episodes()[i].episode_id)
            selected_eps.add(_curr_eps)

        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent._policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = {}
        test_recurrent_hidden_states["past_key_values"] = torch.zeros(
            (
                agent.actor_critic.num_recurrent_layers,
                2,
                envs.num_envs,
                agent.actor_critic.num_heads,
                n_steps + 1,
                agent.actor_critic.recurrent_hidden_size//agent.actor_critic.num_heads,

            ),
            device=device,
            dtype=torch.float
        )

        segment_length = config.habitat_baselines.rl.policy.main_agent.transformer_config.segment_length
        summary_length = config.habitat_baselines.rl.policy.main_agent.transformer_config.summary_length
        self.segment_length = segment_length
        self.summary_length = summary_length
        test_recurrent_hidden_states["softprompt"] = torch.zeros(
            (
                agent.actor_critic.num_recurrent_layers,
                2,
                envs.num_envs,
                agent.actor_critic.num_heads,
                summary_length*(int(n_steps/segment_length)+2),
                agent.actor_critic.recurrent_hidden_size//agent.actor_critic.num_heads,
            ),
            device=device,
            dtype=torch.float
        )

        should_update_recurrent_hidden_states = (
            np.prod(test_recurrent_hidden_states["past_key_values"].shape) != 0
        )

        not_done_masks = torch.zeros(
            envs.num_envs,
            n_steps + 1,
            1,
            device=device,
            dtype=torch.bool,
        )

        prev_actions = torch.zeros(
            envs.num_envs,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # ipdb.set_trace()
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(envs.num_envs)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = len(sum(envs.call(["episodes"]*envs.num_envs), [])) #sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)

            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()

        stats = defaultdict(partial(defaultdict, list))
        
        loop_counter = 0


        while envs.num_envs:
            attn_list = []
            semantic_img_list = []
            target_list = []
            loop_counter += 1
            peak_memories = []
            for _i in tqdm.tqdm(range(n_steps)):


                if max_n_eps > -1 and ((~not_done_masks[:, :_i+1, 0]).sum(axis=1).detach().cpu().numpy() > max_n_eps).all():
                    break
                _summary_idx = (_i) // self.segment_length
                current_episodes_info = envs.current_episodes()
                if 'one_hot_target_sensor' in batch:
                    batch['one_hot_target_sensor'] = batch['one_hot_target_sensor'].float()
                if 'one_hot_receptacle_sensor' in batch:
                    batch['one_hot_receptacle_sensor'] = batch['one_hot_receptacle_sensor'].float()
                with inference_mode():
                    action_data, outputs = agent.actor_critic.act(
                        batch,
                        {
                            'past_key_values': test_recurrent_hidden_states['past_key_values'][..., 1:_i+1, :],
                            'softprompt': test_recurrent_hidden_states['softprompt'][..., :_summary_idx*self.summary_length, :]
                        },
                        prev_actions,
                        not_done_masks[:, :_i+1],
                        deterministic=False,
                        output_attentions=False,
                    )

                    # temp_list = []
                    # for l in range(4):
                    #     temp_list.append(outputs.attentions[l].detach().cpu().numpy())
                    # temp_list = np.stack(temp_list, axis=0)
                    # attn_list.append(temp_list)
                
                    # semantic_img_list.append(batch['masked_semantic_sensor'].detach().cpu().numpy())
                    # target_list.append(batch['one_hot_target_sensor'].detach().cpu().numpy())
                    # outputs.attentions is a tuple of size 4 where 4 is the number of transformer layers
                    # outputs.attentions[0] is of shape 12, 8, 1, n where 8 is the num_heads, 12 is the batch size and n is the number of stepsbeing attended over by the current obs
                    # we want to plot the attention matrix for each env and for each layer. the value n keeps fluctuating.
                    # it starts from 0, then every 320 steps some summary tokens are created and the next obs attentds to the summary vectors instead of the context of size 320. 
                    # along with the previous summary vectors from older segments of size 320.

                    if n_demos > 0:
                        best_actions = envs.call(["best_action"] * envs.num_envs)
                        for i in np.arange(envs.num_envs)[(~not_done_masks[:, :_i+1, 0]).sum(axis=1).detach().cpu().numpy() <= n_demos]:
                            action_data.actions[i] = best_actions[i]
                    outputs.past_key_values = None
                    if action_data.should_inserts is None:
                        test_recurrent_hidden_states['past_key_values'][..., _i+1, :] = (
                            action_data.rnn_hidden_states['past_key_values']
                        )
                        if action_data.rnn_hidden_states['softprompt'] is not None:
                            print('softprompt')
                            test_recurrent_hidden_states['softprompt'][..., ((_summary_idx)*self.summary_length):(_summary_idx+1)*self.summary_length, :] = (
                                action_data.rnn_hidden_states['softprompt']
                            )

                        prev_actions.copy_(action_data.actions)  # type: ignore
                    else:
                        for i, should_insert in enumerate(
                            action_data.should_inserts
                        ):
                            ipdb.set_trace()
                            assert False
                            if not should_insert.item():
                                continue
                            if should_update_recurrent_hidden_states:
                                test_recurrent_hidden_states['past_key_values'][:, :, i,..., _i+1, :] = action_data.rnn_hidden_states[i]  ## what to do here?
                                # test_recurrent_hidden_states['softprompt'][..., _summary_idx*self.summary_length, :] = (
                                    
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore

                # NB: Move actions to CPU.  If CUDA tensors are
                # sent in to env.step(), that will create CUDA contexts
                # in the subprocesses.
                if is_continuous_action_space(env_spec.action_space):
                    # Clipping actions to the specified limits
                    step_data = [
                        np.clip(
                            a.numpy(),
                            env_spec.action_space.low,
                            env_spec.action_space.high,
                        )
                        for a in action_data.env_actions.cpu()
                    ]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()]

                outputs = envs.step(step_data)

                observations, rewards_l, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                # Note that `policy_infos` represents the information about the
                # action BEFORE `observations` (the action used to transition to
                # `observations`).
                policy_infos = agent.actor_critic.get_extra(
                    action_data, infos, dones
                )
                for i in range(len(policy_infos)):
                    infos[i].update(policy_infos[i])

                observations = envs.post_step(observations)
                batch = batch_obs(  # type: ignore
                    observations,
                    device=device,
                )
                batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

                not_done_masks[:, _i+1] = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device="cpu",
                )

                rewards = torch.tensor(
                    rewards_l, dtype=torch.float, device="cpu"
                ).unsqueeze(1)
                current_episode_reward += rewards
                next_episodes_info = envs.current_episodes()
                n_envs = envs.num_envs

                for info_i, info in enumerate(infos):                                                                   ## IMPORTANT BIT
                    info = pd.json_normalize(info).iloc[0].to_dict()
                    key = self.get_key_from_episode_info(current_episodes_info[info_i], ep_eval_count)

                    if dones[info_i]:
                        for k, v in info.items():
                            stats[key][k].append(v)
                        
                        stats[key]["reward_m"].append(rewards_l[info_i])
                        stats[key]["done"].append(dones[info_i])
                        stats[key]["obs"].append({k: v[info_i].detach().cpu().numpy() for k, v in batch.items() if k not in VIS_KEYS})

                for i in range(n_envs):
                    # Exclude the keys from `_rank0_keys` from displaying in the video
                    key = self.get_key_from_episode_info(current_episodes_info[i], ep_eval_count)

                    # disp_info = {
                    #     k: v for k, v in infos[i].items() if k not in rank0_keys
                    # }
                    disp_info = {}
                    disp_info['obj_names'] = OBJ_MAPPING[batch['one_hot_target_sensor'][i].argmax(dim=0).item()].split('_')[1:]
                    # join the list into a string
                    disp_info['obj_names'] = '_'.join(disp_info['obj_names'])
                    disp_info['previous_episode_success'] = stats.get(key, {}).get('custom_pddl_task_success', [0])[-1]
                    # disp_info['full_success'] = stats.get(key, {}).get('custom_pddl_task_success', 0)
                    # save the action taken
                    # disp_info['action'] = step_data[i]
                    # # also store the log probs
                    # disp_info['log_probs'] = action_data.action_log_probs[i].item()

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # TODO move normalization / channel changing out of the policy and undo it here
                        # add top_down_map to the disp_info
                        disp_info["top_down_map"] = infos[i]["top_down_map"]
                        frame = observations_to_image(
                            {k: v[i] for k, v in batch.items()}, disp_info
                        )
                        disp_info.pop("top_down_map")
                        if not not_done_masks[i, _i+1].item():
                            # The last frame corresponds to the first frame of the next episode
                            # but the info is correct. So we use a black frame
                            final_frame = observations_to_image(
                                {k: v[i] * 0.0 for k, v in batch.items()},
                                disp_info,
                            )
                            final_frame = overlay_frame(final_frame, disp_info)                                                 ## are these wrong too?
                            rgb_frames[i].append(final_frame)
                            # The starting frame of the next episode will be the final element..
                            rgb_frames[i].append(frame)
                        else:
                            frame = overlay_frame(frame, disp_info)
                            rgb_frames[i].append(frame)

            # save peak memory list
            # os.makedirs(os.path.join(config.habitat_baselines.video_dir, "stats"), exist_ok=True)
            # np.save(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}_peak_memory.npy"), np.array(peak_memories))


            # # make the parent directory if it doesn't exist
            # os.makedirs(os.path.join(config.habitat_baselines.video_dir, "stats"), exist_ok=True)
            # # log the attn_list to a file (it is a list of numpy arrays which have different length on the last axis)

            # for np_arr_itr, nparr in enumerate(attn_list):
            #     np.save(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}_{np_arr_itr}.npy"), nparr)
            
            # np.save(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}_semantic.npy"), np.array(semantic_img_list))
            # np.save(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}_target.npy"), np.array(target_list))
            if len(stats) > 10:
                df = pd.DataFrame.from_dict(stats, "index")
                os.makedirs(os.path.join(config.habitat_baselines.video_dir, "stats"), exist_ok=True)
                df.to_csv(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}.csv"))
                with open(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter}_{suffix}.pkl"), "wb") as f:
                    pkl.dump(stats, f)
                _last_save_value += 1 # len(stats) // 200
                # stats = defaultdict(lambda: defaultdict(list))
                stats = defaultdict(partial(defaultdict, list)) 

            # episode ended
            pbar.update()

            for i in range(n_envs):
                episode_stats = {
                    "reward": current_episode_reward[i].item()
                }

                # average the values in stats[key]
                key = self.get_key_from_episode_info(current_episodes_info[i], ep_eval_count)
                multi_episode_info = {k: np.mean(v) for k, v in stats[key].items() if k != 'obs'}

                episode_stats.update(extract_scalars_from_info(multi_episode_info))
                current_episode_reward[i] = 0
                k = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                )

                ep_eval_count[k] += 1
                # use scene_id + episode_id as unique id for storing stats
                stats_episodes[(k, ep_eval_count[k])] = episode_stats

                if len(config.habitat_baselines.eval.video_option) > 0:
                    generate_video(
                        video_option=config.habitat_baselines.eval.video_option,
                        video_dir=config.habitat_baselines.video_dir,
                        # Since the final frame is the start frame of the next episode.
                        images=rgb_frames[i][:-1],
                        episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                        checkpoint_idx=checkpoint_index,
                        metrics=extract_scalars_from_info(multi_episode_info),
                        fps=config.habitat_baselines.video_fps,
                        tb_writer=writer,
                        keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                    )

                    # Since the starting frame of the next episode is the final frame.
                    rgb_frames[i] = rgb_frames[i][-1:]

                gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                if gfx_str != "":
                    write_gfx_replay(
                        gfx_str,
                        config.habitat.task,
                        current_episodes_info[i].episode_id,
                    )
                    
            envs.call(["after_update"] * envs.num_envs)
            observations = envs.reset()
            n_envs_visits += 1
            next_episodes_info = envs.current_episodes()

            selected_eps = set()

            envs_to_pause = []
            for i in range(n_envs):
                if len(selected_eps) + sum(ep_eval_count.values()) >= number_of_eval_episodes * evals_per_ep:
                    envs_to_pause.append(i)
                    continue

                _curr_eps = (next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)
                while (
                    ep_eval_count[_curr_eps] + int(_curr_eps in selected_eps) >= evals_per_ep and \
                    n_envs_visits[i] < envs.number_of_episodes[i] * evals_per_ep
                ):
                    envs.call_at(i, "after_update")
                    observations[i] = envs.reset_at(i)[0]
                    n_envs_visits[i] += 1
                    _curr_eps = (envs.current_episodes()[i].scene_id, envs.current_episodes()[i].episode_id)
                if n_envs_visits[i] >= envs.number_of_episodes[i] * evals_per_ep:
                    envs_to_pause.append(i)
                    continue

                selected_eps.add(_curr_eps)


            n_envs_visits = n_envs_visits[[x not in envs_to_pause for x in range(len(n_envs_visits))]]

            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        df = pd.DataFrame.from_dict(stats, "index")
        os.makedirs(os.path.join(config.habitat_baselines.video_dir, "stats"), exist_ok=True)
        df.to_csv(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter + 1}_{suffix}.csv"))
        with open(os.path.join(config.habitat_baselines.video_dir, "stats", f"latest_{loop_counter + 1}_{suffix}.pkl"), "wb") as f:
            pkl.dump(stats, f)

        """
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."
        """
        # aggregated_stats = {}
        # all_ks = set()
        # for ep in stats_episodes.values():
        #     all_ks.update(ep.keys())
        # for stat_key in all_ks:
        #     aggregated_stats[stat_key] = np.mean(
        #         [v[stat_key] for v in stats_episodes.values() if stat_key in v]
        #     )

        # for k, v in aggregated_stats.items():
        #     logger.info(f"Average episode {k}: {v:.4f}")

        # writer.add_scalar(
        #     "eval_reward/average_reward", aggregated_stats["reward"], step_id
        # )

        # metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        # for k, v in metrics.items():
        #     writer.add_scalar(f"eval_metrics/{k}", v, step_id)


def pause_envs(
    envs_to_pause: List[int],
    envs: VectorEnv,
    test_recurrent_hidden_states: Tensor,
    not_done_masks: Tensor,
    current_episode_reward: Tensor,
    prev_actions: Tensor,
    batch: Dict[str, Tensor],
    rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
) -> Tuple[
    VectorEnv,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[str, Tensor],
    List[List[Any]],
]:
    # pausing self.envs with no new episode
    if len(envs_to_pause) > 0:
        state_index = list(range(envs.num_envs))
        for idx in reversed(envs_to_pause):
            state_index.pop(idx)
            envs.pause_at(idx)

        # indexing along the batch dimensions
        if test_recurrent_hidden_states is not None:
            test_recurrent_hidden_states['past_key_values'] = test_recurrent_hidden_states['past_key_values'][
                    :, :, state_index
            ]
            test_recurrent_hidden_states['softprompt'] = test_recurrent_hidden_states['softprompt'][:, :, state_index, :]                   ## ?


        not_done_masks = not_done_masks[state_index]
        current_episode_reward = current_episode_reward[state_index]
        prev_actions = prev_actions[state_index]

        for k, v in batch.items():
            batch[k] = v[state_index]

        if rgb_frames is not None:
            rgb_frames = [rgb_frames[i] for i in state_index]
        # actor_critic.do_pause(state_index)

    return (
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    )

# print(f"Model 1 Peak Memory: {peak_memory_model1 / (1024 ** 2)} MB")
