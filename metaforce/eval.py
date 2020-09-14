import numpy as np
from collections import OrderedDict

from metaforce.pearl.core import logger     # TODO:

def dprint(*args):
    # hacky, but will do for now
    if int(os.environ['DEBUG']) == 1:
        print(args)

def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)

def rollout(env, agent, max_path_length=np.inf, accum_context=False, animated=False, save_frames=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def meta_test(
    env, 
    idx, 
    agent,
    max_path_length,
    max_samples=np.inf,
    max_trajs=np.inf,
    sparse_rewards=False, 
    sample_context=None,
    off_policy=False,
    num_exp_traj_eval=None,
    ):
    assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"

    ### eval train tasks with posterior sampled from the training replay buffer
    env.reset_task(idx)
    paths = []
    
    if off_policy:
        assert sample_context is not None
        context = sample_context(idx)
        agent.infer_posterior(context)
    else:
        agent.clear_z()

    n_steps_total = 0
    n_trajs = 0
    while n_steps_total < max_samples and n_trajs < max_trajs:
        path = rollout(
            env, agent, max_path_length=max_path_length, accum_context=(not off_policy))
        # save the latent context that generated this trajectory
        path['context'] = agent.z.detach().cpu().numpy()
        n_steps_total += len(path['observations'])
        n_trajs += 1
        paths.append(path)
        if num_exp_traj_eval is not None and n_trajs >= num_exp_traj_eval:
            agent.infer_posterior(agent.context)

    if sparse_rewards:
        for p in paths:
            sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
            p['rewards'] = sparse_rewards

    return paths, get_average_returns(paths)