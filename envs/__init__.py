from . import parallel, wrappers


def make_envs(config):
    def constructor(index):
        return lambda: make_env(config, index)

    envs = parallel.ParallelEnv(constructor, int(config.env_num), config.device)
    return envs, envs.observation_space, envs.action_space


def make_env(config, index=0, train=True):
    suite, task = config.task.split("_", 1)
    if suite == "carl":
        from . import carl

        env = carl.make(config, index)
    elif suite == "rwrl":
        from . import rwrl

        env = rwrl.make(config, index, train)
    elif suite == "dpmdp":
        from .dpmdp import make
        from .gym_mujoco import GymMuJoCo

        base = make(task, p_stay=float(config.p_stay))
        env = GymMuJoCo(base, int(config.action_repeat), int(config.seed) + index)
    else:
        raise NotImplementedError(suite)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, int(config.time_limit) // int(config.action_repeat))
    return wrappers.Dtype(env)
