import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, envs):
        self.replay_buffer = replay_buffer
        self.envs = envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self._action_repeat = int(config.action_repeat)
        batch_steps = int(config.batch_size * config.batch_length)
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * self._action_repeat)
        self._should_pretrain = tools.Once()

    def begin(self, agent):
        envs = self.envs
        step = self.replay_buffer.count() * self._action_repeat
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        episode_ids = torch.arange(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_index = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        state = agent.get_initial_state(envs.env_num)
        action = state["prev_action"].clone()
        while step < self.steps:
            step += int((~done).sum()) * self._action_repeat
            transition, done = envs.step(action.detach(), done)
            transition = transition.to(agent.device, non_blocking=True)
            done = done.to(agent.device)
            action, state = agent.act(transition.clone(), state, eval=False)
            transition["action"] = action * ~done.unsqueeze(-1)
            transition["stoch"] = state["stoch"]
            transition["deter"] = state["deter"]
            transition["episode"] = episode_ids
            transition["episode_index"] = episode_index.clone()
            self.replay_buffer.add_transition(transition.detach())
            episode_index += done
            if step // (envs.env_num * self._action_repeat) > self.replay_buffer.sample_length:
                updates = self.pretrain if self._should_pretrain() else self._updates_needed(step)
                for _ in range(updates):
                    agent.update(self.replay_buffer)
        return step
