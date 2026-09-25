from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    torch.set_float32_matmul_precision("high")
    envs, obs_space, act_space = make_envs(config.env)
    try:
        agent = Dreamer(config.model, obs_space, act_space).to(config.device)
        replay = Buffer(config.buffer)
        OnlineTrainer(config.trainer, replay, envs).begin(agent)
        output = Path(config.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": agent.state_dict(),
            "config": OmegaConf.to_container(config.model, resolve=True),
        }, output)
    finally:
        envs.close()


if __name__ == "__main__":
    main()
