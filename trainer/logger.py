from torch.utils.tensorboard import SummaryWriter
import wandb
from dataclasses import asdict
from rich.console import Console
from rich.pretty import pprint
from types import SimpleNamespace
from logging import StreamHandler, FileHandler, Formatter, INFO, getLogger
import torch
import time
import json
from omegaconf import DictConfig, OmegaConf
import os



import logging


def log_initial_info(args, accelerator, policy):
    if accelerator.is_main_process:

        if accelerator.state.deepspeed_plugin:
            print("DeepSpeed is enabled!")
            print(f"DeepSpeed Plugin: {accelerator.state.deepspeed_plugin}")
            print(f"DeepSpeed Config: {accelerator.state.deepspeed_plugin.deepspeed_config}")
        else:
            print("DeepSpeed is NOT enabled.")

        print("Arguments:\n")
        if isinstance(args, DictConfig):
            config_dict = OmegaConf.to_container(args, resolve=True)
        else:
            config_dict = asdict(args)
        pprint(args)
        print("\nPolicy configuration:\n")
        pprint(policy.config)
    
    return


class WandbLogger:
    def __init__(self, args, accelerator):
        self.args = args
        self.run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
        self.accelerator = accelerator
        self.validation_loss = []
        self.setup()


    def setup(self):
        """Initializes the W&B run."""

        with open("secrets.json", "r") as file:
            secrets = json.load(file)

        # Extract W&B API key
        wandb_api_key = secrets.get("wandb_api_key")
        

            # Log in to W&B
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            raise ValueError("W&B API key is missing from secrets.json")
        
        if isinstance(self.args, DictConfig):
            config_dict = OmegaConf.to_container(self.args, resolve=True)
        else:
            config_dict = asdict(self.args)

        if self.accelerator.is_main_process:
            if self.args.track:
                if self.args.wandb_dir:
                    dir = self.args.wandb_dir
                else:
                    dir = None

                wandb.init(
                    project=self.args.wandb_project_name,
                    sync_tensorboard=True,
                    config=config_dict,
                    name=self.run_name,
                    save_code=True,
                    mode="online",
                    dir=dir)

                wandb.define_metric("training/*", step_metric="actual_step")
                wandb.define_metric("validation/*", step_metric="actual_step")

                if self.accelerator.state.deepspeed_plugin.deepspeed_config:
                    
                    wandb.config.update({"deepspeed_config": str(self.accelerator.state.deepspeed_plugin.deepspeed_config)})


                file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
                wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))


    def finalize(self):
        """Finalizes the wandb run."""
        wandb.finish()


class REFUEL_Logger(WandbLogger):

    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

        self.init_validation_metrics()
        self.init_training_metrics()

    def init_validation_metrics(self):
        self.validation_loss = []
        self.chosen_logprobs = []
        self.reject_logprobs = []
        self.chosen_logprobs_old = []
        self.reject_logprobs_old = []
        self.reward_diff = []

    def init_training_metrics(self):
        self.training_loss = []
        self.training_chosen_logprobs = []
        self.training_reject_logprobs = []

    def add_validation_metrics(self, loss, chosen_logprobs, reject_logprobs, chosen_logprobs_old, reject_logprobs_old, reward_diff):
        self.validation_loss.append(loss)
        self.chosen_logprobs.append(chosen_logprobs)
        self.reject_logprobs.append(reject_logprobs)
        self.chosen_logprobs_old.append(chosen_logprobs_old)
        self.reject_logprobs_old.append(reject_logprobs_old)
        self.reward_diff.append(reward_diff)

    def add_training_metrics(self, loss, chosen_logprobs, reject_logprobs):
        self.training_loss.append(loss)
        self.training_chosen_logprobs.append(chosen_logprobs)
        self.training_reject_logprobs.append(reject_logprobs)


    def log_training(self, step):
        """Logs training metrics."""

        loss = torch.tensor(self.training_loss, device=self.accelerator.device)
        chosen_logprobs = torch.cat(self.chosen_logprobs)
        reject_logprobs = torch.cat(self.reject_logprobs)

        loss = self.accelerator.gather(loss).cpu()
        chosen_logprobs = self.accelerator.gather(self.training_chosen_logprobs).cpu()
        reject_logprobs = self.accelerator.gather(self.training_reject_logprobs).cpu()


        if self.accelerator.is_main_process:
            #new_logprobs = [[chosen_logprobs1, reject_logprobs1], [chosen_logprobs2, reject_logprobs2], ...]
            #take the mean of the chosen and reject logprobs

            metrics = {
                "actual_step": step,  # Add this line
                "training/loss": loss.mean().item(),
                "training/chosen_logprob": chosen_logprobs.mean().item(),
                "training/reject_logprob": reject_logprobs.mean().item(),
                "training/logprob_diff": (chosen_logprobs - reject_logprobs).mean().item(),
                "training/num_samples": len(chosen_logprobs)
            }
            wandb.log(metrics)

            print(f"Training loss: {loss.mean().item()}")
            print(f"Training chosen logprob: {chosen_logprobs.mean().item()}")
            print(f"Training reject logprob: {reject_logprobs.mean().item()}")

        self.init_training_metrics()

        del loss, chosen_logprobs, reject_logprobs

    def log_validation(self, step):
        """Logs validation metrics."""

        loss_tensor = torch.stack(self.validation_loss)
        loss = self.accelerator.gather(loss_tensor).cpu()

        chosen_logprobs_tensor = torch.cat(self.chosen_logprobs)
        chosen_logprobs = self.accelerator.gather(chosen_logprobs_tensor).cpu()

        reject_logprobs_tensor = torch.cat(self.reject_logprobs)
        reject_logprobs = self.accelerator.gather(reject_logprobs_tensor).cpu()

        chosen_logprobs_tensor_old = torch.cat(self.chosen_logprobs_old)
        chosen_logprobs_old = self.accelerator.gather(chosen_logprobs_tensor_old).cpu()

        reject_logprobs_tensor_old = torch.cat(self.reject_logprobs_old)
        reject_logprobs_old = self.accelerator.gather(reject_logprobs_tensor_old).cpu()

        reward_diff_tensor = torch.cat(self.reward_diff)
        reward_diff = self.accelerator.gather(reward_diff_tensor).cpu()


        if self.accelerator.is_main_process:
            #new_logprobs = [[chosen_logprobs1, reject_logprobs1], [chosen_logprobs2, reject_logprobs2], ...]
            #take the mean of the chosen and reject logprobs
           
            metrics = {
                "actual_step": step,  # Add this line
                "validation/loss": loss.mean().item(),
                "validation/chosen_logprob": chosen_logprobs.mean().item(),
                "validation/reject_logprob": reject_logprobs.mean().item(),
                "validation/chosen_kl": (chosen_logprobs - chosen_logprobs_old).mean().item(),
                "validation/reject_kl": (reject_logprobs - reject_logprobs_old).mean().item(),
                "validation/kl": (chosen_logprobs - chosen_logprobs_old - reject_logprobs + reject_logprobs_old).mean().item(),
                "validation/logprob_diff": (chosen_logprobs - reject_logprobs).mean().item(),
                "validation/num_samples": len(chosen_logprobs),
                "validation/reward_diff": reward_diff.mean().item()
            }
            wandb.log(metrics)

            print(f"Validation loss: {loss.mean().item()}")

        del loss, chosen_logprobs, reject_logprobs, chosen_logprobs_old, reject_logprobs_old, reward_diff
        self.init_validation_metrics()


    
        
class DPO_Logger(WandbLogger):

    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

        self.init_validation_metrics()


    def init_validation_metrics(self):
        self.validation_loss = []
        self.chosen_logprobs = []
        self.reject_logprobs = []
        self.chosen_logprobs_old = []
        self.reject_logprobs_old = []


    def add_validation_metrics(self, loss, chosen_logprobs, reject_logprobs, chosen_logprobs_old, reject_logprobs_old):
        self.validation_loss.append(loss)
        self.chosen_logprobs.append(chosen_logprobs)
        self.reject_logprobs.append(reject_logprobs)
        self.chosen_logprobs_old.append(chosen_logprobs_old)
        self.reject_logprobs_old.append(reject_logprobs_old)



    def log_training(self, step):
        """Logs training metrics."""
        pass

    def log_validation(self, step):
        """Logs validation metrics."""

        loss_tensor = torch.stack(self.validation_loss)
        loss = self.accelerator.gather(loss_tensor).cpu()

        chosen_logprobs_tensor = torch.cat(self.chosen_logprobs)
        chosen_logprobs = self.accelerator.gather(chosen_logprobs_tensor).cpu()

        reject_logprobs_tensor = torch.cat(self.reject_logprobs)
        reject_logprobs = self.accelerator.gather(reject_logprobs_tensor).cpu()

        chosen_logprobs_tensor_old = torch.cat(self.chosen_logprobs_old)
        chosen_logprobs_old = self.accelerator.gather(chosen_logprobs_tensor_old).cpu()

        reject_logprobs_tensor_old = torch.cat(self.reject_logprobs_old)
        reject_logprobs_old = self.accelerator.gather(reject_logprobs_tensor_old).cpu()



        if self.accelerator.is_main_process:
            #new_logprobs = [[chosen_logprobs1, reject_logprobs1], [chosen_logprobs2, reject_logprobs2], ...]
            #take the mean of the chosen and reject logprobs
           
            metrics = {
                "actual_step": step,  # Add this line
                "validation/loss": loss.mean().item(),
                "validation/chosen_logprob": chosen_logprobs.mean().item(),
                "validation/reject_logprob": reject_logprobs.mean().item(),
                "validation/chosen_kl": (chosen_logprobs - chosen_logprobs_old).mean().item(),
                "validation/reject_kl": (reject_logprobs - reject_logprobs_old).mean().item(),
                "validation/kl": (chosen_logprobs - chosen_logprobs_old - reject_logprobs + reject_logprobs_old).mean().item(),
                "validation/logprob_diff": (chosen_logprobs - reject_logprobs).mean().item(),
                "validation/num_samples": len(chosen_logprobs)
            }
            wandb.log(metrics)

            print(f"Validation loss: {loss.mean().item()}")

        del loss, chosen_logprobs, reject_logprobs, chosen_logprobs_old, reject_logprobs_old
        self.init_validation_metrics()
   




class GRPO_Logger(WandbLogger):
    
    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

        self.init_validation_metrics()
        self.init_training_metrics()


    def init_validation_metrics(self):
        self.validation_loss = []
        self.rewards = []
        self.rewards_std = []
        self.kl = []
        self.prob_ratio = []

    def init_training_metrics(self):
        self.training_loss = []
        self.training_rewards = []
        self.training_rewards_std = []
        self.training_kl = []
        self.training_prob_ratio = []

    def add_validation_metrics(self, loss, rewards, rewards_std, kl, prob_ratio):
        self.validation_loss.append(loss)
        self.rewards.append(rewards)
        self.rewards_std.append(rewards_std)
        self.kl.append(kl)
        self.prob_ratio.append(prob_ratio)
    
    def add_training_metrics(self, loss, rewards, rewards_std, kl, prob_ratio):
        self.training_loss.append(loss)
        self.training_rewards.append(rewards)
        self.training_rewards_std.append(rewards_std)
        self.training_kl.append(kl)
        self.training_prob_ratio.append(prob_ratio)

    def log_training(self, step):
        """Logs training metrics."""

        loss = torch.tensor(self.training_loss, device=self.accelerator.device)
        rewards = torch.tensor(self.training_rewards, device=self.accelerator.device)
        rewards_std = torch.tensor(self.training_rewards_std, device=self.accelerator.device)
        kl = torch.tensor(self.training_kl, device=self.accelerator.device)
        prob_ratio = torch.tensor(self.training_prob_ratio, device=self.accelerator.device)

        loss = self.accelerator.gather(loss).cpu()
        rewards = self.accelerator.gather(rewards).cpu()
        rewards_std = self.accelerator.gather(rewards_std).cpu()
        kl = self.accelerator.gather(kl).cpu()
        prob_ratio = self.accelerator.gather(prob_ratio).cpu()

        if self.accelerator.is_main_process:
            metrics = {
                "actual_step": step,  # Add this line
                "training/loss": loss.mean().item(),
                "training/rewards": rewards.mean().item(),
                "training/rewards_std": rewards_std.mean().item(),
                "training/kl": kl.mean().item(),
                "training/prob_ratio": prob_ratio.mean().item()
            }
            wandb.log(metrics)

        self.init_training_metrics()

        del loss, rewards, rewards_std, kl, prob_ratio

    def log_validation(self, step):
        """Logs validation metrics."""

        loss_tensor = torch.stack(self.validation_loss)
        loss = self.accelerator.gather(loss_tensor).cpu()

        rewards_tensor = torch.stack(self.rewards)
        rewards = self.accelerator.gather(rewards_tensor).cpu()

        rewards_std_tensor = torch.stack(self.rewards_std)
        rewards_std = self.accelerator.gather(rewards_std_tensor).cpu()

        kl_tensor = torch.stack(self.kl)
        kl = self.accelerator.gather(kl_tensor).cpu()

        prob_ratio_tensor = torch.stack(self.prob_ratio)
        prob_ratio = self.accelerator.gather(prob_ratio_tensor).cpu()

        if self.accelerator.is_main_process:
            metrics = {
                "actual_step": step,  # Add this line
                "validation/loss": loss.mean().item(),
                "validation/rewards": rewards.mean().item(),
                "validation/rewards_std": rewards_std.mean().item(),
                "validation/kl": kl.mean().item(),
                "validation/prob_ratio": prob_ratio.mean().item()
            }
            wandb.log(metrics)

        self.init_validation_metrics()

        del loss, rewards, rewards_std, kl, prob_ratio

    























class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue_background = "\x1b[47m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue_background + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger():
    logger = logging.getLogger(__name__)

    # create console handler with a higher log level
    ch = logging.StreamHandler()

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    return logger
