# Multiturn LLM Training

A simple and concise framework for training large language models (LLMs) to handle multi-turn conversations.

## Methods Implemented

- [REFUEL](https://github.com/ZhaolinGao/REFUEL)
- [Token-based-DPO](https://arxiv.org/pdf/2404.12358)
- [Offline-GRPO](https://arxiv.org/pdf/2501.12948) *(experimental)*

## Installation and Usage

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configuration

- Create a training configuration file in configs_training/.
- Create an Accelerate configuration file referencing deepspeed_config.json.

### 3. Run Training

For the required format of the dataset and possible configuration settings see the markdowns of the respective training methods
(To-do)

```bash
accelerate launch --config_file=$ACCELERATE_CONFIG_PATH train.py --config-name $TRAINING_CONFIG_NAME
```