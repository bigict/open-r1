import functools
import logging
from typing import Optional

import unittest

import datasets
from datasets import load_dataset
import torch
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from open_r1.configs import GRPOConfig
from open_r1.rewards import get_reward_funcs
from open_r1.grpo import GRPOScriptArguments
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks


logger = logging.getLogger(__name__)


def _prediction_step_taco(
    self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None
):
    # inputs = self._prepare_inputs(inputs)
    ################
    # Process inputs
    ################
    # print("xxxx", inputs)
    print("xxxxx")
    return None, None, None

class ComputeLoss(unittest.TestCase):
    def test_compute_loss(self):
        parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config(
            ["--config", "recipes/Qwen2.5-1.5B-Instruct/grpo/config_test_code.yaml"]
        )

        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

        ################
        # Load tokenizer
        ################
        tokenizer = get_tokenizer(model_args, training_args)

        # Get reward functions from the registry
        reward_funcs = get_reward_funcs(script_args)

        logger.info("*** Initializing model kwargs ***")
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        training_args.model_init_kwargs = model_kwargs


        #############################
        # Initialize the GRPO trainer
        #############################
        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            reward_funcs = reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split].select(range(64)),
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            processing_class=tokenizer,
        )

        setattr(
            trainer,
            "prediction_step",
            functools.partial(_prediction_step_taco, trainer)
        )

        train_dataloader = trainer.get_train_dataloader()
        trainer.prediction_loop(train_dataloader, "test_prediction_step_taco")
        assert True

if __name__ == "__main__":
    unittest.main()
