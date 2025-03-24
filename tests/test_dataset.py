from collections import defaultdict
import logging
import re

import unittest

from datasets import load_dataset
from trl import ModelConfig, TrlParser

from open_r1.configs import GRPOConfig
from open_r1.grpo import GRPOScriptArguments


logger = logging.getLogger(__name__)


class TestDataset(unittest.TestCase):
    def test_dataset_taco(self):
        time_limit_pattern = re.compile(".* second[s]?", re.I)
        memory_limit_pattern = re.compile(".* (megabytes|bytes)", re.I)

        parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config(
            ["--config", "recipes/Qwen2.5-1.5B-Instruct/grpo/config_test_code.yaml"]
        )

        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        # Format into conversation
        def make_conversation(example):
            prompt = []

            if training_args.system_prompt is not None:
                prompt.append({"role": "system", "content": training_args.system_prompt})

            prompt.append({"role": "user", "content": example["question"]})
            return {"prompt": prompt}
        dataset = dataset.map(make_conversation)

        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

        statistics = defaultdict(int)
        for data in dataset[script_args.dataset_train_split]:
            statistics["capacity"] += 1
            if data["time_limit"]:
                statistics["has_time_limit"] += 1
                if time_limit_pattern.match(data["time_limit"]):
                    self.assertTrue(True)
                else:
                    logger.error("time_limit: ", data["time_limit"])
                    self.assertTrue(False)
            if data["memory_limit"]:
                statistics["has_memory_limit"] += 1
                if memory_limit_pattern.match(data["memory_limit"]):
                    self.assertTrue(True)
                else:
                    logger.error("memory_limit: ", data["memory_limit"])
                    self.assertTrue(False)
        print(statistics)

        assert True

if __name__ == "__main__":
    unittest.main()
