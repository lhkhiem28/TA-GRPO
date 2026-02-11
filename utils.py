import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_map_fn(split):
    def process_fn(example, index):
        instruction_following = "You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{}. Please stop generating immediately after outputting the box."

        question = example.pop("question")
        if len(question) == 0:
            type = "pad"
        else:
            type = "train"
        question = instruction_following + "\n" + question

        if "sub_index" in example.keys():
            index = example.pop("index")
            sub_index = example.pop("sub_index")
        else:
            sub_index = -1
        data = {
            "data_source": "lighteval/MATH",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example.pop("answer")},
            "extra_info": {"split": split, "index": index, "sub_index": sub_index, "type": type},
        }
        return data

    return process_fn