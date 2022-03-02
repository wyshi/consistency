import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import regex as re
import random
import itertools
import tqdm
import time

import warnings

warnings.filterwarnings("ignore")

from gpt_model import GPT2SimpleLM
from pytorch_pretrained_bert import GPT2Tokenizer

# In[3]:

tokenizer_dir = (
    "/home/wyshi/persuasion/consistency/ARDM/persuasion/special3_gpt2_tokenizer.pkl"
)
model_dir = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"


class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = tokenizer.encode("\n\n\n")
        self.dialog_ending = [tokenizer.encoder["[EOS]"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dial_tokens = [
            tokenizer.encode(item) + self.turn_ending for item in self.data[index]
        ]
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        dial_tokens[-1] = dial_tokens[-1][:-2] + self.dialog_ending
        return role_ids, dial_tokens


class Collate_Function:
    """This function handles batch collate."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.EOS = self.tokenizer.encoder["[EOS]"]

    def __call__(self, unpacked_data):
        return unpacked_data


# In[4]:


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
        top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
            whose total probability mass is greater than or equal to the threshold top_p.
            In practice, we select the highest probability tokens whose cumulative probability mass exceeds
            the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -float("Inf"),
            logits,
        )
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill_(
            sorted_indices_to_remove, filter_value
        )
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits


tokenizer = torch.load(tokenizer_dir)


class GPT2SmallConfig:
    vocab_size = 50257 + len(tokenizer.__special_tokens__)
    n_special = len(tokenizer.__special_tokens__)
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = False


class GPT2MediumConfig:
    vocab_size = 50257 + len(tokenizer.__special_tokens__)
    n_special = len(tokenizer.__special_tokens__)
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1024
    n_layer = 24
    n_head = 16
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True


class PersuasiveBot:
    def __init__(self):
        self.tokenizer = torch.load(tokenizer_dir)
        # In[10]:

        # model_A = GPT2SimpleLM(GPT2SmallConfig)
        # model_B = GPT2SimpleLM(GPT2SmallConfig)
        # model_A_states, model_B_states = torch.load("CheckpointMedium/model_state_epoch_3.th")

        # define the model
        self.model_A = GPT2SimpleLM(GPT2MediumConfig)
        self.model_B = GPT2SimpleLM(GPT2MediumConfig)

        # load the model
        model_A_states, model_B_states = torch.load(model_dir)
        self.model_A.load_state_dict(model_A_states)
        self.model_B.load_state_dict(model_B_states)

        self.device = torch.device("cuda")
        self.model_A = self.model_A.to(self.device)
        self.model_B = self.model_B.to(self.device)

        self.model_A.eval()
        self.model_B.eval()

        self.eos = self.tokenizer.encode("\n\n\n")
        self.temperature = 0.7

        # Memory
        self.past = None
        print("inited")

    def chat(self, input_text, sid):
        sid = 0

        if self.past is None:
            prev_input = self.tokenizer.encode("A:")
            prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)

            "Sampling based method"
            sent = []
            with torch.no_grad():
                for i in range(200):
                    logits, self.past = self.model_A(prev_input, past=self.past)
                    logits = logits[:, -1, :] / self.temperature
                    logits = top_filtering(logits, top_k=500, top_p=0.9)
                    # prev_input = logits.argmax(-1).unsqueeze(1)
                    probs = F.softmax(logits, -1)
                    prev_input = torch.multinomial(probs, num_samples=1)
                    prev_word = prev_input.item()

                    if prev_word == 628:
                        break
                    elif prev_word == self.tokenizer.encoder["[EOS]"]:
                        self.past = None
                        return "ARDM MEMORY RESTARTS!"
                        break
                    else:
                        sent.append(prev_word)

            # print("A:" + tokenizer.decode(sent))
            # finish tail
            prev_input = torch.LongTensor(self.eos).unsqueeze(0).to(self.device)
            _, self.past = self.model_A(prev_input, past=self.past)

            return self.tokenizer.decode(sent)
        else:
            if input_text == "quit":
                self.past = None
                return "ARDM MEMORY RESTARTS!"

            user = self.tokenizer.encode("B:" + input_text)
            prev_input = user + self.eos
            prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)

            _, self.past = self.model_B(prev_input, past=self.past)

            # start A's utterance
            suffix = self.tokenizer.encode("A:")
            prev_input = torch.LongTensor(suffix).unsqueeze(0).to(self.device)

            "Sampling based method"
            sent = []
            with torch.no_grad():
                for i in range(200):
                    logits, self.past = self.model_A(prev_input, past=self.past)
                    logits = logits[:, -1, :] / self.temperature
                    logits = top_filtering(logits, top_k=500, top_p=0.9)
                    # prev_input = logits.argmax(-1).unsqueeze(1)
                    probs = F.softmax(logits, -1)
                    prev_input = torch.multinomial(probs, num_samples=1)
                    prev_word = prev_input.item()

                    if prev_word == 628:
                        break
                    elif prev_word == self.tokenizer.encoder["[EOS]"]:
                        self.past = None
                        return "ARDM MEMORY RESTARTS!"
                        break
                    else:
                        sent.append(prev_word)

            # print("A:" + tokenizer.decode(sent))
            # finish tail
            prev_input = torch.LongTensor(self.eos).unsqueeze(0).to(self.device)
            _, self.past = self.model_A(prev_input, past=self.past)

            return self.tokenizer.decode(sent)

    def reload(self):
        self.past = None
        print("reloaded")


if __name__ == "__main__":
    bot = PersuasiveBot()
    user_text = ""

    while True:
        if bot.past is not None:
            user_text = input("B: ")
        else:
            print("INIT MEMORY!")

        response = bot.chat(user_text, 0)

        if response == "ARDM MEMORY RESTARTS!":
            print("ARDM MEMORY RESTARTS!")
        else:
            print("A: ", response)
