from PersuasionInteract import PersuasiveBot



[A:"hello ",   
B:"hello how are you", 
A:"I am good"]

past = None

    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            # breakpoint()
            # [can1: "hi", 'hello', 'how are you'] + [] ---> buffer
            [can1: 'I am good', 1
            can2: 'I am fine', 2] + 
            state: [A:"hello ",    past = model_A("hello", past=None)
                                                B:"hello how are you", 
                                    logits, _ = model_A(can1, past=past)
                                    criterion(logits, reward) #if ppo, old_prob & new_prob
                                                              #if not ppo, reward*log(sent)
                                                ]
            logits, _ = model_A(dial_turn_inputs, past=past)
            model_A(dial_turn_inputs, past=)
            all_logits.append(logits)
            candidates = model_A.generate_candidates()
            target = dial_turn_inputs[1:]
            turn_loss = torch.mean([SequenceLoss(logits, c) for c in candidates+[target]])
        else:
            # breakpoint()
            logits, past = model_B(dial_turn_inputs, past=past)
            all_logits.append(logits)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time
import logging
import numba

#from transformers import WarmupLinearSchedule
from apex.optimizers import FusedLAMB, FusedAdam
# from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM
# from torchfly.criterions import SequenceCrossEntropyLoss
# from torchfly.decode import top_filtering

# logging is important
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# define model
model = GPT2SimpleLM(GPT2SmallConfig)
# load model
model.load_state_dict(torch.load("../supervised_warmup/Checkpoint/model_state_epoch_1.th"))

tokenizer = UnifiedTokenizer()
device = torch.device("cuda")
model = model.to(device)

class Trainer:
    """Reinforcement Learning Trainer
    """
    def __init__(self):
        self.sample_size = 32
        self.maxlen = 128
        self.replay_buffer = ReplayBuffer(self.maxlen)
        self.actor = actor
        
        self.reward_func = CustomRewardFunc(tokenizer)
        
    def collect_generations(self, total_size=64, normalize_reward=True):
        logger.info("Collecting Samples")
        # define storage
        all_generated_sequences = []
        all_old_logprobs = []
        all_rewards = []
        
        while total_size > 0:
            real_sample_size = min(self.sample_size, total_size)
            
            # sample sequences
            generated_sequences, old_logprobs = self.actor.sample_generations(real_sample_size)
            # calculate reward
            rewards = self.reward_func(generated_sequences)
            # add tuple into replay buffer
            all_generated_sequences.extend(generated_sequences)
            all_old_logprobs.extend(old_logprobs)
            all_rewards.extend(rewards)
                    
            # decrease
            total_size -= self.sample_size
            logger.info(f"{total_size} samples remaining")
        
        all_rewards = np.array(all_rewards)

        if normalize_reward:
            final_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        else:
            final_rewards = all_rewards
        
        self.replay_buffer.add(zip(all_generated_sequences, all_old_logprobs, final_rewards))
        
        # logging
        logger.info("Collecting Samples finished!")
        
        return all_rewards


def calculate_old_logprobs(buffer_sequences):

    buffer_old_logprobs = []
    indices = np.arange(len(buffer_sequences))

    with torch.no_grad():
        for i in range(PpoParams.batchsize // PpoParams.mini_batchsize):
            batch_indices = indices[i * PpoParams.mini_batchsize : (i + 1) * PpoParams.mini_batchsize]
            batch_sequences = [buffer_sequences[j] for j in batch_indices]

            # make batch
            batch_sequences = make_batch_sequences(batch_sequences, padding_value=pad_token)
            mask = batch_sequences.ne(pad_token).float()

            # to device
            batch_sequences = batch_sequences.to(device)
            mask = mask.to(device)

            logits, past = model(batch_sequences)

             # prepare the loss func inputs
            logits = logits[:, :-1].contiguous()
            target = batch_sequences[:, 1:].contiguous()
            mask = mask[:, 1:].contiguous()

            sequences_logprobs = - criterion(logits, target, mask)
            old_logprobs = sequences_logprobs.sum(1)

            # store
            buffer_old_logprobs.extend(old_logprobs.tolist())
            
    return buffer_old_logprobs


def train_one_iter(batch, update_count, mode, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    
    past = None
    all_logits = []
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            # breakpoint()
            logits, past = model_A(dial_turn_inputs, past=past)
            all_logits.append(logits)
        else:
            # breakpoint()
            logits, past = model_B(dial_turn_inputs, past=past)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=1) # torch.Size([1, 505, 50260]), 505 = sum of tokens from 21 sentences
    
    
    
    # target
    all_logits = all_logits[:, :-1].contiguous() # torch.Size([1, 504, 50260])
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()# torch.Size([1, 504])
    target_mask = torch.ones_like(target).float()# torch.Size([1, 504])
    
    if False:
        loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce=True) # torch.Size([])
    else:
        loss = unlikelihood_criterion(all_logits, target)
    loss /= num_gradients_accumulation
    
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
        
    record_loss = loss.item() * num_gradients_accumulation
    # print("record_loss: {}".format(record_loss))
    perplexity = np.exp(record_loss)
    
    return record_loss, perplexity




    bot = PersuasiveBot()
    bot.reload()
    user_text = ""
    # signal.signal(signal.SIGINT, signal.default_int_handler)

    MAX_DIALOGS = 2
    i = 0
    while i < MAX_DIALOGS:
        try:
            if bot.past is not None:
                if not cfg.self_play_mode:
                    user_text  = input("user: ")
                else:
                    user_text = None
            else:
                i += 1
                print("\n\n\n")
                print("INIT MEMORY!")
                bot.save()
                bot.reload()
            
            response, [sents_success, sents_failed] = bot.chat(0, user_text)
            if cfg.candidate_select_strategy != cfg.HUMAN_SELECTION:
                if cfg.verbose:
                    bot.global_profile.print()
            
            if response == "ARDM MEMORY RESTARTS!":
                print("ARDM MEMORY RESTARTS!")
            else:
                print("Turn {}".format(bot.turn_i))
                print("system: ", response)
            print("$$$$$$$$$$$$$$$$$$$$$")

        except KeyboardInterrupt:
            bot.save()
            sys.exit()
