import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import tqdm
import time
import pickle as pkl
import pdb

from torch.utils.tensorboard import SummaryWriter
from apex import amp
from allennlp.training.checkpointer import Checkpointer
from pytorch_pretrained_bert import GPT2Tokenizer, OpenAIAdam, GPT2Model


torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
np.random.seed(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import pdb

class GPT2MultipleChoiceHead(nn.Module):
    """ Multiple Choice Head for the transformer """

    def __init__(self, config):
        super(GPT2MultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        # self.multiple_choice_token = multiple_choice_token
        # self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.first_dropout = nn.Dropout(config.summary_last_dropout)
        self.linear1 = nn.Linear(config.n_embd, 500)
        self.linear2 = nn.Linear(500, config.n_class)
        self.activation = nn.Tanh()
        self.device = config.device
        # self.last_dropout = Identity()#
        # self.loss = nn.CrossEntropyLoss()

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, 0)

        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear2.bias, 0)

    def forward(self, hidden_states, mc_labels=None):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
                        # num_layer * (, , num_heads, seq_length, hidden_size)
        # hidden_state torch.Size([2, 1, 16, 22, 64])
        #              torch.Size([2, 1, 16, 7, 64])
        # mc_token_ids (bsz, num_choices)
        # mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        # multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # (bsz, num_choices, hidden_size)
        # multiple_choice_h = self.dropout(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        output = hidden_states[:, -1, :]
        output = self.first_dropout(output)
        output = self.linear1(output)
        output = self.activation(output)
        output = self.linear2(output)
        output = self.activation(output)
        # output = self.last_dropout(output)

        # self.loss(multiple_choice_logits, target)
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output.view(-1, output.size(-1)), mc_labels.view(-1))
            # output = (loss,) + output
            return loss, output
        
        # (bsz, num_choices)
        return output


def split_train_val():
    import pickle as pkl

    with open("data/imitation_learning/demonstration.pkl", "rb") as fh:
        data = pkl.load(fh)

    data[0]["individual_features"][0]["hidden_states_after_generation"]
    data[0]["individual_features"][0]["pick_or_not"]

    all_data = []
    for turn in data:
        for d in turn["individual_features"]:
            if d["different_from_edition"] != "human_added_sentence":
                # features = d['hidden_states_after_generation']
                features = torch.cat(
                    [
                        d["past_after_generation"][:, :, :, -1, :].reshape(
                            (1, 1, 16 * 64 * 2)
                        ),
                        d["hidden_states_after_generation"],
                    ],
                    dim=2,
                )
                all_data.append([features, d["pick_or_not"]])
                if d["hidden_states_after_generation"].shape[1] != 1:
                    print("here")
            else:
                print(d["hidden_states_after_generation"].shape)
    import random

    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[:754]
    val_data = all_data[754:]

    with open("data/imitation_learning/demonstration_train_with_past.pkl", "wb") as fh:
        pkl.dump(train_data, fh)

    with open("data/imitation_learning/demonstration_val_with_past.pkl", "wb") as fh:
        pkl.dump(val_data, fh)


# split_train_val()

class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.tokenizer = tokenizer
        # self.tokenizer.max_len = 1500
        # self.turn_ending = tokenizer.encode("\n\n\n")
        # self.dialog_ending = [tokenizer.encoder["[EOS]"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def collate(self, unpacked_data):
        return unpacked_data

class ClassifierConfig:
    n_embd = 1024 * 3
    n_class = 2
    summary_last_dropout = 0.2


# build model
model_clf = GPT2MultipleChoiceHead(ClassifierConfig)


# load data

with open("data/imitation_learning/demonstration_train_with_past.pkl", "rb") as fh:
    train_data = pkl.load(fh)

with open("data/imitation_learning/demonstration_val_with_past.pkl", "rb") as fh:
    val_data = pkl.load(fh)

train_dataset = SentenceDataset(train_data)
val_dataset = SentenceDataset(val_data)

batch_size = 16
# collate_func = Collate_Function(tokenizer)

train_dataloader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=train_dataset.collate,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=train_dataset.collate,
)


# ## Define the model
device = torch.device("cuda:2")
torch.cuda.set_device(device)
model_clf = model_clf.to(device)
# model_A = model_A.to(device)
# model_B = model_B.to(device)


def train_one_iter(batch, update_count, fp16=False):
    total, correct = (0, 0)
    try:
        hidden_states = torch.cat(list(map(lambda x: x[0], batch)))
    except:
        pdb.set_trace()
    labels = torch.tensor(list(map(lambda x: int(x[1]), batch))).to(device)

    loss, outputs = model_clf(hidden_states=hidden_states, mc_labels=labels)
    loss /= num_gradients_accumulation
    _, predicted_labels = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted_labels == labels).sum().item()

    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    record_loss = loss.item() * num_gradients_accumulation
    # print("record_loss: {}".format(record_loss))
    # perplexity = np.exp(record_loss)

    return record_loss, correct / total, labels.tolist(), predicted_labels.tolist()


from sklearn.metrics import f1_score


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        correct = 0
        total = 0
        y_true, y_pred = [], []
        for batch in pbar:
            # batch = batch[0]

            hidden_states = torch.cat(list(map(lambda x: x[0], batch)))
            labels = torch.tensor(list(map(lambda x: int(x[1]), batch))).to(device)

            import pdb

            # pdb.set_trace()
            y_true.extend(labels.tolist())
            outputs = model_clf(hidden_states=hidden_states)

            _, predicted_labels = torch.max(outputs, 1)
            y_pred.extend(predicted_labels.tolist())
            # print(predicted_labels)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
        f1 = f1_score(y_true, y_pred, average="binary")
        print(f"Epcoh {ep} Validation accuracy: {correct/total}, f1: {f1}")

        return correct / total, f1



checkpointer = Checkpointer(
    serialization_dir="Checkpoint_clf",
    keep_serialized_model_every_num_seconds=3600 * 2,
    num_serialized_models_to_keep=5,
)


# optimizer
num_epochs = 600
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = (
    len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation * 5
)
print("num_train_optimization_steps: {}".format(num_train_optimization_steps))

optimizer = OpenAIAdam(
    model_clf.parameters(),
    lr=2e-5,
    warmup=0.1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    t_total=num_train_optimization_steps,
)


from tqdm import tqdm as tqdm_bar

update_count = 0
progress_bar = tqdm.tqdm_notebook
start = time.time()
best_acc = -float("Inf")
best_f1 = -float("Inf")
for ep in tqdm_bar(range(num_epochs)):

    "Training"
    pbar = progress_bar(train_dataloader)
    model_clf.train()
    ys_true = []
    ys_pred = []
    for batch in pbar:
        # batch = batch[0]
        # without relative position

        record_loss, acc, y_true, y_pred = train_one_iter(
            batch, update_count, fp16=False
        )
        # print(f"train acc: {acc}")
        ys_true.extend(y_true)
        ys_pred.extend(y_pred)
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            # update for gradient accumulation
            optimizer.step()
            optimizer.zero_grad()

            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end

            # show progress
            pbar.set_postfix(loss=record_loss, accuracy=acc, speed=speed)

    from sklearn.metrics import accuracy_score

    "Evaluation"
    train_f1 = f1_score(ys_true, ys_pred, average="binary")
    train_acc = accuracy_score(ys_true, ys_pred)
    print(f"train acc: {train_acc}, train f1: {train_f1}")
    model_clf.eval()
    val_acc, val_f1 = validate(val_dataloader)

    is_best_so_far = val_acc > best_acc
    if is_best_so_far:
        best_acc = val_acc
        torch.save(
            model_clf.state_dict(),
            f"Checkpoint_clf/best_acc_{best_acc}_f1_{val_f1}_with_past.pth",
        )
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(
            model_clf.state_dict(),
            f"Checkpoint_clf/best_acc_{best_acc}_f1_{best_f1}_with_past.pth",
        )
    checkpointer.save_checkpoint(
        ep, model_clf.state_dict(), {"None": None}, is_best_so_far
    )

print("best acc: {}, best f1: {}".format(best_acc, best_f1))
