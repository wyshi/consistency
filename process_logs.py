import pandas as pd
import numpy as np
import re
from AgentProfile.core import SystemAct

log_dir = "logs/amt_new_model-with_RL-real-new_strategy_model-real.log"
# log_dir = "logs/amt_new_model-no_RL-real-strategy_on-real.log" # RFS - RL
# log_dir = "logs/amt_new_model-no_RL-real-real.log"
# log_dir = "logs/amt_new_model-no_RL-random_select-real-real-real.log"
log_dir = "logs/amt_baseline_test_app_real_multi-thread-test-new.log"
# "amt_new_model-no_RL-real-real.log"

# INFO:PersuasionInteract:Statistics:
# INFO:PersuasionInteract:total success: 44, total different: 9, turn #: 10
# INFO:PersuasionInteract:total repetition: 56, total inconsistency: 0
# INFO:PersuasionInteract:total selected in success: 18, total selected in failed: 5

lines = []
with open(f"{log_dir}", "r") as fh:
    lines = fh.readlines()

dialogs_stat = []
for i, line in enumerate(lines):
    if "INFO:PersuasionInteract:Statistics:" in line:
        success, different, turn = list(map(int, re.findall(r"[0-9]+", lines[i + 1])))
        rep, inconsis = list(map(int, re.findall(r"[0-9]+", lines[i + 2])))
        select1, select2 = list(map(int, re.findall(r"[0-9]+", lines[i + 3])))
        if (success + rep + inconsis) != turn * 10:
            print(i, success + rep + inconsis - turn * 10)
        dic = {
            "success": success,
            "diff": different,
            "turn": turn,
            "rep": rep,
            "inconsis": inconsis,
            "selected_in_suc": select1,
            "selected_in_fail": select2,
        }
        dialogs_stat.append(dic)


# total success


np.sum(list(map(lambda x: x["success"], dialogs_stat)))
# total candidates#
np.sum(list(map(lambda x: x["success"] + x["inconsis"] + x["rep"], dialogs_stat)))
# total turn#
np.sum(list(map(lambda x: x["turn"], dialogs_stat)))
# total selected in success
np.sum(list(map(lambda x: x["selected_in_suc"], dialogs_stat)))

np.sum(list(map(lambda x: x["rep"], dialogs_stat)))
np.sum(list(map(lambda x: x["inconsis"], dialogs_stat)))


# for strategy
all_num_strategy = 0
all_num_acts = 0
all_unique_acts = 0
all_has_strategy = 0
all_has_strategy_list = []
i = 0
for line in lines:
    if re.match("INFO:PersuasionInteract:.* act:", line):
        i += 1
        acts = line.split("[")[1].split("]")[0].split(",")
        acts = [a.strip().strip("'") for a in acts]
        num_strategy = 0
        has_strategy = 0
        for act in acts:
            if act in SystemAct.strategy_list:
                has_strategy += 1
                break
        for act in acts:
            if act in SystemAct.strategy_list:
                num_strategy += 1
        if has_strategy > 0:
            all_has_strategy_list.append(1)
        else:
            all_has_strategy_list.append(0)
        num_acts = len(acts)
        all_has_strategy += has_strategy
        all_num_strategy += num_strategy
        all_num_acts += num_acts
        # all_unique_acts =

all_num_strategy, all_num_acts
all_num_strategy / all_num_acts

all_has_strategy, i
all_has_strategy / i


# sentence length
import os

all_sent_len = 0
all_num_sents = 0
all_num_dialog = 0
for txt_dir in os.listdir("collected_data/newmodel_strategy_on/emnlp_dialogs_txt/"):
    if "incomplete" not in txt_dir and "sandbox" not in txt_dir:  # and\
        # txt_dir.replace(".txt", ".pkl") in os.listdir("collected_data/amt_data/personachat_chat/dialogs_pkl_randomselect/"):
        with open(
            f"collected_data/newmodel_strategy_on/emnlp_dialogs_txt/{txt_dir}", "r"
        ) as fh:
            lines = fh.readlines()
            sent_len = 0
            num_sents = 0
            for line in lines:
                if line.startswith("(0,"):
                    num_sents += 1
                    sent_len += len(line[5:-2].split())
            all_sent_len += sent_len
            all_num_sents += num_sents
            all_num_dialog += 1

all_sent_len, all_num_sents
all_sent_len / all_num_sents
all_num_sents, all_num_dialog
all_num_sents / all_num_dialog


# sent length for full model
all_sent_len = 0
all_sent_len_list_full_model = []
all_num_sents = 0
all_num_dialog = 0
model1_dir = "collected_data/rl_strategy_on_model"
baseline_dir = "collected_data/baseline"
for txt_dir in sorted(os.listdir(f"{model1_dir}/emnlp_dialogs_txt/")):
    if txt_dir in os.listdir(f"{baseline_dir}/emnlp_dialogs_txt/"):
        continue
    if "incomplete" not in txt_dir and "sandbox" not in txt_dir:
        with open(f"{model1_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
            lines = fh.readlines()
            sent_len = 0
            num_sents = 0
            for line in lines:
                if line.startswith("(0,"):
                    num_sents += 1
                    all_sent_len_list_full_model.append(len(line[5:-2].split()))
                    sent_len += len(line[5:-2].split())
            all_sent_len += sent_len
            all_num_sents += num_sents
        all_num_dialog += 1

all_sent_len, all_num_sents
all_sent_len / all_num_sents
all_num_sents, all_num_dialog
all_num_sents / all_num_dialog


# sent length for RFS - RL
all_sent_len = 0
all_sent_len_list_noRL_model = []
all_num_sents = 0
all_num_dialog = 0
model1_dir = "collected_data/newmodel_strategy_on"
baseline_dir = "collected_data/baseline"
for txt_dir in sorted(os.listdir(f"{model1_dir}/emnlp_dialogs_txt/")):
    if txt_dir in os.listdir(f"{baseline_dir}/emnlp_dialogs_txt/"):
        continue
    if "incomplete" not in txt_dir and "sandbox" not in txt_dir:
        with open(f"{model1_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
            lines = fh.readlines()
            sent_len = 0
            num_sents = 0
            for line in lines:
                if line.startswith("(0,"):
                    num_sents += 1
                    all_sent_len_list_noRL_model.append(len(line[5:-2].split()))
                    sent_len += len(line[5:-2].split())
            all_sent_len += sent_len
            all_num_sents += num_sents
        all_num_dialog += 1

all_sent_len, all_num_sents
all_sent_len / all_num_sents
all_num_sents, all_num_dialog
all_num_sents / all_num_dialog


# sent length for baseline model
all_sent_len = 0
all_sent_len_list_base_model = []
all_num_sents = 0
all_num_dialog = 0
for txt_dir in sorted(os.listdir(f"{baseline_dir}/emnlp_dialogs_txt/")):
    with open(f"{baseline_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
        # with open(f"{baseline_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
        if "incomplete" not in txt_dir and "sandbox" not in txt_dir:
            lines = fh.readlines()
            sent_len = 0
            num_sents = 0
            for line in lines:
                if line.startswith("(0,"):
                    num_sents += 1
                    all_sent_len_list_base_model.append(len(line[5:-2].split()))
                    sent_len += len(line[5:-2].split())
            all_sent_len += sent_len
            all_num_sents += num_sents
            all_num_dialog += 1

all_sent_len, all_num_sents
all_sent_len / all_num_sents
all_num_sents, all_num_dialog
all_num_sents / all_num_dialog


from scipy import stats

stats.ttest_ind(all_sent_len_list_base_model, all_sent_len_list_full_model)

stats.ttest_ind(all_has_strategy_list, all_has_strategy_list_full_model)
