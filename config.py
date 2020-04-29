model_clf_dir = "Checkpoint_act_clf/epoch7_multitask_TF_best_acc_0.7944444444444444_f1_0.7861271676300577_A_acc_0.687741935483871_f1_0.6602596916886914_B_acc_0.6437699680511182_f1_0.6186370327752058.pth"#"Checkpoint_act_clf/multitask_TF_best_acc_0.7777777777777778_f1_0.776536312849162_A_acc_0.6954838709677419_f1_0.6707423935799665_B_acc_0.6166134185303515_f1_0.5898033645875225.pth"
model_clf_device1 = "cuda:0"
model_clf_device2 = "cuda:0"

rl_finetune = True

self_play_mode = "self_play_mode"
supervised_mode = "supervised_mode"
interactive_mode = "interactive_mode"
mode = interactive_mode

max_sequence_len = 200
model_size = "medium"
use_old_model = True
old_medium_model_dir = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
old_tokenizer_dir = "/home/wyshi/persuasion/consistency/ARDM/persuasion/special3_gpt2_tokenizer.pkl"
new_medium_model_dir = "models/persuasion-gpt2-medium.pth"
new_small_model_dir = "models/persuasion-gpt2-small.pth"

debug = True
verbose = False
print_candidates = True
repetition_threshold = 0.5
similarity_threshold = 0.707

# select_candidates_strategy
RANDOM_SELECT = "random_select"
REPETITION_RATIO = "repetition_ratio"
FIRST_OF_CANDIDATES = "first_of_candidates"
HUMAN_SELECTION = "human_selection"
IMITATION_LEARNING_SELECTION = "imitation_learning_selection"
candidate_select_strategy = IMITATION_LEARNING_SELECTION#HUMAN_SELECTION#FIRST_OF_CANDIDATES#HUMAN_SELECTION

# num of candidates to select from
NUM_CANDIDATES = 10
MAX_NUM_CANDIDATES = 20


# human rule configuration
HAVE_TO_PROPOSE = 3

PASS = "pass"
REPETITION = "repetition"
INCONSISTENCY = "inconsistency"
NOT_PASS = "not_pass" # (when there are multiple utterances in one sentence, e.g. ABC, 
                      # A maybe repetition, B maybe contradiction, C maybe repetition, 
                      # then return "NOT_PASS" instead of indivisual conflict status)
SYSTEM_CORRECTION = "system_correction"

domain = "persuasion"



if use_old_model:
    log_file = 'logs/old_model/example.log'
    demonstration_csv = "demonstration/old_model/demonstration.csv"
    demonstration_pkl = "demonstration/old_model/demonstration.pkl"
else:
    log_file = 'logs/new_model/example.log'
    demonstration_csv = "demonstration/new_model/demonstration.csv"
    demonstration_pkl = "demonstration/new_model/demonstration.pkl"
