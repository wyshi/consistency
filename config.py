class Config:
    dialogAct_model_dir = "./classifier/best_model_state_er.pkl"

debug = True
repetition_threshold = 0.5

# select_candidates_strategy
RANDOM_SELECT = "random_select"
REPETITION_RATIO = "repetition_ratio"
FIRST_OF_CANDIDATES = "first_of_candidates"
candidate_select_strategy = FIRST_OF_CANDIDATES

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

log_file = 'example.log'