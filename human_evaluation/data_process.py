import pandas as pd
from scipy.stats import ttest_ind
import os

baseline_dir = "collected_data/baseline"

columns = ['id', 'exp_year', 'exp_month', 'age', 'why_human', 'effective', 'repeat', 'consis', 'persuasive', 
           'grammar', 'overall', 'interact_again', 'future_donation', 'intention_increase', 'engage', 
           'competent', 'confident', 'warm', 'sincere', 'check', 'human', 'experience', 'usage', 
           'gender', 'race', 'edu', 'marriage', 'kid', 'income', 'religion', 'political'
]
df_baseline = pd.read_csv(f"{baseline_dir}/post_task_survey.csv", names=columns)
sandbox_id = ['AFU4P8DM74I5', 'A2LUQBLVXKKIFF', 'A2MMW0BRDVDXEC', 'A36HO7VSU7ON09', 'A1A6P3VOR53H7D', 'A3BFFI1JV5H7TC']
df = df_baseline[~df_baseline['id'].isin(sandbox_id)]
# drop duplicates
df = df.drop_duplicates(subset='id', keep='first')
# donation
userids = []
donation0 = []
for txt_dir in os.listdir("/home/wyshi/persuasion/test/ParlAI/data/personachat_chat/emnlp_dialogs_txt"):
# for txt_dir in os.listdir(baseline_dir+"/emnlp_dialogs_txt"):
    with open(f"/home/wyshi/persuasion/test/ParlAI/data/personachat_chat/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
    # with open(f"{baseline_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
        if "incomplete" not in txt_dir and "sandbox" not in txt_dir:
            first_line = fh.readline()
            userid = first_line.split(",")[0].split(": ")[1]
            if userid == "AAHVZF0ZP7HMD":
                print(txt_dir)
            if userid in userids:
                print(f"userid: {userid}")
                continue
            userids.append(userid)
            donation0.append(float(first_line.split(",")[1]))
            if float(first_line.split(",")[1]) == -1:
                print(txt_dir, userid)

donation0.remove(-1)
ttest_ind(donation0[2:101], donation0[101:])
ttest_ind(np.array(donation0[2:101])>0, np.array(donation0[101:])>0)

# model1
df1 = pd.read_csv(f"../test/ParlAI/data/personachat_chat/post_task_survey (5:20:20, 11:34 AM).csv", names=columns)
df1 = df1.iloc[df_baseline.shape[0]:]
df1 = df1[~df1['id'].isin(sandbox_id)]

df_check = df[df['check']=="right"]
df1_check = df1[df1['check']=="right"]
ttest_ind(df.repeat, df1.repeat)
ttest_ind(df.consis, df1.consis)
ttest_ind(df.persuasive, df1.persuasive)
ttest_ind(df.grammar, df1.grammar)
ttest_ind(df.overall, df1.overall)
ttest_ind(df.human=="Human", df1.human=="Human")
ttest_ind(df_check.human=="Human", df1_check.human=="Human")