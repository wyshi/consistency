import pandas as pd
from scipy.stats import ttest_ind
import os
import numpy as np

def ttest(df1, df2):
    print(ttest_ind(df1.repeat, df2.repeat))
    print(ttest_ind(df1.consis, df2.consis))
    print(ttest_ind(df1.grammar, df2.grammar))
    print(ttest_ind(df1.persuasive, df2.persuasive))
    print(ttest_ind(df1.overall, df2.overall))
    print(ttest_ind(df1.human=="Human", df2.human=="Human"))
    print("donation")
    print(ttest_ind(df1.donation, df2.donation))
    print(f"{df1.donation.mean()}, {df2.donation.mean()}")
    print(ttest_ind(df1.donation>0, df2.donation>0))
    print(f"{(df1.donation>0).mean()}, {(df2.donation>0).mean()}")

baseline_dir = "collected_data/baseline"

columns = ['id', 'exp_year', 'exp_month', 'age', 'why_human', 'effective', 'repeat', 'consis', 'persuasive', 
           'grammar', 'overall', 'interact_again', 'future_donation', 'intention_increase', 'engage', 
           'competent', 'confident', 'warm', 'sincere', 'check', 'human', 'experience', 'usage', 
           'gender', 'race', 'edu', 'marriage', 'kid', 'income', 'religion', 'political'
]
df_baseline = pd.read_csv(f"{baseline_dir}/post_task_survey_full.csv", names=columns)
sandbox_id = ['AFU4P8DM74I5', 'A2LUQBLVXKKIFF', 'A2MMW0BRDVDXEC', 'A36HO7VSU7ON09', 'A1A6P3VOR53H7D', 'A3BFFI1JV5H7TC']
df_baseline = df_baseline[~df_baseline.id.isin(sandbox_id)]
df_baseline = df_baseline.drop_duplicates(subset='id', keep='first')

# get userids
userids = []
donation0 = []
txt_dirs = []
for txt_dir in sorted(os.listdir(f"{baseline_dir}/emnlp_dialogs_txt/")):
    with open(f"{baseline_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
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
            txt_dirs.append(txt_dir)
            # donation_df.append()
            if float(first_line.split(",")[1]) == -1:
                print(txt_dir, userid)

donation_df = pd.DataFrame(zip(userids, donation0, txt_dirs), columns=["id", "donation", "txt_dir"])
donation_df = donation_df[donation_df['donation']>=0]
donation_df = donation_df[~donation_df.id.isin(sandbox_id)]
donation_df.to_csv(f"{baseline_dir}/donation_df.csv", index=None)

df_baseline = df_baseline[df_baseline.id.isin(donation_df.id)]
df_donation_baseline = pd.merge(df_baseline, donation_df, how='inner', on='id')
df_donation_baseline.to_csv(f"{baseline_dir}/baseline_post_survey.csv", index=None)

df0_check = df_donation_baseline[df_donation_baseline.check=='right']
# baseline done ########

# model 1 starts ###
model1_dir = "collected_data/full_model"
df_model1 = pd.read_csv(f"{model1_dir}/post_task_survey_full.csv", names=columns)

userids = []
donation0 = []
txt_dirs = []
for txt_dir in sorted(os.listdir(f"{model1_dir}/emnlp_dialogs_txt/")):
    if txt_dir in os.listdir(f"{baseline_dir}/emnlp_dialogs_txt/"):
        continue
    with open(f"{model1_dir}/emnlp_dialogs_txt/{txt_dir}", "r") as fh:
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
            txt_dirs.append(txt_dir)
            # donation_df.append()
            if float(first_line.split(",")[1]) == -1:
                print(txt_dir, userid)
donation_df1 = pd.DataFrame(zip(userids, donation0, txt_dirs), columns=["id", "donation", "txt_dir"])
donation_df1 = donation_df1[donation_df1['donation']>=0]
donation_df1 = donation_df1[~donation_df1.id.isin(sandbox_id)]
donation_df1.to_csv(f"{model1_dir}/donation_df.csv", index=None)

df_model1 = df_model1[df_model1.id.isin(donation_df1.id)]
df_donation_model1 = pd.merge(df_model1, donation_df1, how='inner', on='id')
df_donation_model1.to_csv(f"{model1_dir}/model1_post_survey.csv", index=None)

df1_check = df_donation_model1[df_donation_model1.check=='right']

# model1 done ###

# model2 starts
df_model2 = pd.read_csv(f"~/persuasion/test/ParlAI/data/personachat_chat/post_task_survey.csv", names=columns)
df_model2 = df_model2.iloc[304:]
df_model2 = df_model2[df_model2.check=="right"]
df_model2.repeat.mean()

userids = []
donation0 = []
txt_dirs = []
for txt_dir in sorted(os.listdir(f"/home/wyshi/persuasion/test/ParlAI/data/personachat_chat/emnlp_dialogs_txt/")):
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
            txt_dirs.append(txt_dir)
            # donation_df.append()
            if float(first_line.split(",")[1]) == -1:
                print(txt_dir, userid)
donation_df2 = pd.DataFrame(zip(userids, donation0, txt_dirs), columns=["id", "donation", "txt_dir"])
donation_df2 = donation_df2[donation_df2['donation']>=0]
donation_df2 = donation_df2[~donation_df2.id.isin(sandbox_id)]
donation_df2 = donation_df2[donation_df2.id.isin(df_model2.id)]

df_model2 = df_model2[df_model2.id.isin(donation_df2.id)]
df_donation_model2 = pd.merge(df_model2, donation_df2, how='inner', on='id')

# model2 done ###

# comparison
ttest(df0_check, df1_check)
ttest(df0_check.iloc[:100], df1_check.iloc[:100])
ttest(df0_check, df1_check[~df1_check.id.isin(sids)])
# ttest(df_donation_baseline, df_donation_model1)






sids = ["A2JS0X3RSHFFSN",
"A1ZYTJAQNS5ICC",
"AAHVZF0ZP7HMD",
"A1LYQQX0VGOM7U",
"A3SGNO9VQJFTGE",
"A3M6WFTCGA96VO",
"A1BVCHJZ7WC4RM",
"A331WPSSIY9UAN",
"A2XFQN3VHO8BZ7",
"A31VIS6N5FNCOV",
"A47LB2B3894MD",
"A1C4JUVC3PGTPW",
"A1251G78LIS2M3",
"A1JOOI3ZA092G0",
"A20BPDX03BQ5KT",
"AY645XMBR8TSQ",
"A331WPSSIY9UAN",
"A2P07TZISWVX2W",
"A3M6WFTCGA96VO",
"A2H6Q22X1V1FTX",
"A1P4OFQPT2I4JT",
"A331WPSSIY9UAN",
"A2AFIK2NOSZ9QQ",
"A18Q27VK32D8M7",
"A8S744ESOB89J",
"AA9A1N3MLWMNF",
"A1RPE3HZZBRGEY",
"A2IJ0C0PFYKSHQ",
"A2ADV7VIX7ZY8X",
"A3W818CX4RRF1",
"A253CZ1PYVQW84",
"ARR3UJTM2ZGK4",
"A3AJTTYI2F9CSF",
"A3BD26W136ESTS",
"A18NWYE5285AUY",
"AZTZX37P2UJW2",
"A1LQQQ8MKXH4QD",
"A28KAXBRI8QWIL",
"A3AD8SPDR5VJ1K",
"AGTIA5T9ET7AA",
"A2F12MSDNGTAU0",
"A3C5II9VDLB5VQ",
"A3I2M2C6KXZGYV",
"AYRJKBYXBBMCJ",
"DORSIN",
"A2ZY1EGEPDMOKW",
"A28DY8NAS5KJ1C",
"A4VXQTDDKZFC3",
"A3MN6AOQO8PT1Y",
"ARKLOO3JWWPD9",
"A3D21KWEL1511E",
"A34TK265INT0KY",
"AKE0VEP9DEW3B",
"A2SBTTY0ZX10ST",
"A2XFQN3VHO8BZ7",
"A2WPHVMLLEV5ZB",
"A1ST592G6ZWM5S",
"A2HL06EX1L2MEG",
"A2ZY1EGEPDMOKW",
"A4VXQTDDKZFC3",
"A139HRQ3TVHY7X",
"A3FXII56OACV22",
"A2WIN1VKM7K0ZW",
"A2O1ZX5UADUX2M",
"ADBTHA9M2OHJ5",
"A1I0C2MAPI53SN",
"A1VA5QI3J1U1AN",
"A1QU6P9ETXMJCW",
"A31VIS6N5FNCOV",
"A1D4C4NYRWMTRQ",
"A3CSB2Y74ZQFOA",
"ALC4JY3L74TRZ",
"AEXEAIKPI0AE4",
"A1B48X7XYMFYN0",
"AENKV952BWE0W",
"A3MPRLOQP8D3DL",
"A34AEHTM13FJYB",
"A1OEH7O06Z8J0M",
"A1POXN67UONZJV",
"A3LCO8WBGHAI4K",
"A114AWIGPPTXE1",
"A2SBTTY0ZX10ST",
"AO4GRK9YCUDBD",
"A19QGOCCXFEK71",
"A1SETQLJM04TZS",
"A3ICUZZKLPICQB",
"A1ADMMKDBXBNL1",
"A3I62460289O85",
"A2HQMB1GSEADL",
"A3ASN79T825BV1",
"A1GMSPWCRCY2KN",
"A27QHI0EQSXRCU",
"A1O1ADBQ6FU5WS",
"A3SZF91DCOQPW3",
"AW03SYNGU10Z",
"A1ABE79PHJV1T4",
"AP99WWRWXO6UI",
"A33L3Y5TOSPI1G",
"A35BMKEHPRNYG7",
"A31YH28X2104DS",
"A1VV8QCMXO470U",
"AUCXMMMNKGS3Y",
"A2ZK5I1MDWNPRU",
"A331WPSSIY9UAN",
"A3W3TEOEEF4Z3X",
"A323ZCSAW42EMY",
"A308OMP709HTJP",
"A2L3X4ZZ0J2DZ2",
"A2WJUDX7VT85RD",
"A29Z7CCK4KBCRU",
"A1I0C2MAPI53SN",
"ANGCLG8HRZXTE",
"A49LI81B5C3XW",
"A2OQFHIHC1QQJF",
"A3447AO10KCZS",
"ASEDOB6O4A0JP",
"A3D7P074MHSRGP",
"A37RLZY7Y1EJ2Q",
"ACDIDJHRJ5THM",
"A3DC7HJXM7OQKV",
"AYCNF0YILY9Q",
"A2RTAW3AP0T15I",
"A34N2LQWZFRJIF",
"A14AQJCF66TDA4",
"A1U8E29BSX6JDW",
"A3SJPJLXDRE1RG",
"A30G8DHVOX7UI7",
"A1ZRFYDF1OOCJ3",
"AM6NPZA9Y1BFW",
"A3BG2S9T5CCGC",
"A30OANDXMGLRYR",
"A1ABE79PHJV1T4",
"A184LB0RXXRQL7",
"A33L3Y5TOSPI1G",
"A1I0C2MAPI53SN",
"A240OTT1T85UGT",
"A34XRTZ6NKTZYA",
"A2WFP6NRKAM2EL",
"A3II2ONBAAHOF",
"A13M3RWXUHALFH",
"A1AZDAD63P6RF8",
"AXFP0LBNIMX31",
"A1CB6ZN8ZXJO2M",
"A14YZOETZIKZV7",
"A322SS69X82CX3",
"A2PZ5DB9WQ7UNZ",
"A2DKW0KPFIH5BK",
"AEBAU4TBXN5K9",
"A4J1ORMYLP3CZ",
"A3C8CJJB89BTEJ",
"A1JXRHY1XGJVWU",
"A2O1ZX5UADUX2M",
"AOB6ZJTDB416G",
"A15YGNTVVBSSY4",
"A2ORC04Z3SEX0R",
"A1SGL0PAOYW7U7",
"A2NDBWMRIM869H",
"A2UZ2UH2W9GE0L",
"A1PNA063MOAPNH",
"A18EM8NGUWCGGA",
"A3GBU9CLGN16LR",
"AR117V5SHH230",
"A133OFB7WTNKCP",
"A2RMSP08CB1GN",
"A32Q8RI7CHRQSE",
"A2L7KCAURG1HBG",
"A1KFK92GEFQ9F5",
"A3EF2PPVXN51B5",
"A3LCO8WBGHAI4K",
"A29XYWTD63Z7BS",
"A11L9K56F9IA1I",
"A304UJAE051J89",
"AVB6A6RZ12Y9G",
"ACDIDJHRJ5THM",
"A1UZCHDDWRTMU",
"AKOL40KW5ZRG5",
"A2IF4TYEH1MUKQ",
"A2NE4E2ECRK1RS"]