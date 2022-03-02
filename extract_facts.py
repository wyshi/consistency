import re
import dialog_config
from AgentProfile.profiles import UsrProfile, SysProfile

# ['provide-org-facts', 'personal-story', 'task-related-inquiry',
# 'provide-donation-procedure', 'non-strategy', 'have-you-heard-of-the-org',
# 'logical-appeal', 'foot-in-the-door', 'personal-related-inquiry',
# 'emotion-appeal', 'example-donation']


# # user
# usr_profile = {"donated_before": [],
# "have_kids": [],
# "other_inquiry": [],
# }

# # sys
# sys_profile = {
#     "have_you_heard_of": [],
#     "propose_donation": [],

# }


def main(test_dial):
    # pred_model = strategy_model(model_to_load="./classifier/best_model_state_er.pkl")
    # sys_profile, usr_profile = {}, {}
    usr_profile = UsrProfile()
    sys_profile = SysProfile()

    # initialize params
    context = "<Start>"
    turn_i = 0
    cnt = 0
    while cnt < len(test_dial):
        sys_text = test_dial[cnt][2:]  # input("system input:")
        cnt += 1
        sys_label = sys_profile.update(sys_text, context, turn_i)
        # predicted_label = pred_model.predict(text=sys_text, his=context, turn=turn_i)
        # # print(predicted_label)
        # sys_profile, sys_label = update_sys(sys_profile, sys_text, predicted_label)

        if cnt < len(test_dial):
            usr_text = test_dial[cnt][2:]  # input("user input:")
            cnt += 1
            usr_profile.update(usr_text, sys_label)
            # usr_profile = update_usr(usr_profile, usr_text, sys_label)

            # update params
            context = usr_text
            turn_i += 1

    print("---------- profiles ---------")
    print(sys_profile())
    print(usr_profile())


if __name__ == "__main__":
    test_dial = [
        "A:Hello! How are you doing today?",
        "B:Good, thank you. How are you?",
        "A:I am good, thank you.  Have you heard of the charity Save the Children?",
        "B:Yes, I have heard about it. I would like to donate a dollar to it.",
        "A:That is wonderful, it is so important to provide education and support to the children.  I am sure you will agree that this is a great cause.  Do you donate to charities?",
        "B:Yes, I do donate to several charities.",
        "A:Are you sure you want to donate a dollar to Save the Children?  Children all over the world are suffering due to poverty, war, and other issues.  Would you consider donating a little bit of your incoming task payment?",
        "B:I may donate a dollar.  I don't have a lot of money right now but I would like to help.",
        "A:That is great to hear.  I know your donation is a small amount but the impact this will have on the world can be significant.  Your donation would be directly deducted from your task payment.",
        "B:That is true. I will donate 0.10 dollar to the charity.",
        "A:Thank you so much for your donation, it will help many children.",
        "B:Thank you.  I hope you will continue to donate to the charity.",
    ]

    main(test_dial)
