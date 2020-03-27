import config as cfg 
from AgentProfile.core import SystemAct
import numpy as np

class SystemTemplate(object):
    def __init__(self):
        self.SYS_TEMPLATE_ACT_DIC = {
            # SystemAct.HAVE_YOU_HEARD_OF_THE_ORG: ["Have you heard of the organization \"Save the Children\"?", 
            #                                       "Are you aware of the organization \"Save the Children\"?"],
            # SystemAct.TASK_RELATED_INQUIRY: ["Have you donated to a charity before?"],
            # SystemAct.PERSONAL_RELATED_INQUIRY: ["Are you aware of the dangerous situations children face in conflicted areas?", 
            #                                      "Do you have kids yourself?"],
            SystemAct.propose_donation_inquiry: ["Would you like to make a donation to Save the Children?",
                                        "Do you want to make a donation to Save the Children?"],
            # SystemAct.ASK_DONATION_AMOUNT: ["How much would you like to donate?"],
            # SystemAct.CONFIRM_DONATION: ["Just to confirm, how much would you like to donate?"],
            #                                                #are you sure you want to make a donation?"],
            # SystemAct.ASK_DONATE_MORE: ["Any chance you would consider making a greater donation?"],
            # SystemAct.THANK: ["Thank you so much!",
            #                   "Thank you a lot!",
            #                   "Thanks so much!"],
            # SystemAct.YOU_ARE_WELCOME: ["You are welcome."],
            # SystemAct.ASK_NOT_DONATE_REASON: ["May I ask why you don't want to donate?"],
            # SystemAct.GREETING: ["Hello there! How are you doing?",
            #                      "I am doing good!"],
            SystemAct.PROVIDE_DONATION_PROCEDURE: ["You can choose any amount from $0 to all your task payment ($2) to donate, which will be directly deducted from your payment; after task completion, the research team will send the donations to Save the Children."], #["Your donation will be directly deducted from your task payment, and you can choose any amount from $0 to all your payment ($2). The research team will collect all donations and send it to \"Save the Children\"."], # It\'s as simple as that."],
            # SystemAct.CLOSING: ["Thank you. It\'s been lovely talking to you. Enjoy your day and bye!"],
            # SystemAct.EXAMPLE_DONATION: ['I\'ll match your donation. And together we can double the amount.',
            #                              'I think I\'ll donate all my task payment to Save the Children. $2.'],
            # SystemAct.PERSONAL_STORY: ['Someone told me that he and his brother replaced birthday gifts with charity donations a few years ago, and it was a really rewarding experience for them.'],
            # for CHI
            # SMALL_TALK_1: ["Well, are you doing anything fun after this?"],
            # SMALL_TALK_2: ["Well, I plan to relax a bit, what do you usually do to relax?"], 
            # SMALL_TALK_3: ["Alright, it’s been a long day. Thanks for sharing."]
            # LAST_PROPOSE: ["Do you still not consider making a donation?"]
        }

    def get_template(self, acts):
        sents = []
        for act in acts:
            sents.append(self.sample(self.SYS_TEMPLATE_ACT_DIC[act]))
        
        return sents

    def sample(self, candidates):
        return np.random.choice(candidates, 1)[0]
