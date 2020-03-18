import config as cfg
from utils import is_repetition_with_context
import sys
from .template import SystemTemplate
from AgentProfile.core import SystemAct
import itertools
# sys.path.append()

# domain attributes status
class Domain(object):
    def __init__(self, domain="persuasion"):
        if domain == "persuasion":
            self.USR = "user"
            self.SYS = "system"

            self.HOW_ARE_YOU = 'how-are-you'
            self.HEARD_OF_THE_ORG = 'heard-of-the-org'
            self.HAVE_KIDS = 'have-kids'
            self.DONATED_BEFORE = 'donated-before'
            self.WANT_TO_DONATE = 'want-to-donate'
            self.DONATION_AMOUNT = 'donation-amount'

            self.ATT_TO_QUESTION = {self.HOW_ARE_YOU: 'how are you?', 
                        self.HEARD_OF_THE_ORG: 'have you heard of the organization?', 
                        self.HAVE_KIDS: 'do you have kids?', 
                        self.DONATED_BEFORE: 'have you donated before?',
                        self.WANT_TO_DONATE: 'do you want to donate?', 
                        self.DONATION_AMOUNT: 'how much do you want to donate?'}

            self.attributes = list(self.ATT_TO_QUESTION.keys())

            self.questions = list(self.ATT_TO_QUESTION.values())


            self.INIT = "init"
            self.YES = "yes"
            self.NO = "no"
            self.NOT_SURE = "not_sure"

            self.attribute_status = [self.INIT, self.YES, self.NO, self.NOT_SURE]


class HumanRule(object):
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.sys_template = SystemTemplate()

    def enforce(self, sent_candidates, sent_act_candidates, past_candidates):
        """
        return: 
               None: no rule needed,
               int: one candidate selected
               str: no candidate selected, should append the returned sentence to the end
        """
        print("\n\n\n--------- rule enforce --------------")
        if self.chatbot.turn_i >= cfg.HAVE_TO_PROPOSE:
            # have to propose donation at this turn if it hasn't proposed yet
            enforced_acts = [SystemAct.propose_donation_inquiry, SystemAct.PROVIDE_DONATION_PROCEDURE]
            enforced_templates = self.sys_template.get_template(enforced_acts)
            if SystemAct.propose_donation_inquiry not in self.chatbot.global_profile.sys_world.sent_profile.keys():
                # we should enforce rule
                # we should check the enforced templates are not repetition
                is_repetition, repetition_score = is_repetition_with_context(enforced_templates[0], 
                                              itertools.chain(*self.chatbot.global_profile.sys_world.sent_profile.values()), 
                                              threshold=cfg.repetition_threshold)
                if is_repetition:
                    print("case 1")
                    print(enforced_templates[0])
                    return None
                else:
                    for i, acts in enumerate(sent_act_candidates):
                        for act in acts:
                            if act == SystemAct.propose_donation_inquiry:
                                print("case 2")
                                return i
                    print("case 3")
                    return enforced_templates, enforced_acts # didn't find appropriate candidates, so we append this sentence 

                # edited_enforced_templates = []
                # edited_enforced_acts = []
                # for template, act in zip(enforced_templates, enforced_acts):
                #     if act == SystemAct.propose_donation_inquiry and \
                #         is_repetition_with_context(template, 
                #                                   itertools.chain(*self.chatbot.sys_profile.values()), 
                #                                   threshold=cfg.repetition_threshold):
                #         pass
                #     else:
                #         edited_enforced_templates.append(template)
                #         edited_enforced_acts.append(act)
                

            else:
                print("case 4")
                return None
        
        print("case 5")
        return None
                
 