import imitation_learning.config as cfg
from imitation_learning.utils import is_repetition_with_context
import sys
from imitation_learning.KnowledgeBase.template import SystemTemplate
from imitation_learning.AgentProfile.core import SystemAct
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
    def __init__(self, chatbot, with_rule):
        self.chatbot = chatbot
        self.sys_template = SystemTemplate()
        self.with_rule = with_rule

    def enforce(self, sents, sent_acts, past):
        """
        return: 
               None: no rule needed,
               int: one candidate selected
               str: no candidate selected, should append the returned sentence to the end
        """
        if not self.with_rule:
            return None
        if cfg.verbose:
            print("\n\n\n--------- rule enforce --------------")
        if self.chatbot.turn_i >= cfg.HAVE_TO_PROPOSE:
            # have to propose donation at this turn if it hasn't proposed yet
            enforced_acts = [SystemAct.propose_donation_inquiry, SystemAct.PROVIDE_DONATION_PROCEDURE]
            enforced_templates = self.sys_template.get_template(enforced_acts)
            enforced_acts_last_ask = [SystemAct.propose_donation_inquiry]
            enforced_templates_last_ask = self.sys_template.get_template(enforced_acts_last_ask)
            import pdb
            # pdb.set_trace()
            usr_labels = self.chatbot.global_profile.history_label
            any_is_agree = any([("agree-donation" in usr_label or "provide-donation-amount" in usr_label) for usr_label in usr_labels])

            if any_is_agree:
                return None

            if (not any_is_agree) and (self.chatbot.turn_i == cfg.HAVE_TO_ASK_AGAIN) and (not (SystemAct.propose_donation_inquiry in sent_acts)):
                return ["Just to confirm, how much would like to donate to the charity now?"], enforced_acts_last_ask

            if (SystemAct.propose_donation_inquiry in self.chatbot.global_profile.sys_world.sent_profile.keys()):
                return None

            if (self.chatbot.global_profile.usr_world.usr_profile[self.chatbot.domain.WANT_TO_DONATE] == self.chatbot.domain.INIT)\
                and (SystemAct.propose_donation_inquiry not in self.chatbot.global_profile.sys_world.sent_profile.keys()):
            # if SystemAct.propose_donation_inquiry not in self.chatbot.global_profile.sys_world.sent_profile.keys():
                # we should enforce rule
                # we should check the enforced templates are not repetition
                is_repetition, repetition_score = is_repetition_with_context(enforced_templates[0], 
                                              itertools.chain(*self.chatbot.global_profile.sys_world.sent_profile.values()), 
                                              threshold=cfg.repetition_threshold)
                if is_repetition:
                    if cfg.verbose:
                        print("case 1")
                        print(enforced_templates[0])
                    return None
                else:
                    # for i, acts in enumerate(sent_act_candidates):
                    for act in sent_acts:
                        if act == SystemAct.propose_donation_inquiry:
                            if cfg.verbose:
                                print("case 2")
                            return True
                    if cfg.verbose:
                        print("case 3")
                    if SystemAct.PROVIDE_DONATION_PROCEDURE in sent_acts:
                        # if already said donation procedure, don't repeat
                        return enforced_templates[:1], enforced_acts[:1]
                    else:
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
                if cfg.verbose:
                    print("case 4")
                return None
        
        if cfg.verbose:
            print("case 5")
        return None
                
 