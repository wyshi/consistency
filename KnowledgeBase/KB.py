import config as cfg
from utils import is_repetition_with_context
import sys
from .template import SystemTemplate
from AgentProfile.core import SystemAct
import itertools
# sys.path.append()

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
        if self.chatbot.turn_i >= cfg.HAVE_TO_PROPOSE:
            # have to propose donation at this turn if it hasn't proposed yet
            enforced_acts = [SystemAct.propose_donation_inquiry, SystemAct.PROVIDE_DONATION_PROCEDURE]
            enforced_templates = self.sys_template.get_template(enforced_acts)
            if SystemAct.propose_donation_inquiry not in self.chatbot.sys_profile.keys():
                # we should enforce rule
                # we should check the enforced templates are not repetition
                if is_repetition_with_context(enforced_templates[0], 
                                              itertools.chain(*self.chatbot.sys_profile.values()), 
                                              threshold=cfg.repetition_threshold):
                    return None
                else:
                    for i, acts in enumerate(sent_act_candidates):
                        for act in acts:
                            if act == SystemAct.propose_donation_inquiry:
                                return i
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
                return None
        
        return None
                
 