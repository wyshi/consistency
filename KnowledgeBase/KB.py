import config as cfg
import sys
from .template import SystemTemplate
from AgentProfile.core import SystemAct
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
            if SystemAct.propose_donation_inquiry not in self.chatbot.sys_profile.keys():
                for i, act in enumerate(sent_act_candidates):
                    if act == SystemAct.propose_donation_inquiry:
                        return i
                return self.sys_template.get_template([SystemAct.propose_donation_inquiry,
                                                       SystemAct.PROVIDE_DONATION_PROCEDURE]) # didn't find appropriate candidates, so we append this sentence 
            else:
                return None
        
        return None
                
 