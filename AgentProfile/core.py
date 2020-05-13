import logging
import copy

class DialogAttribution(object):

    # for 2.2) fake repetition
    QA_PAIR_WITH_UNIQUE_ANSWER_DB = [('ask-donation-procedure', 'provide-donation-procedure')]

class SharedDialogAct(object):
    # inquiries
    greeting_inquiry = "greeting-inquiry"    
    kids_related_inquiry = "kids-related-inquiry"
    donation_related_inquiry = "donation-related-inquiry"
    other_inquiry = "other-inquiry"
    greeting_answer = "greeting-answer"

    # non-inquiry
    PERSONAL_RELATED_INQUIRY = 'personal-related-inquiry'
    TASK_RELATED_INQUIRY = 'task-related-inquiry'
    POSITIVE_TO_INQUIRY = "positive-to-inquiry"
    NEGATIVE_TO_INQUIRY = "negative-to-inquiry"
    NEUTRAL_TO_INQUIRY = "neutral-to-inquiry"

    GREETING = "greeting"
    ACKNOWLEDGEMENT = "acknowledgement"
    YOU_ARE_WELCOME = "you-are-welcome"
    THANK = "thank"
    CONFIRM_DONATION = "confirm-donation"
    CLOSING = "closing"

    OTHER = "other"
    OFF_TASK = "off-task"

class SystemAct(object):

    #################### all NEW inquiries so far ###################
    organization_related_inquiry = "organization-related-inquiry"
    propose_donation_inquiry = "propose-donation-inquiry"
    kids_related_inquiry = SharedDialogAct.kids_related_inquiry
    donation_related_inquiry = SharedDialogAct.donation_related_inquiry    
    other_inquiry = SharedDialogAct.other_inquiry    
    greeting_inquiry = SharedDialogAct.greeting_inquiry
    greeting_answer = SharedDialogAct.greeting_answer
    ### other things that contain inquiries

    #############################################################

    #################### original dialog acts ###################
    ####################### pairs ########################
    PROVIDE_DONATION_PROCEDURE = "donation-information"
    ASK_DONATION_AMOUNT = "ask-donation-amount"
    HAVE_YOU_HEARD_OF_THE_ORG = "source-related-inquiry"
    ASK_NOT_DONATE_REASON = "ask-not-donate-reason"
    ASK_DONATE_MORE = "'ask-donate-more"
    COMMENT_PARTNER = 'comment-partner'
    PRAISE_USER = "praise-user"

    ####################### strategies ########################
    PROVIDE_ORG_FACTS = "credibility-appeal"
    PERSONAL_STORY = "personal-story"#
    EMOTION_APPEAL = "emotion-appeal"#
    PROPOSE_DONATION = "proposition-of-donation"
    FOOT_IN_THE_DOOR = "foot-in-the-door"#
    LOGICAL_APPEAL = "logical-appeal"
    EXAMPLE_DONATION = "self-modeling"

    ############ dialog-act ##############
    TASK_RELATED_INQUIRY = SharedDialogAct.TASK_RELATED_INQUIRY
    PERSONAL_RELATED_INQUIRY = SharedDialogAct.PERSONAL_RELATED_INQUIRY
    POSITIVE_TO_INQUIRY = SharedDialogAct.POSITIVE_TO_INQUIRY
    NEGATIVE_TO_INQUIRY = SharedDialogAct.NEGATIVE_TO_INQUIRY
    NEUTRAL_TO_INQUIRY = SharedDialogAct.NEUTRAL_TO_INQUIRY

    CONFIRM_DONATION = SharedDialogAct.CONFIRM_DONATION
    THANK = SharedDialogAct.THANK
    YOU_ARE_WELCOME = SharedDialogAct.YOU_ARE_WELCOME

    GREETING = SharedDialogAct.GREETING
    CLOSING = SharedDialogAct.CLOSING
    ACKNOWLEDGEMENT = SharedDialogAct.ACKNOWLEDGEMENT

    ############ currently in "other" ##############
    # level 1: other
    OTHER = SharedDialogAct.OTHER
    OFF_TASK = SharedDialogAct.OFF_TASK


class UserAct(object):

    #################### all NEW inquiries so far ###################
    organization_info_inquiry = "organization-info-inquiry"
    donation_procedure_inquiry = "donation-procedure-inquiry"
    persuader_intention_inquiry = "persuader-intention-inquiry"
    kids_related_inquiry = SharedDialogAct.kids_related_inquiry
    donation_related_inquiry = SharedDialogAct.donation_related_inquiry    
    other_inquiry = SharedDialogAct.other_inquiry    
    greeting_inquiry = SharedDialogAct.greeting_inquiry
    greeting_answer = SharedDialogAct.greeting_answer
    ### other things that contain inquiries

    #############################################################

    #################### original dialog acts ###################
    ####################### pairs ########################
    NEGATIVE_REACTION_TO_DONATION = 'negative-reaction-to-donation'
    NEUTRAL_REACTION_TO_DONATION = 'neutral-reaction-to-donation'
    POSITIVE_REACTION_TO_DONATION = 'positive-reaction-to-donation'
    DISAGREE_DONATION_MORE = 'disagree-donation-more'
    PROVIDE_DONATION_AMOUNT = 'provide-donation-amount'
    ASK_ORG_INFO = 'ask-org-info'
    DISAGREE_DONATION = 'disagree-donation'
    ASK_DONATION_PROCEDURE = 'ask-donation-procedure'
    ASK_DONATION_INTENTION = 'ask-persuader-donation-intention'
    AGREE_DONATION = 'agree-donation'

    ############ shared ##############
    TASK_RELATED_INQUIRY = SharedDialogAct.TASK_RELATED_INQUIRY
    PERSONAL_RELATED_INQUIRY = SharedDialogAct.PERSONAL_RELATED_INQUIRY
    POSITIVE_TO_INQUIRY = SharedDialogAct.POSITIVE_TO_INQUIRY
    NEGATIVE_TO_INQUIRY = SharedDialogAct.NEGATIVE_TO_INQUIRY
    NEUTRAL_TO_INQUIRY = SharedDialogAct.NEUTRAL_TO_INQUIRY

    CONFIRM_DONATION = SharedDialogAct.CONFIRM_DONATION
    THANK = SharedDialogAct.THANK
    YOU_ARE_WELCOME = SharedDialogAct.YOU_ARE_WELCOME

    GREETING = SharedDialogAct.GREETING
    CLOSING = SharedDialogAct.CLOSING
    ACKNOWLEDGEMENT = SharedDialogAct.ACKNOWLEDGEMENT

    ############ currently in "other" ##############
    # level 1: other
    OTHER = SharedDialogAct.OTHER
    OFF_TASK = SharedDialogAct.OFF_TASK
