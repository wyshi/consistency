import logging
import copy

class DialogAttribution(object):

    # for 2.2) fake repetition
    QA_PAIR_WITH_UNIQUE_ANSWER_DB = [('ask-donation-procedure', 'provide-donation-procedure')]


class SystemAct(object):

    #################### all inquiries so far ###################
    kids_related_inquiry = "kids-related-inquiry"
    donation_related_inquiry = "donation-related-inquiry"
    organization_related_inquiry = "organization-related-inquiry"
    other_inquiry = "other-inquiry"
    propose_donation_inquiry = "propose-donation-inquiry"
    greeting_inquiry = "greeting-inquiry"
    greeting_answer = "greeting-answer"
    ### other things that contain inquiries

    #############################################################


    ####################### pairs ########################
    PROVIDE_DONATION_PROCEDURE = "provide_donation_procedure"
    PROVIDE_ORG_FACTS = "provide_org_facts"
    ASK_DONATION_AMOUNT = "ask_donation_amount"
    CONFIRM_DONATION = "confirm_donation"
    HAVE_YOU_HEARD_OF_THE_ORG = "have_you_heard_of_the_org"
    ASK_NOT_DONATE_REASON = "ask_not_donate_reason"
    ASK_DONATE_MORE = "ask_donate_more"

    ####################### strategies ########################
    PERSONAL_STORY = "personal_story"#
    EMOTION_APPEAL = "emotion_appeal"#
    PROPOSE_DONATION = "propose_donation"
    FOOT_IN_THE_DOOR = "foot_in_the_door"#
    LOGICAL_APPEAL = "logical_appeal"
    EXAMPLE_DONATION = "example_donation"

    ############ dialog-act ##############
    POSITIVE_TO_INQUIRY = "positive_to_inquiry"
    NEGATIVE_TO_INQUIRY = "negative_to_inquiry"
    NEUTRAL_TO_INQUIRY = "neutral_to_inquiry"

    COMMENT_PARTNER = 'comment_partner'
    PRAISE_USER = "praise_user"
    THANK = "thank"
    YOU_ARE_WELCOME = "you_are_welcome"

    GREETING = "greeting"
    CLOSING = "closing"
    ACKNOWLEDGEMENT = "acknowledgement"

    ############ currently in "other" ##############
    # level 1: other
    OTHER = "other"
    OFF_TASK = "off_task"
