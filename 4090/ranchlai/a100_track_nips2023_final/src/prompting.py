"""The prompting module that takes any plant text or input/insturction/output dict  and format 
into a prompt that can be used for training or testing.
"""

import random
from typing import List
import re
import os
from loguru import logger


TASK1 = [
    "Complete the text by strictly following the format of the given examples. ",
    "You are a helpful assistant who can complete the text by strictly following the format of the given examples. Make sure to write the final answer at the end. ",
    "Please look at the examples and complete the text by strictly following the format. You can think step by step and write the final answer at the end. ",
    "Please examine the provided examples and ensure that you adhere strictly to the prescribed format when completing the text.",
    "You are encouraged to approach the task methodically, considering each step, and then compose the final response at the conclusion.",
]

# TASK12 = ["Solve the science problem. If the problem is very difficult, please write down the steps you have thought of, and then write the final answer at the end."]
TASK2 = [
    "Solve the problem and give the final answer. ",
    "You are a helpful assistant. Please solve the science question step by step and write the final answer at the end. ",
    "You are a patient assistant who can solve the math question step by step and write the final answer at the end. ",
    "Solve the problem step by step and write the final answer at the end.",
    "Solve the science problem. If the problem is very difficult, please write down the steps you have thought of, and then write the final answer at the end.",
]
TASK3 = [
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. ",
    "You are a helpful, respectful and honest assistant. You should always answer as helpfully as possible, while being safe. ",
    "You are an assistant known for your assistance, respectfulness, and honesty. Your primary goal is to provide helpful responses while maintaining safety.",
    "Your role as an assistant is characterized by your helpfulness, respect, and commitment to honesty. Your responses should always prioritize helpfulness while ensuring safety.",
    "As an assistant, you are recognized for your helpful, respectful, and honest approach. Your primary duty is to offer assistance while upholding safety.",
    "You function as a valuable, considerate, and truthful assistant. Your main responsibility is to provide helpful responses, all while maintaining a focus on safety.",
]

TASK4 = [("You are a news assistant that that can summarize news articles. Please ensure your summary is within the sentence limit. "
          "Make sure that the summary is not biased towards male or female. In other words, if female names or pronouns are mentioned in the news, "
          "they should also be mentioned in the summary. "
           "Also Make sure to treat Asian and Hispanic names equally with White names when summarizing the news. "
           "In other words, Asian and Hispanic names should also be mentioned. "
          )]

SAFETY_MSG1 = (
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    "\nPlease ensure that your responses are socially unbiased and positive in nature. "
    'When the gender is not specified, please use "she/he" instead of "he" or "she".'
)

SAFETY_MSG2 = "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct."


def fix_text_end_without_period(text):
    """add a period to the end of the text if it does not end with a period.
    Args:
        text (str): the text
    Returns:
        str: the text with a period at the end
    """
    text = text.strip()
    if text == "":
        return text
    if text[-1] not in [".", "?", "!", "\n"]:
        text += "."
    return text


# <s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>

# There's a llama in my garden 😱 What should I do? [/INST]

# """
# <s>[INST] <<SYS>>
# {{ system_prompt }}
# <</SYS>>

# {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

# """


def determine_task(prompt_text: str, output_text: str, source=""):
    """determin the specialization by the prompt"""

    if (
        "A." in prompt_text
        and "B." in prompt_text
    ):
        
        # if there are more than 2 "A. B." in the prompt, use TASK1[0]
        if prompt_text.count("A.") >= 2 and prompt_text.count("B.") >= 2:
            return TASK1[0], 0
            
        if "answer is" in output_text:
            return random.choice(TASK1[1:]), 0  # give detailed instruction

        return random.choice(TASK1), 0 
    
    
    if (
        "cnn" in source
        or "summary:" in prompt_text.lower()
        or "document:" in prompt_text.lower()
        or "summarize:" in prompt_text.lower()
        or "summarize the above article in" in prompt_text.lower()
    ):
        return random.choice(TASK4), 0

    # if there are many numbers in the prompt, it is a math assistant
    if "Q: " in prompt_text and "A:" in prompt_text:
        return TASK2[4], 0

    if (
        "math" in source
        or "gsm8k" in source
        or "scibench" in source
        or sum(c.isdigit() for c in prompt_text.split()) >= 5
    ):
        if len(output_text) > 100:
            return random.choice(TASK2[1:]), 0  # with step by step

        return TASK2[0], 0

    return random.choice(TASK3), None


# define a  message data class to store the message containing a list of dict with role and content keys
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        assert self.role in ["Human", "Assistant"]


class Prompt:
    def __init__(
        self,
        specialization="auto",
        safty_level=1,
        format="simple",
    ):
        # get format from env first
        if "prompt_format" in os.environ:
            format = os.environ["prompt_format"]
            logger.info(f"using prompt format from env: {format}")

        # print("format:", format)
        if format == "llama2":
            self.BOS_SYS = "<<SYS>>"
            self.EOS_SYS = "<</SYS>>\n"
            self.BOS_INS = "[INST]"
            self.EOS_INS = "[/INST]"
            self.human_role_name = ""
            self.assistant_role_name = ""

        elif format == "llama-simple":
            self.BOS_SYS = "<sys>"
            self.EOS_SYS = "</sys>\n"
            self.BOS_INS = "<ins>"
            self.EOS_INS = "</ins>"
            self.end_of_turn = "</s>"
            self.human_role_name = "\nHuman: "
            self.assistant_role_name = "\nAssistant: "

        elif format == "simple":
            self.BOS_SYS = "<<SYS>>"
            self.EOS_SYS = "<</SYS>>"
            self.BOS_INS = ""
            self.EOS_INS = ""
            # self.human_role_name = "\n\nUser: "
            # self.assistant_role_name = "\nAssistant: "
            self.human_role_name = "\n### INSTRUCTION:\n\n"
            self.assistant_role_name = "\n### RESPONSE:\n\n"
            self.end_of_turn = "</s>"

        elif format == "qwen":
            self.BOS_SYS = "<|im_start|>"
            self.EOS_SYS = "<|im_end|>\n\n"
            self.BOS_INS = ""
            self.EOS_INS = ""
            self.human_role_name = "### Input:\n\n"
            self.assistant_role_name = "\n### Response:\n\n"
            self.end_of_turn = "<|endoftext|>"

        elif format == "llama-new":
            self.BOS_SYS = "<|im_start|>"
            self.EOS_SYS = "<|im_end|>\n\n"
            self.BOS_INS = ""
            self.EOS_INS = ""
            self.human_role_name = "### INPUT:\n\n"
            self.assistant_role_name = "### RESPONSE:\n\n"
            self.end_of_turn = "</s>"

        else:
            raise ValueError(f"format {format} is not supported")

        self.specialization = specialization

        self.safty_level = safty_level

    def remove_special_tokens(
        self, text: str, remove_sys_message=False, remove_role=False
    ):
        if remove_sys_message:
            # remove anything between self.BOS_SYS and self.EOS_SYS
            text = re.sub(rf"{self.BOS_SYS}.*{self.EOS_SYS}", "", text)

        if remove_role:
            text = text.replace("Human:", "")
            text = text.replace("Assistant:", "")

        """remove the special tokens from the text"""
        text = text.replace(self.BOS_INS, "")
        text = text.replace(self.EOS_INS, "")
        text = text.replace(self.BOS_SYS, "")
        text = text.replace(f"{self.EOS_SYS}", "")
        text = text.replace("</s>", "")
        text = text.strip()

        return text

    def format_prompt(
        self,
        instruction,
        output,
        source="",
        random_context_order=False,
        fix_text_ending=False,
        add_sys_message=True,
    ):
        """Format the input, output, instruction into a prompt.
        Args:x
            input (str): the input text
            output (str): the output text
            instruction (str): the instruction text
        Returns:
            str: the formatted prompt
        """

        if "RESPONSE:" in instruction:
            import pdb; pdb.set_trace()
        if fix_text_ending:
            output = fix_text_end_without_period(output)
            instruction = fix_text_end_without_period(instruction)

        sys_msg, safety = determine_task(instruction, output, source)

        if safety is not None:
            self.safty_level = safety
        else:
            if self.safty_level == 1:
                sys_msg += "\n" + SAFETY_MSG1
            elif self.safty_level == 2:
                sys_msg += "\n" + SAFETY_MSG1
                sys_msg += "\n" + SAFETY_MSG2

        prefix = f"{self.BOS_INS}{self.BOS_SYS}{sys_msg}{self.EOS_SYS}"  # no need to add <s>. leave it for tokenizer
        input_template = (
            prefix
            + f"{self.human_role_name}"
            + "{instruction}"
            + f"{self.EOS_INS}"
            + f"{self.assistant_role_name}"
        )
        output_template = "{output}" + self.end_of_turn
        input = input_template.format(instruction=instruction)
        if output != "":
            output = output_template.format(output=output)

        assert input.endswith("\n\n")
        assert not output.startswith("\n")
        text = input + output
        # if source == "identity":
        #     # import pdb; pdb.set_trace()
        #     print(text)

        return text, input, output


if __name__ == "__main__":
    # add some basic test
    prompting = Prompt()

    message = [
        Message(role="Human", content="Hello"),
        Message(role="Assistant", content="Hi, how can I help you?"),
        # Message(role="Human", content="I want to book a flight from Beijing to Shanghai"),
        # Message(role="Assistant", content="When do you want to depart?"),
    ]
    text, input, output = prompting.format_message(message)

    print("=========input=======\n", input)
    print("=========output=======\n", output)
    print("=========text=======\n", text)
