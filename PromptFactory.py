# Simple Factory pattern that builds multiple types of prompts from a string and data object as this
# gets more complicated we can add more types of prompts and break it into a true Factory pattern

# Side note: Factories are often singletons as well

import pandas as pd

class PromptFactory():
    STOP_TOKEN = "[stop]"

    @classmethod
    def producePrompt(cls, questionData, promptType, examples=None):
        # Add to this ladder and create an internal method
        if promptType == "distractor_only":
            return cls._disOnlyPrompt(questionData, examples)
        elif promptType == "distractor_and_answer":
            return cls._disAnsPrompt(questionData, examples)
        elif promptType == "distractor_and_answer_with_feedback":
            return cls._disAnsFeedPrompt(questionData, examples)
        elif promptType == "zero_shot":
            return cls._zeroShotPrompt(questionData, examples)
        elif promptType == "rule_based_random":
            return cls.rule_based_random_prompt(questionData, examples)
        elif promptType == "rule_based_selection":
            return cls.rule_based_selection_prompt(questionData, examples)
        else:
            raise ValueError(promptType + " is not an available prompt type")

    @classmethod
    def _disOnlyPrompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        Question: XXX\n
        Distractor1: XXX\n
        Distractor2: XXX\n
        Distractor3: XXX\n
        [STOP]
        === PROMPT ===
        Question: XXX\n        
        """
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += f"Question: {row['question']}\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN
        prompt = examples_text + f"\nQuestion: {questionData['question']}\n"

        return prompt
    
    @classmethod
    def _disAnsPrompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        Question: XXX\n
        Answer: XXX\n
        Distractor1: XXX\n
        Distractor2: XXX\n
        Distractor3: XXX\n
        [STOP]
        === PROMPT ===
        Question: XXX\n
        Answer: XXX\n
        """
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += f"Question: {row['question']}\n" + f"Answer: {row['correct_option']['option']}\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN
        prompt = examples_text + f"\nQuestion: {questionData['question']}\nAnswer: {questionData['correct_option']['option']}\n"

        return prompt
    
    @classmethod
    def _disAnsFeedPrompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        Distractor1 Feedback: XXX\n
        Distractor1: XXX\n
        Distractor2 Feedback: XXX\n
        Distractor2: XXX\n
        Distractor3 Feedback: XXX\n
        Distractor3: XXX\n
        [STOP]
        === PROMPT ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        """
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1} Feedback: {x['explanation']}\nDistractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += f"Question: {row['question']}\n" + f"Explanation: {row['correct_option']['explanation']}\nAnswer: {row['correct_option']['option']}\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN
        prompt = examples_text + f"\nQuestion: {questionData['question']}\nExplanation: {questionData['correct_option']['explanation']}\nAnswer: {questionData['correct_option']['option']}\n"

        return prompt

    @classmethod
    def _zeroShotPrompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        """
        instructions="You are given the following math question along with the correct answer and explanation. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n \
        [Template]\n \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:"
        prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nExplanation: {questionData['correct_option']['explanation'].strip()}\nAnswer: {questionData['correct_option']['option'].strip()}"
        return prompt
    
    @classmethod
    def rule_based_random_prompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        Error1: XXX\n
        Error2: XXX\n
        Error3: XXX\n
        """
        instructions="You are given the following math question along with the correct answer, explanation, and three errors. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam based on the given three errors. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n \
        [Template]\n \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:"
        examples_text = ""
        for idx, example in enumerate(examples):
            examples_text += f"Error{idx+1}: {example}\n"
        prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nExplanation: {questionData['correct_option']['explanation'].strip()}\nAnswer: {questionData['correct_option']['option'].strip()}\n{examples_text}"
        prompt = prompt[:-1]
        return prompt
    
    @classmethod
    def rule_based_selection_prompt(cls, questionData, examples):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        Error list: ...
        """
        instructions="You are given the following math question along with the correct answer, explanation, and a list of errors. Please follow the template to first select three most likely errors for this question and use the selected errors to generate three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer. If the list of errors is not given, generate three errors instead and do not contain any explanation in the three incorrect answer\n \
        [Template]\n \
        Error1: \
        Error2: \
        Error3: \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:"
        examples_text = "Error list:\n"
        for idx, example in enumerate(examples):
            examples_text += f"{example}\n"
        prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nExplanation: {questionData['correct_option']['explanation'].strip()}\nAnswer: {questionData['correct_option']['option'].strip()}\n{examples_text}"
        prompt = prompt[:-1]
        return prompt