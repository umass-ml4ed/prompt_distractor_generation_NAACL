# Simple Factory pattern that builds multiple types of prompts from a string and data object as this
# gets more complicated we can add more types of prompts and break it into a true Factory pattern

# Side note: Factories are often singletons as well

def get_question_topic(item):
    # Yaya
    topic = "Math"
    if (item["construct_info"]["construct1"][1]):
        topic = item["construct_info"]["construct1"][1]
    if (item["construct_info"]["construct2"][1]):
        topic = item["construct_info"]["construct2"][1]

    return topic

def get_question_full_info(q_data):
    return f"Question: {q_data['question']}\n" +\
        f"Topic: {get_question_topic(q_data)}\n" +\
        f"Concept: {q_data['construct_info']['construct3'][1]}\n" +\
        f"Explanation: {q_data['correct_option']['explanation']}\n" +\
        f"Answer: {q_data['correct_option']['option']}"

class PromptFactory():
    STOP_TOKEN = "[stop]"

    @classmethod
    def producePrompt(cls, questionData, promptType, num_distractors, examples=None):
        # Add to this ladder and create an internal method
        if promptType == "distractor_only":
            return cls._disOnlyPrompt(questionData, examples)
        elif promptType == "distractor_and_answer":
            return cls._disAnsPrompt(questionData, examples)
        elif promptType == "distractor_and_answer_with_feedback":
            return cls._disAnsFeedPrompt(questionData, examples)
        elif promptType == "distractor_all_info":
            return cls._disAllInfoPrompt(questionData, examples, num_distractors)
        elif promptType == "distractor_all_info_with_feedback":
            return cls._disAllInfoFeedPrompt(questionData, examples, num_distractors)
        elif promptType == "distractor_all_info_with_errors":
            return cls._disAllInfoErrorPrompt(questionData, examples, num_distractors)
        elif promptType == "zero_shot":
            return cls._zeroShotPrompt(questionData, examples)
        elif promptType == "zero_shot_all_info_error":
            return cls._zeroShotAllInfoErrorPrompt(questionData, num_distractors)
        elif promptType == "rule_based_random":
            return cls.rule_based_random_prompt(questionData, examples)
        elif promptType == "rule_based_selection":
            return cls.rule_based_selection_prompt(questionData, examples, num_distractors)
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
    def _disAllInfoPrompt(cls, questionData, examples, num_distractors):
        instructions = "You will be given a math question along with the correct answer and explanation. " +\
            "You will be also provided with several example questions that include incorrect distractor answers. " +\
            f"Please generate {num_distractors} incorrect distractor answers for the current question to be used as " +\
            "multiple-choice options in a multiple-choice exam." +\
            "\n[Template]\n" +\
            "Distractor1:\n" +\
            "...\n" +\
            f"Distractor{num_distractors}:\n"
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += get_question_full_info(row) + "\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN + "\n"
        prompt = instructions + "\n" + examples_text + get_question_full_info(questionData) + "\n"

        return prompt

    @classmethod
    def _disAllInfoFeedPrompt(cls, questionData, examples, num_distractors):
        instructions = "You will be given a math question along with the correct answer and explanation. " +\
            "You will be also provided with several example questions that include incorrect distractor answers. " +\
            f"Please generate {num_distractors} incorrect distractor answers for the current question to be used as " +\
            "multiple-choice options in a multiple-choice exam." +\
            "\n[Template]\n" +\
            "Distractor1 Feedback:\n" +\
            "Distractor1:\n" +\
            "...\n" +\
            f"Distractor{num_distractors} Feedback:\n" +\
            f"Distractor{num_distractors}:\n"
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1} Feedback: {x['explanation']}\nDistractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += get_question_full_info(row) + "\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN + "\n"
        prompt = instructions + "\n" + examples_text + get_question_full_info(questionData) + "\n"

        return prompt

    @classmethod
    def _disAllInfoErrorPrompt(cls, questionData, examples, num_distractors):
        instructions = "You will be given a math question along with the correct answer and explanation. " +\
            "You will be also provided with several example questions that include incorrect distractor answers. " +\
            f"Please generate {num_distractors} incorrect distractor answers for the current question to be used as " +\
            "multiple-choice options in a multiple-choice exam." +\
            "\n[Template]\n" +\
            "Distractor1 Error:\n" +\
            "Distractor1:\n" +\
            "...\n" +\
            f"Distractor{num_distractors} Error:\n" +\
            f"Distractor{num_distractors}:\n"
        examples_text = ""
        for _, row in examples.iterrows():
            distractors_text_list = [f"Distractor{i+1} Error: {x['misconception']}\nDistractor{i+1}: {x['option']}\n" for i, x in enumerate(row["distractors"])]
            distractor_text = ''.join(distractors_text_list)
            examples_text += get_question_full_info(row) + "\n" + distractor_text
            examples_text += PromptFactory.STOP_TOKEN + "\n"
        prompt = instructions + "\n" + examples_text + get_question_full_info(questionData) + "\n"

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
    def _zeroShotAllInfoErrorPrompt(cls, questionData, num_distractors):
        instructions = "You are given the following math question along with the correct answer and explanation. " +\
            f"Please use the following template to give {num_distractors} alternative incorrect answers to be used as " +\
            "multiple-choice options in a multiple-choice exam. " +\
            "Prior to the incorrect answer, provide the underlying error corresponding to that incorrect answer. " +\
            "These errors should be conceptual in nature and should not refer to numbers, variables, or names in the question." +\
            "\n[Template]\n" +\
            "Distractor1 Error:\n" +\
            "Distractor1:\n" +\
            "...\n" +\
            f"Distractor{num_distractors} Error:\n" +\
            f"Distractor{num_distractors}:\n"
        prompt = instructions + "\n" + get_question_full_info(questionData)
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
    def rule_based_selection_prompt(cls, questionData, examples, num_distractors):
        """
        === EXAMPLE ===
        <Instructions>
        === PROMPT ===
        Question: XXX\n
        Explanation: XXX\n
        Answer: XXX\n
        Error list: ...
        """
        instructions = f"You are given the following math question along with the correct answer, explanation, and a list of errors. Please follow the template to first select {num_distractors} most likely errors for this question and use the selected errors to generate {num_distractors} alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer. If the list of errors is not given, generate {num_distractors} errors instead and do not contain any explanation in the {num_distractors} incorrect answer.\n" +\
        "[Template]\n" +\
        "Error 1:\n" +\
        "...\n" +\
        f"Error {num_distractors}\n" +\
        "Distractor1 Feedback:\n" +\
        "Distractor1:\n" +\
        "...\n" +\
        f"Distractor{num_distractors} Feedback:\n" +\
        f"Distractor{num_distractors}:\n"
        examples_text = "Error list:\n"
        for idx, example in enumerate(examples):
            examples_text += f"{example}\n"
        prompt = f"{instructions}\nQuestion: {questionData['question'].strip()}\nExplanation: {questionData['correct_option']['explanation'].strip()}\nAnswer: {questionData['correct_option']['option'].strip()}\n{examples_text}"
        prompt = prompt[:-1]
        return prompt
