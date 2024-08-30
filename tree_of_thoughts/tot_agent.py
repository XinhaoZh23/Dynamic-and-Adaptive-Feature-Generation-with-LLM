import logging
from downstreamtask import train_classifier, test_classifier
from data_augmentation import apply_operation
from swarms import Agent
import re

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToTAgent:
    """

    OpenAI Language Model API Wrapper

    Args:
        agent (Agent): Agent class from swarms
        strategy (str): Strategy to use for generating thoughts
        evaluation_strategy (str): Strategy to use for evaluating states
        enable_react (bool): Enable ReAct prompting
        k (int): Number of thoughts to generate

    Methods:
        run(task: str) -> list: Generate text from prompt using OpenAI API
        generate_thoughts(state, k, initial_prompt, rejected_solutions=None) -> list: Generate thoughts from state using OpenAI API
        generate_solution(initial_prompt, state, rejected_solutions=None) -> str: Generate solution from state using OpenAI API
        evaluate_states(states, initial_prompt) -> dict: Evaluate states of reasoning using OpenAI API

    Examples:
        >>> from tree_of_thoughts.tot_agent import ToTAgent
        >>> from swarms import Agent
        >>> agent = Agent()
        >>> model = ToTAgent(agent)
        >>> thoughts = model.run("Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'.")
        >>> print(thoughts)
        ['Observation:xxxx', 'Thoughts:xxxx']

    """

    def __init__(
        self,
        agent: Agent,
        model: str = "DecisionTree",
        strategy: str = "cot",
        evaluation_strategy: str = "value",
        enable_react: bool = True,
        k: int = 3,
        *args,
        **kwargs,
    ):
        self.agent = agent
        self.use_chat_api = True
        self.enable_react = enable_react
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.k = k
        self.model_name = model

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ""
        if enable_react:
            self.react_prompt = (
                "Write down your operations in format 'Operation:xxxx',"
                " then write down the attributes you are operating on in format 'Attribute:xxxx',"
                " then write down your reasoning in format 'Reasoning:xxxx'."
            )

    def run(self, task: str):
        """Generate text from prompt using"""
        if self.use_chat_api:
            thoughts = []
            for _ in range(self.k):
                response = self.agent.run(task)
                content = response.content
                thoughts += [content]
            return thoughts


    def generate_thoughts(
                self, state, k, initial_prompt, operations,operation_examples, current_feature_names,rejected_solutions=None
        ):
            """
            Generate thoughts from state using OpenAI API

            Args:
                state (str or list): State of reasoning
                k (int): Number of thoughts to generate
                initial_prompt (str): Initial prompt
                rejected_solutions (list): List of rejected solutions

            Returns:
                list: List of thoughts
            """
            if type(state) == str:
                state_text = state
            else:
                state_text = "\n".join(state)

            # print("New state generating thought:", state, "\n\n")

            # Adjust feature names display based on the number of items
            if len(current_feature_names) > 40:
                features_display = ', '.join(current_feature_names[:3]) + ', ..., ' + ', '.join(
                    current_feature_names[-6:])
            else:
                features_display = ', '.join(current_feature_names)

            if state == initial_prompt:
                prompt = f"""
You are a superintelligent AI model devoted to helping humans. Your purpose is to generate a series of solutions to comply with the user's instructions.
Considering the task provided:

###'{initial_prompt}'###

Input:
features name: {features_display}
operation: {operations}

Example that may help you:
{operation_examples}

Devise the best possible solution for the task:
Here are evaluated solutions that were rejected:
###{rejected_solutions}###,
complete the task without making the same mistakes as before. Be simple, direct, and intuitive.
Possible next three steps:
"""
            else:
                prompt = f"""
You are a superintelligent AI model devoted to helping humans. Your purpose is to generate a series of solutions to comply with the user's instructions.
Considering the task provided:

###'{initial_prompt}'###

Input:
features name: {features_display}
operation: {operations}

Example that may help you:
{operation_examples}

Accepted solutions so far:

###'{state_text}'###

Devise the best possible solution for the task:
Here are evaluated solutions that were rejected:
###{rejected_solutions}###,
complete the task without making the same mistakes as before. Be simple, direct, and intuitive.
The possible next three step:
"""

            prompt += self.react_prompt
            # Generate k thoughts

            thoughts = self.run(prompt)

            return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:
            ###'{state_text}'###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = self.run(prompt)
            print(f"Answer {answer}")
            # print(thoughts)
            # print(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                print(
                    "We receive a state of type",
                    type(state),
                    "For state: ",
                    state,
                    "\n\n",
                )
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                response = self.agent.run(prompt)
                try:
                    value_text = response.content
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == "vote":
            states_text = "\n".join([" ".join(state) for state in states])
            prompt = (
                "Given the following states of reasoning, vote for the best"
                " state utilizing an scalar value"
                f" 1-10:\n{states_text}\n\nVote, on the probability of this"
                f" state of reasoning achieveing {initial_prompt} and become"
                " very pessimistic very NOTHING ELSE"
            )
            response = self.agent(prompt)
            print(f"state response: {response}")
            best_state_text = self.agent(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            print(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError(
                "Invalid evaluation strategy. Choose 'value' or 'vote'."
            )

    def format_states(self, states, feature_names):
        if not states:
            return {}

        formatted_responses = []  # Initialize an empty list to store responses

        for state in states:
            if type(state) == str:
                state_text = state
            else:
                state_text = "\n".join(state)
            # print(
            #     "We receive a state of type",
            #     type(state),
            #     "For state: ",
            #     state,
            #     "\n\n",
            # )

            # Adjust feature names display based on the number of items
            if len(feature_names) > 40:
                features_display = ', '.join(feature_names[:3]) + ', ..., ' + ', '.join(
                    feature_names[-6:])
            else:
                features_display = ', '.join(feature_names)


            prompt = f"""
Please reformat the latest solution using the structure provided below. Evaluate the context from past solutions to align the new format.

Current solution needs reformatting:
{state_text}

Instructions:
1. Start each step with 'stepX:'.
2. Use '<>' for attributes and operations, no spaces between.
3. End steps with '|'.

Example format:
'step1:<Attribute><Operation><Attribute>|step2:<Operation><Attribute>|step3:<Attribute><Operation><Attribute>|'

Only adhere to this output format.DO NOT RETURN ANYTHING ELSE.
"""


            formatted_response_raw = self.agent.run(prompt)
            formatted_response = formatted_response_raw.content
            formatted_responses.append(formatted_response)  # Append the response to the list

        return formatted_responses

    def parse_step(self, step, operations, feature_names):
        """
        Parse a single step to extract the attributes and operation.

        Args:
            step (str): The step to parse.
            operations (list): A list of possible operations.
            feature_names (list): A list of possible feature names.

        Returns:
            tuple: (attr1, operation, attr2) for binary operations, or (operation, attr1) for unary operations.
        """
        found_operation = None
        found_features = []

        # Search for the operation in the step
        for operation in operations:
            if operation in step:
                found_operation = operation
                break

        # Search for the attributes in the step
        for feature in feature_names:
            if feature in step:
                found_features.append(feature)

        if found_operation and len(found_features) == 2:  # Binary operation
            return found_features[0], found_operation, found_features[1]
        elif found_operation and len(found_features) == 1:  # Unary operation
            return found_operation, found_features[0]
        else:
            raise ValueError(f"Invalid step format: {step}")



    def evaluate_states_feature_generation(self,
                                           formatted_states,
                                           trainingset,
                                           validationset,
                                           testset,
                                           initial_metrics,
                                           operations,
                                           feature_names,
                                           model_params):
        if not formatted_states:
            return {}

        evaluated_states = {}
        for state in formatted_states:
            modified_trainingset = trainingset.copy()
            modified_validationset = validationset.copy()
            modified_testset = testset.copy()
            illegal_operations = []

            initial_accuracy, initial_precision, initial_recall, initial_f1 = initial_metrics


            matches = re.findall(r"step\d*:(.*?)(?:\||$)", state)

            if not matches:
                illegal_operations.append("Invalid format or no steps found")
                clean_state = ""
            else:
                clean_state = '|'.join(matches)

            steps = clean_state.split('|')
            for step in steps:


                if not step:
                    continue
                try:
                    parsed_result = self.parse_step(step, operations, feature_names)

                    # parsed_result = ('log', 'Height')

                    if len(parsed_result) == 3:  # Binary operation
                        attr1, operation, attr2 = parsed_result
                        modified_trainingset, error_message, new_column_name = apply_operation(modified_trainingset,
                                                                                               operation, attr1, attr2)
                        modified_validationset, error_message, new_column_name = apply_operation(modified_validationset,
                                                                                                 operation, attr1,
                                                                                                 attr2)
                        modified_testset, error_message, new_column_name = apply_operation(modified_testset, operation,
                                                                                           attr1, attr2)

                    elif len(parsed_result) == 2:  # Unary operation
                        operation, attr1 = parsed_result
                        modified_trainingset, error_message, new_column_name = apply_operation(modified_trainingset,
                                                                                               operation, attr1)
                        modified_validationset, error_message, new_column_name = apply_operation(modified_validationset,
                                                                                                 operation, attr1)
                        modified_testset, error_message, new_column_name = apply_operation(modified_testset, operation,
                                                                                           attr1)


                    new_feature_names = modified_trainingset.columns.tolist()


                    if len(modified_trainingset.columns) != len(modified_validationset.columns) or len(
                            modified_trainingset.columns) != len(modified_testset.columns):

                        if new_column_name in modified_trainingset.columns:
                            modified_trainingset.drop(columns=[new_column_name], inplace=True)
                        if new_column_name in modified_validationset.columns:
                            modified_validationset.drop(columns=[new_column_name], inplace=True)
                        if new_column_name in modified_testset.columns:
                            modified_testset.drop(columns=[new_column_name], inplace=True)
                        print(
                            "Column count mismatch after adding new feature. The inconsistent column has been removed.")

                    if error_message:
                        illegal_operations.append(error_message)
                        print(error_message)
                except ValueError as e:
                    print(e)
                    continue

            # Train and test the model with the modified dataset
            classifier = train_classifier(modified_trainingset.iloc[:, :-1], modified_trainingset.iloc[:, -1],
                                          self.model_name, model_params)
            y_pred, (new_accuracy, new_precision, new_recall, new_f1) = test_classifier(classifier, modified_validationset.iloc[:, :-1],
                                              modified_validationset.iloc[:, -1])
            y_pred_test, (new_accuracy_test, new_precision_test, new_recall_test, new_f1_test) = test_classifier(classifier, modified_testset.iloc[:, :-1],
                                                modified_testset.iloc[:, -1])

            # Calculate improvements
            accuracy_improvement = new_accuracy - initial_accuracy
            precision_improvement = new_precision - initial_precision
            recall_improvement = new_recall - initial_recall
            f1_improvement = new_f1 - initial_f1
            metrics_new = (new_accuracy, new_precision, new_recall, new_f1)
            metrics_new_test = (new_accuracy_test, new_precision_test, new_recall_test, new_f1_test)
            # metrics_improvement = (accuracy_improvement, precision_improvement, recall_improvement, f1_improvement)

            state_summary = f"{clean_state}\nIllegal operations: {', '.join(illegal_operations)}" if illegal_operations else clean_state
            evaluated_states[state_summary] = (modified_trainingset, modified_validationset, modified_testset, new_feature_names, accuracy_improvement, metrics_new, metrics_new_test)

        return evaluated_states




