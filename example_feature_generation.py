import os
from tree_of_thoughts import ToTAgent, MonteCarloSearch
from dotenv import load_dotenv
from swarms import Agent, OpenAIChat
from data_processing import load_dataset, split_dataset, load_feature_names,list_to_string, load_dataset_description, load_operations, read_dataset_description, generate_operation_examples,encode_categorical_variables
from downstreamtask import train_classifier, test_classifier
import argparse

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Run the LFG with a specified dataset")
parser.add_argument("--dataset", type=str, required=False,default='abalone', choices=['ionosphere', 'amazon', 'abalone',  'Diabetes'], help="Dataset to use in the application")
# DecisionTree
parser.add_argument("--model", type=str, required=False,default='decision_tree', choices=['random_forest', 'decision_tree', 'knn', 'mlp'], help="Name of the model to use for downstream tasks")
args = parser.parse_args()

# LLM参数
num_thoughts = 3
max_steps = 8
max_states = 1
pruning_threshold = 0.003
train_size = 0.55
valid_size = 0.2
test_size = 1- train_size - valid_size


# Basline 参数
if args.model == 'decision_tree':
    model_params = {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None
    }

elif args.model == 'random_forest':
    model_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }

elif args.model == 'knn':
    model_params = {
        'n_neighbors': 5
    }

elif args.model == 'mlp':
    # hidden_layer_sizes:表示隐藏层结构的参数
    # (100,) 表示有一个隐藏层，其中包含 100 个神经元。
    # (50, 50) 表示有两个隐藏层，每层各有 50 个神经元。
    model_params = {
        'hidden_layer_sizes': (100,),
        'batch_size': 100,  # 可能需要调整为一个具体的数值如200，因为‘auto’已不被推荐使用
        'max_iter': 200
    }


# Print the dataset name
print("Dataset Name:", args.dataset)


# load dataset and slpit into train、set and test
dataset_raw = load_dataset(args.dataset)


# data prepocessing
dataset = encode_categorical_variables(dataset_raw)

if dataset is not None:
    # You can work with 'dataset' DataFrame here
    print(dataset.head())


train_set, validation_set,test_set = split_dataset(dataset, train_size, valid_size,test_size)
# read features names
feature_names = load_feature_names(args.dataset)
feature_names_string = list_to_string(feature_names, delimiter=', ')
description = load_dataset_description(args.dataset)
operations_unary, operations_binary = load_operations()
operations_unary_string = list_to_string(operations_unary, delimiter=', ')
operations_binary_string = list_to_string(operations_binary, delimiter=', ')
# 将binary和unary的operations合并
operations = operations_unary + operations_binary

# 生成prompt示例
operation_examples = generate_operation_examples(feature_names, operations_unary, operations_binary)

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize an agent from swarms
agent = Agent(
    agent_name="tree_of_thoughts",
    agent_description=(
        "This agent uses the tree_of_thoughts library to generate thoughts."
    ),
    system_prompt=None,
    llm=OpenAIChat( model_name="gpt-4o",
                    openai_api_key=api_key,
                    temperature=1,
                    max_tokens=1000),
    max_loops=1,
)

# Initialize the ToTAgent class with the API key
model = ToTAgent(
    agent,
    strategy="cot",
    evaluation_strategy="value",
    enable_react=True,
    k=num_thoughts,
    model=args.model
)


# Initialize the MonteCarloSearch class with the model
tree_of_thoughts = MonteCarloSearch(model)

# Define the initial prompt

dataset_description = read_dataset_description(args.dataset)

initial_prompt = f"""
Dataset description:
{description}

Task: produce 3 possible next steps to generate features using the operations, trying to improve the {args.model} model's performance. Explain the reasoning behind each step.
"""



# Set the initial performance of the model
X_train = train_set.iloc[:, :-1]
y_train = train_set.iloc[:, -1]
X_validation = validation_set.iloc[:, :-1]
y_validation = validation_set.iloc[:, -1]
X_test = test_set.iloc[:, :-1]
y_test = test_set.iloc[:, -1]

# Train the classifier
classifier = train_classifier(X_train, y_train, args.model, model_params)
_, initial_metrics_test = test_classifier(classifier, X_test, y_test)
_, initial_metrics_val = test_classifier(classifier, X_validation, y_validation)

# Unpack the metrics for clearer access
initial_accuracy, initial_precision, initial_recall, initial_f1 = initial_metrics_val

# Print the results
print("Initial Validation Results:")
print(f"Accuracy: {initial_accuracy:.4f}")
print(f"Precision: {initial_precision:.4f}")
print(f"Recall: {initial_recall:.4f}")
print(f"F1: {initial_f1:.4f}")

print("Initial Test Results:")
print(f"Accuracy: {initial_metrics_test[0]:.4f}")
print(f"Precision: {initial_metrics_test[1]:.4f}")
print(f"Recall: {initial_metrics_test[2]:.4f}")
print(f"F1: {initial_metrics_test[3]:.4f}")

# Generate the thoughts
solution = tree_of_thoughts.solve_feature_generation(
    initial_prompt=initial_prompt,
    num_thoughts=num_thoughts,
    max_steps=max_steps,
    max_states=max_states,
    pruning_threshold=pruning_threshold,
    trainingset=train_set,
    validationset=validation_set,
    testset=test_set,
    initial_metrics=initial_metrics_val,
    operations = operations,
    operation_examples = operation_examples,
    feature_names = feature_names,
    model_params = model_params
    # sleep_time=sleep_time
)

