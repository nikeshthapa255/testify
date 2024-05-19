import yaml
import os
from datasets import Dataset, DatasetDict

def load_data(data_dir):
    components = []
    tests = []
    
    print(f"Loading data from {data_dir}...")

    for file in os.listdir(data_dir):
        if file.endswith("_components.yaml"):
            print(f"Loading components from {file}...")
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                components.extend(yaml.safe_load(f))
        elif file.endswith("_tests.yaml"):
            print(f"Loading tests from {file}...")
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                tests.extend(yaml.safe_load(f))
    
    print(f"Loaded {len(components)} components and {len(tests)} tests.")
    return components, tests

def create_training_examples(components, tests):
    print("Creating training examples...")
    examples = []
    for component in components:
        corresponding_tests = [test['content'] for test in tests if test['path'].startswith(component['path'].rsplit('/', 1)[0])]
        if corresponding_tests:
            examples.append({
                "input": component['content'],
                "output": "\n".join(corresponding_tests)
            })
    
    print(f"Created {len(examples)} training examples.")
    return examples

def save_training_data(examples, output_file):
    print(f"Saving training data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(f"Component:\n{example['input']}\nTests:\n{example['output']}\n\n")
    
    print("Training data saved.")

def main():
    data_dir = "../data-scrape/data"
    print("Starting data processing...")
    components, tests = load_data(data_dir)
    examples = create_training_examples(components, tests)
    save_training_data(examples, "training_data.txt")
    print("Data processing completed.")

if __name__ == "__main__":
    main()
