import pandas as pd

def test_1(model):
    test_examples = pd.DataFrame([
        {
            "instruction": "Create an array of length 5 which contains all even numbers between 1 and 10.",
            "input": ''
        },
        {
            "instruction": "Create an array of length 15 containing numbers divisible by 3 up to 45.",
            "input": "",
        },
        {
            "instruction": "Create a nested loop to print every combination of numbers between 0-9",
            "input": ""
        },
        {
            "instruction": "Generate a function that computes the sum of the numbers in a given list",
            "input": "",
        },
        {
            "instruction": "Create a class to store student names, ages and grades.",
            "input": "",
        },
        {
            "instruction": "Print out the values in the following dictionary.",
            "input": "my_dict = {\n  'name': 'John Doe',\n  'age': 32,\n  'city': 'New York'\n}",
        },
    ])

    predictions = model.predict(test_examples)[0]
    for input_with_prediction in zip(test_examples['instruction'], test_examples['input'], predictions['output_response']):
        print(f"Instruction: {input_with_prediction[0]}")
        print(f"Input: {input_with_prediction[1]}")
        print(f"Generated Output: {input_with_prediction[2][0]}")
        print("\n\n")