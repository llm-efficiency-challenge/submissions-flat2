import numpy as np; np.random.seed(123)
import pandas as pd

def data_prep():
    df = pd.read_json("https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json")

    # We're going to create a new column called `split` where:
    # 90% will be assigned a value of 0 -> train set
    # 5% will be assigned a value of 1 -> validation set
    # 5% will be assigned a value of 2 -> test set
    # Calculate the number of rows for each split value
    total_rows = len(df)
    split_0_count = int(total_rows * 0.9)
    split_1_count = int(total_rows * 0.05)
    split_2_count = total_rows - split_0_count - split_1_count

    # Create an array with split values based on the counts
    split_values = np.concatenate([
        np.zeros(split_0_count),
        np.ones(split_1_count),
        np.full(split_2_count, 2)
    ])

    # Shuffle the array to ensure randomness
    np.random.shuffle(split_values)

    # Add the 'split' column to the DataFrame
    df['split'] = split_values
    df['split'] = df['split'].astype(int)

    # For this webinar, we will just 500 rows of this dataset.
    df = df.head(n=500)

    return df

def data_summary(df):
    num_self_sufficient = (df['input'] == '').sum()
    num_need_contex = df.shape[0] - num_self_sufficient
    # We are only using 100 rows of this dataset for this webinar
    print(f"Total number of examples in the dataset: {df.shape[0]}")
    print(f"% of examples that are self-sufficient: {round(num_self_sufficient/df.shape[0] * 100, 2)}")
    print(f"% of examples that are need additional context: {round(num_need_contex/df.shape[0] * 100, 2)}")
    return

def data_distribution(df):
    # Calculating the length of each cell in each column
    df['num_characters_instruction'] = df['instruction'].apply(lambda x: len(x))
    df['num_characters_input'] = df['input'].apply(lambda x: len(x))
    df['num_characters_output'] = df['output'].apply(lambda x: len(x))

    # Show Distribution
    df.hist(column=['num_characters_instruction', 'num_characters_input', 'num_characters_output'])

    # Calculating the average
    average_chars_instruction = df['num_characters_instruction'].mean()
    average_chars_input = df['num_characters_input'].mean()
    average_chars_output = df['num_characters_output'].mean()

    print(f'Average number of tokens in the instruction column: {(average_chars_instruction / 3):.0f}')
    print(f'Average number of tokens in the input column: {(average_chars_input / 3):.0f}')
    print(f'Average number of tokens in the output column: {(average_chars_output / 3):.0f}', end="\n\n")