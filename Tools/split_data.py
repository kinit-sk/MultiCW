# import ollama
# model="llama3.2:3b"
# client=ollama.Client("http://localhost:11434")

# conversation = [
#         {"role": "system", "content": "You are a helpfull assistent"},
#         {"role": "user", "content": "Hi my name is jakub. Do you know my name?"}]
# print("Response1: ", client.chat(model=model, messages=conversation)['message']['content'])

# conversation = [{"role": "user", "content": "Do you know my name?"}]
# print("Response2: ", client.chat(model=model, messages=conversation)['message']['content'])



import os
import pandas as pd
from py_markdown_table.markdown_table import markdown_table


def split_part(df, test_ratio: float, dev_ratio: float):
    # Split based on writing style
    noisy = df[df['style'] == 'noisy']
    struc = df[df['style'] == 'struct']

    # Noisy part
    noisy_size = [int(noisy.shape[0] * test_ratio), int(noisy.shape[0] * dev_ratio)]
    struc_size = [int(struc.shape[0] * test_ratio), int(struc.shape[0] * dev_ratio)]


    # Split each part into train, dev and test
    noisy_train, noisy_dev, noisy_test  = split_origin(noisy, noisy_size[0], noisy_size[1])
    struc_train, struc_dev, struc_test = split_origin(struc, struc_size[0], struc_size[1])

    # Combine sets
    test = [noisy_test, struc_test]
    dev = [noisy_dev, struc_dev]
    train = [noisy_train, struc_train]

    return train, dev, test


def split_origin(df, test_size: int, dev_size: int):
    # Split based on origin. Prioritize manual samples but use all of them.
    manual = df[df['origin'] == 'manual']
    rest = df[df['origin'] != 'manual']
    sorted = pd.concat([manual, rest]).sample(frac=1)

    # Split into classes
    class_0, class_1 = split_class(sorted)

    # Split each class into train, dev and test set
    idx = [int(test_size / 2), int(dev_size / 2)]
    test_0 = class_0.iloc[:idx[0]].copy()
    dev_0 = class_0.iloc[idx[0]: idx[0] + idx[1]].copy()
    train_0 = class_0.iloc[idx[0] + idx[1]:, :].copy()

    test_1 = class_1.iloc[:idx[0]].copy()
    dev_1 = class_1.iloc[idx[0]: idx[0] + idx[1]].copy()
    train_1 = class_1.iloc[idx[0] + idx[1]:, :].copy()

    train = pd.concat([train_0, train_1])
    dev = pd.concat([dev_0, dev_1])
    test = pd.concat([test_0, test_1])

    return train, dev, test

def split_class(df):
    # Split based on label
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]

    return class_0, class_1

# Function to split the dataframe
def split_dataframe(df, test_ratio=0.015, dev_ratio=0.15):
    train_set = []
    test_set = []
    dev_set = []
    stats = []

    for lang in df['lang'].unique():
        # Extract the specific language from the dataframe
        lang_df = df[df['lang'] == lang]

        # Split to train, dev and test
        train, dev, test = split_part(lang_df, test_ratio, dev_ratio)

        # Save statistics
        stats.append({'Language': lang,
                'Noisy Train:': train[0].shape[0],
                'Noisy Dev:': dev[0].shape[0],
                'Noisy Test:': test[0].shape[0],
                'Structured Train:': train[1].shape[0],
                'Structured Dev:': dev[1].shape[0],
                'Structured Test:': test[1].shape[0]})

        # Combine sets
        train_set.append(pd.concat(train))
        dev_set.append(pd.concat(dev))
        test_set.append(pd.concat(test))

    train_set = pd.concat(train_set)
    test_set = pd.concat(test_set)
    dev_set = pd.concat(dev_set)

    total = [{'Train set:': train_set.shape[0],
              'Test set:': test_set.shape[0],
              'Dev set:': dev_set.shape[0],
              'Total:': multicw.shape[0]}]

    table1 = markdown_table(stats).set_params(row_sep = 'markdown').get_markdown()
    print(table1.replace('```',''))

    table2 = markdown_table(total).set_params(row_sep = 'markdown').get_markdown()
    print(table2.replace('```',''))

    return train_set, test_set, dev_set


multicw = pd.read_csv("/datadrive/Final-dataset/multicw-no-en-translations.csv")
# multicw = multicw.drop('index', axis=1)
train_df, test_df, dev_df = split_dataframe(multicw)

out="large_scale_dataframes"
train_df.to_csv(os.path.join(out, 'multicw-train-small-test.csv'), index=False)
test_df.to_csv(os.path.join(out, 'multicw-test-small-test.csv'), index=False)
dev_df.to_csv(os.path.join(out, 'multicw-dev-small-test.csv'), index=False)


# print(f'Training, validation and test sets has been saved to {multicw_path}')