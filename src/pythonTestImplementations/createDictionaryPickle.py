# TESTING: Reading CV traintest.pkl file
import pickle

filePath = 'data/traintest.pkl'

with open(filePath, 'rb') as file:
    data = pickle.load(file)

print(data)




import pickle
import os
import numpy as np 



# This Function Will Create a .pkl file that has data about each image such as name and label 

def create_traintest_pickle(data_dir, output_pickle='traintest.pkl'):
    # Categories
    mapping = [
        'airport_terminal',
        'campus',
        'desert',
        'elevator',
        'forest',
        'kitchen',
        'lake',
        'swimming_pool'
    ]

    train_dir = 'data/Training'
    test_dir = 'data/Testing'

    train_imagenames = []
    test_imagenames = []
    train_labels = []
    test_labels = []

    for i, category in enumerate(mapping):
        label = i + 1

        # Training images (e.g., data/Training/desert/*.jpg)
        category_train_path = os.path.join(train_dir, category)
        train_files = sorted(os.listdir(category_train_path))
        for f in train_files:
            train_imagenames.append(os.path.join(category, f))
            train_labels.append(label)

        category_test_path = os.path.join(test_dir, 'test_' + category)
        test_files = sorted(os.listdir(category_test_path))
        for f in test_files:
            test_imagenames.append(os.path.join('test_' + category, f))
            test_labels.append(label)

    # Convert to numpy arrays
    train_labels = np.array(train_labels, dtype=float)
    test_labels = np.array(test_labels, dtype=float)

    # Combine all
    all_imagenames = train_imagenames + test_imagenames
    all_labels = np.concatenate((train_labels, test_labels))

    # Create the dictionary
    data_dict = {
        'all_labels': all_labels,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'all_imagenames': all_imagenames,
        'train_imagenames': train_imagenames,
        'test_imagenames': test_imagenames,
        'mapping': mapping
    }

    # Save to pickle
    with open(output_pickle, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Pickle file saved to {output_pickle}")


# UNCOMMENT TO CREATE PICKLE 
# create_traintest_pickle('data', 'data/traintest.pkl')