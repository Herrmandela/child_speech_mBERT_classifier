from torch.utils.data import TensorDataset, random_split


def trainingAndValidationVanilla():

    print()
    print("Training and Validation in training_and_validation.py")
    print(17)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, scores)

    # Creat a 90-10 train-validation split and calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
