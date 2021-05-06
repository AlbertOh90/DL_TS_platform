#Copyright (c) 2020 Hanwei Wu
import torch
def data_standlization(dataset):
    """standardize a tensor dataset

    Args:
    dataset -- tensors of shape (n, T)

    Returns:
    dataset -- tensors of shape (n, T), contains normalized values.
    """
    means = dataset.mean()
    stds = dataset.std()
    dataset = (dataset - means) / stds
    return dataset


def save_checkpoint(state, save_dir, epoch):
    save_path = save_dir + " {}.pt".format(epoch)
    with open(save_path, 'wb') as f:
        print("Saving model", save_path)
        torch.save(state, save_path)