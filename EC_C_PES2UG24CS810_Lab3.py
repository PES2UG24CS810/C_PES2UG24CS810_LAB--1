# student_lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor) -> float:
    """Calculate entropy of dataset (last column is target)."""
    target_col = tensor[:, -1]
    values, counts = torch.unique(target_col, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log2(probs)).sum().item()
    return round(float(entropy), 4)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int) -> float:
    """Calculate weighted entropy of splitting on a given attribute."""
    attr_col = tensor[:, attribute]
    values = torch.unique(attr_col)
    total = len(tensor)
    avg_info = 0.0

    for v in values:
        subset = tensor[attr_col == v]
        weight = len(subset) / total
        avg_info += weight * get_entropy_of_dataset(subset)

    return round(float(avg_info), 4)


def get_information_gain(tensor: torch.Tensor, attribute: int) -> float:
    """Calculate info gain of an attribute."""
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """Return dict of info gains and the best attribute index."""
    n_attributes = tensor.shape[1] - 1
    info_gains = {}

    for attr in range(n_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
