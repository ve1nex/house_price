import torch


def select_positive_negative(distances, labels, hard_negative=False):
    """Select one positive and one negative distance for every valid anchor.

    hard_negative=False: random positive/random negative from the current batch.
    hard_negative=True: hardest positive + closest (hardest) negative.
    """
    positives, negatives = [], []
    labels = labels.reshape(-1)
    batch = len(labels)

    for i in range(batch):
        same = torch.where(labels == labels[i])[0]
        same = same[same != i]
        different = torch.where(labels != labels[i])[0]
        if len(same) == 0 or len(different) == 0:
            continue

        if hard_negative:
            pos = distances[i, same].max()
            neg = distances[i, different].min()
        else:
            pos_index = same[torch.randint(len(same), (1,), device=labels.device)]
            neg_index = different[torch.randint(len(different), (1,), device=labels.device)]
            pos = distances[i, pos_index].squeeze()
            neg = distances[i, neg_index].squeeze()

        positives.append(pos)
        negatives.append(neg)

    if not positives:
        return None, None
    return torch.stack(positives), torch.stack(negatives)
