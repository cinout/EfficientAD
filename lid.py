import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score

path = "features_anomaly_detection/feature_maps_breakfast_box/"

train_features = torch.load(os.path.join(path, "train_split/normal_features.t"))

test_normal_features = torch.load(os.path.join(path, "test_split/normal_features.t"))
test_logical_anomaly = torch.load(
    os.path.join(path, "test_split/logic_anomaly_features.t")
)
test_structure_anomaly = torch.load(
    os.path.join(path, "test_split/structure_anomaly_features.t")
)


def lid_mle(data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    b = data.shape[0]
    k = min(k, b - 2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.0e-4), dim=1)
    return lids


lid_mle(test_normal_features, train_features)
lid_mle(test_logical_anomaly, train_features)
lid_mle(test_structure_anomaly, train_features)


# logical anomaly

y = torch.zeros(len(test_logical_anomaly) + len(test_normal_features))
y[: len(test_logical_anomaly)] = 1

lid_score = lid_mle(
    data=torch.cat(
        [test_logical_anomaly.mean(dim=[1]), test_normal_features.mean(dim=[1])], dim=0
    ),
    reference=train_features.mean(dim=[1]),
    k=16,
)

roc_auc_score(y.long().numpy(), lid_score.detach().cpu().numpy())


# structure anomaly

y = torch.zeros(len(test_structure_anomaly) + len(test_normal_features))
y[: len(test_structure_anomaly)] = 1

lid_score = lid_mle(
    data=torch.cat(
        [test_structure_anomaly.flatten(1), test_normal_features.flatten(1)], dim=0
    ),
    reference=train_features.flatten(1),
    k=16,
)

roc_auc_score(y.long().numpy(), lid_score.detach().cpu().numpy())
