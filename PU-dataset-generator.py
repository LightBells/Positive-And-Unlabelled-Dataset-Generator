# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import math
import os
import sys

# + executionInfo={"elapsed": 5915, "status": "ok", "timestamp": 1684893889867, "user": {"displayName": "Lightbells -", "userId": "12609649367297122986"}, "user_tz": -540} id="xuY1I39tCXpk"
import pickle
import sys

import numpy as np
import torch
from tqdm.notebook import tqdm

from common import DatasetType, RootConfig


# -


class Config(RootConfig):
    DATASET = DatasetType.CIFAR10
    CLASS_NUM = 10


# + executionInfo={"elapsed": 59977, "status": "ok", "timestamp": 1684894048994, "user": {"displayName": "Lightbells -", "userId": "12609649367297122986"}, "user_tz": -540} id="zZZJA0krCdzM"
RAW_DATA_ROOT = os.path.join(Config.DATA_ROOT, f"raw_{Config.DATASET.name}")
with open(
    os.path.join(RAW_DATA_ROOT, f"{Config.DATASET.name}train.pickle"), "rb"
) as file:
    train = pickle.load(file)
with open(
    os.path.join(RAW_DATA_ROOT, f"{Config.DATASET.name}test.pickle"), "rb"
) as file:
    test = pickle.load(file)


# + executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1684894180050, "user": {"displayName": "Lightbells -", "userId": "12609649367297122986"}, "user_tz": -540} id="XRcm1616sNwA"
def probablisticMissingGenerator(data, p, generator=lambda p: np.random.binomial(1, p)):
    new_data = []
    for indice in data:
        manipulated_indice = [idx for idx in indice if generator(p) == 0]
        new_data.append(
            torch.Tensor(
                [idx in manipulated_indice for idx in range(Config.CLASS_NUM)])
        )
    return new_data


# -


def intervalSampler(f, t, sampler=None, points=5):
    l = min(f, t)
    u = max(f, t)
    if sampler is None:
        print(
            'Warning: A sampler is explictly set. The default sampler is "uniform".',
            file=sys.stderr,
        )
        return [l + (u - l) / (points - 1) * i for i in range(points)]
    if sampler == "uniform":
        return [l + (u - l) / (points - 1) * i for i in range(points)]
    elif sampler == "log_uniform":
        return [
            math.e ** (math.log(l) + (math.log(u) -
                       math.log(l)) / (points - 1) * i)
            for i in range(points)
        ]
    else:
        raise NotImplementedError(
            "The sampler {} is not implemented.".format(sampler))


len(train[0])

# + executionInfo={"elapsed": 330677, "status": "ok", "timestamp": 1684894510721, "user": {"displayName": "Lightbells -", "userId": "12609649367297122986"}, "user_tz": -540} id="3jsiP4p4u44u"
sampled_value = set(
    intervalSampler(0.01, 0.1, "log_uniform") +
    intervalSampler(0.1, 1.0, "uniform", 10)[1:-1]
    # + intervalSampler(0.9, 1.0, "uniform")[1:-1]
    # + intervalSampler(0.975, 1.0, "uniform")[1:-1]
)

GENERATED_DATA_ROOT = os.path.join(
    Config.DATA_ROOT, Config.GENERATED_DATA_RELATIVE_PATH.format(
        Config.DATASET.name
    )
)

if not os.path.exists(GENERATED_DATA_ROOT):
    print(f"{GENERATED_DATA_ROOT} have not existed. It will be created.")
    os.makedirs(GENERATED_DATA_ROOT)

FILENAME = os.path.join(GENERATED_DATA_ROOT,
                        f"{Config.DATASET.name}train.pickle")
print(f"Generating {FILENAME}...")
with open(FILENAME, "wb") as f:
    labels = probablisticMissingGenerator(train[1], 0)
    pickle.dump(
        (train[0], labels), f)
print("Done✨")


FILENAME = os.path.join(GENERATED_DATA_ROOT,
                        f"{Config.DATASET.name}test.pickle")
print(f"Generating {FILENAME}...")
with open(FILENAME, "wb") as f:
    labels = probablisticMissingGenerator(test[1], 0)
    pickle.dump(
        (test[0], labels), f)
print("Done✨")


print("Generated data will save on {}".format(
    GENERATED_DATA_ROOT), file=sys.stderr)
for prob in sampled_value:
    filename = os.path.join(GENERATED_DATA_ROOT,
                            f"{Config.DATASET.name}train_{prob}.pickle")
    print(f"Generating {filename}...")
    with open(filename, "wb") as f:
        labels = probablisticMissingGenerator(train[1], prob)
        pickle.dump(
            (train[0], labels), f)
    print("Done✨")

print("Finish✅")
# -
