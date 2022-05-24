#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from statistics import NormalDist
from sklearn.utils import shuffle
import random


# In[2]:


def flip(row, idx_list):
    if row["index"] in idx_list:
        if row["class"] == 0:
            row["class"] = 1
        else:
            row["class"] = 0
    return row


def get_overlap(l1, l2):
    # print(f"There were {len(l1)} noisy datapoints")
    overlap = list(set(l1) & set(l2))
    # print(f"{len(l2)} points were identified as noise")
    # print(
    #     f"{len(overlap)}/{len(l2)} were correctly identified as noise. That is {round(len(overlap) / len(l2), 3) * 100}%")
    return overlap


def check(overlap, overlap_target, threshold=0.01):
    overlap_threshold = overlap_target * threshold
    low, high = overlap_target - overlap_threshold, overlap_target + overlap_threshold
    if low <= overlap <= high:
        return True
    return False


def get_mean_for_overlap(overlap_target, precision=3):
    overlap_target = overlap_target
    mean = 0
    overlap = NormalDist(mu=0, sigma=1).overlap(NormalDist(mu=mean, sigma=1))
    while not check(overlap, overlap_target):
        mean += 0.01
        overlap = round(NormalDist(mu=0, sigma=1).overlap(NormalDist(mu=mean, sigma=1)), precision)
    #     print("Mean: ", mean)
    return round(mean, 2)


def create_synthethic_data(overlap_target, n_points=1000, precision=3, exact_points=False):
    feature1 = []
    feature2 = []
    classes = []

    n_points = n_points
    mean = get_mean_for_overlap(overlap_target, precision=precision)

    if exact_points:
        for point in range(0, (int(n_points / 2))):
            feature1.append(np.random.normal())
            feature2.append(np.random.normal())
            classes.append(0)

        for point in range((int(n_points / 2)), n_points):
            feature1.append(np.random.normal())
            feature2.append(np.random.normal(loc=mean))
            classes.append(1)
    else:
        for point in range(n_points):
            if np.random.normal() > 0:
                feature1.append(np.random.normal())
                feature2.append(np.random.normal())
                classes.append(0)
            else:
                feature1.append(np.random.normal())
                feature2.append(np.random.normal(loc=mean))
                classes.append(1)

    if exact_points:
        df = pd.DataFrame({"feature1": feature1, "feature2": feature2, "class": classes})
        df = shuffle(df)
    else:
        df = pd.DataFrame({"feature1": feature1, "feature2": feature2, "class": classes})

    return df


def get_noise_list(noise_level, data):
    noise = len(data) * noise_level
    noise_counter = 0
    df_noisy = data.copy()
    noise_idx = pd.DataFrame(columns=data.columns)

    while noise_counter != noise:
        row = df_noisy.sample(replace=True)
        if row["index"].iloc[0] not in noise_idx["index"]:
            noise_idx = noise_idx.append(row)
            noise_counter += 1
    return noise_idx


def prepare_data(n_points, noise_level, overlap, exact_points=False, weights=None):
    data_pure = create_synthethic_data(overlap, n_points=n_points, exact_points=exact_points)
    data_pure["index"] = data_pure.index

    data_noisy = data_pure.copy()

    # weights in format of a dictionary: key=class, value=weight
    if weights:
        df = data_pure.copy()
        df['weights'] = df['class'].apply(lambda x: weights[x])
        noise_idx = df.sample(frac=noise_level, weights='weights')
    else:
        noise_idx = data_noisy.sample(frac=noise_level, random_state=18)
        # noise_idx = get_noise_list(noise_level, data_pure)

    data_noisy = data_noisy.apply(flip,
                                  idx_list=noise_idx["index"].tolist(),
                                  axis=1)

    data_red_noise = data_pure.drop(noise_idx["index"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.set_title(f'Pure, points: {n_points}')
    sns.scatterplot(x="feature2",
                    y="feature1",
                    hue="class",
                    palette=sns.color_palette("hls", 2),
                    data=data_pure,
                    legend="full",
                    ax=ax1)

    ax2.set_title(f'Artificial Noise: {noise_level * 100}%')
    sns.scatterplot(x="feature2",
                    y="feature1",
                    hue="class",
                    palette=sns.color_palette("hls", 2),
                    data=data_noisy,
                    legend="full",
                    ax=ax2)

    ax3.set_title(f'After removing artificial noise ({int(n_points * noise_level)} points removed)')
    sns.scatterplot(x="feature2",
                    y="feature1",
                    hue="class",
                    palette=sns.color_palette("hls", 2),
                    data=data_red_noise,
                    legend="full",
                    ax=ax3)
    plt.close(fig)
    fig.savefig(f"dataset_{n_points}_{noise_level}_{overlap}.jpg")

    return data_pure, data_noisy, data_red_noise, noise_idx

# In[ ]:
