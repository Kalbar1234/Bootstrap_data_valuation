#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# In[2]:


def vis_overlap(df, parameter_dict=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    color = sns.color_palette("hls", 2)

    sns.scatterplot(x="feature2",
                    y="feature1",
                    hue="class",
                    palette=color,
                    data=df,
                    legend="full",
                    ax=ax1)
    ax2.set_title(f"Distribution overlap: {parameter_dict['overlap']}")

    ax2.hist(df[df["class"] == 0]["feature2"], bins=30, edgecolor="black", color=color[0])
    ax2.hist(df[df["class"] == 1]["feature2"], bins=30, edgecolor="black", color=color[1])

    ax2.set_xlabel("feature2")
    ax2.set_ylabel("number of points")

    if parameter_dict:
        ax1.set_title(f"Number of points: {parameter_dict['n']}")
        ax2.set_title(f"Distribution of points for {parameter_dict['overlap']} overlap")
        plt.close(fig)
        fig.savefig(f"{parameter_dict['n']}_{parameter_dict['overlap']}_data_overlap.jpg")

    else:
        ax1.set_title(f"Number of points: {len(df)}")
        ax2.set_title(f"Distribution of points for {parameter_dict['overlap']} overlap")
        plt.close(fig)
        fig.savefig("data_overlap.jpg")


# In[ ]:

def heat_map(df, title, col="class", measure='mean', sep=True, figsize=(15, 5), parameter_dict=None, vmin=None, vmax=None, xmin=None,ymin=None,xmax=None,ymax=None):
    print("title: ", title)

    if sep:
        g = sns.FacetGrid(df, col=col)
    else:
        g = sns.FacetGrid(df)

    def facet_scatter(x, y, c, **kwargs):
        """Draw scatterplot with point colors from a faceted DataFrame columns."""
        kwargs.pop("color")
        if (xmin or ymin):
            ax = plt.gca()
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.scatter(x,y,c=c,**kwargs)
        else:
            plt.scatter(x, y, c=c, **kwargs)

    if not(vmin or vmax):
        vmin, vmax = df[measure].min(), df[measure].max()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    g = g.map(facet_scatter, 'feature2', 'feature1', measure,
              s=100, alpha=0.9, vmin=vmin, vmax=vmax, cmap=cmap)

    g.fig.set_size_inches(figsize)
    g.fig.suptitle(f"n_points: {parameter_dict['n']}, noise_level: {parameter_dict['noise']}, overlap: "
                   f"{parameter_dict['overlap']}, max_depth: {parameter_dict['tree']}\n" + title, y=1.0)

    # Make space for the colorbar
    g.fig.subplots_adjust(right=.92)

    # Define a new Axes where the colorbar will go
    cax = g.fig.add_axes([.94, .25, .02, .6])

    # Get a mappable object with the same colormap as the data
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

    # Draw the colorbar
    g.fig.colorbar(points, cax=cax)

    plt.close(g.fig)
    g.fig.savefig(f"{title}.jpg")
    return g


# In[3]:


def vis_folds(df, n_splits, parameter_dict=None):
    if n_splits == 1:
        rows, columns = 2, 1
    else:
        rows = int(n_splits / 2)
        columns = int(n_splits / rows)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 30))

    fig.tight_layout(pad=10.0)
    classes = {"0": {"negative": [], "positive": []},
               "1": {"negative": [], "positive": []}}

    custom_legend = [Line2D([], [], marker='o', color='tomato', linestyle='None'),
                     Line2D([], [], marker='o', color='mediumturquoise', linestyle='None'),
                     Line2D([], [], marker='o', color='black', linestyle='None')]

    for index, ax in enumerate(axes.reshape(-1, )):
        try:
            data = df[~df[index].isnull()]
            class_0_n = len(data[(data["class"] == 0) & (data["mean"] < 0)])
            class_1_n = len(data[(data["class"] == 1) & (data["mean"] < 0)])

            classes["0"]["negative"].append(class_0_n)
            classes["1"]["negative"].append(class_1_n)

            class_0_p = len(data[(data["class"] == 0) & (data["mean"] > 0)])
            class_1_p = len(data[(data["class"] == 1) & (data["mean"] > 0)])

            classes["0"]["positive"].append(class_0_p)
            classes["1"]["positive"].append(class_1_p)

            plot = sns.scatterplot(x="index",
                                   y=index,
                                   hue="class_with_noise",
                                   palette={2: "black",
                                            0: "tomato",
                                            1: "mediumturquoise"},
                                   legend=False,
                                   data=data,
                                   ax=ax)
        except:
            fig.delaxes(ax)

        plot.legend(custom_legend, ["Class 0", "Class 1", "Noise"], loc='lower right')

        plot.set_xlabel("point index")
        plot.set_ylabel("Mean change in accuracy")
        plot.grid()
        plot.axhline(color="black")
        plot.set_title(
            f"Fold: {index}\nClass 0 points with mean < 0: {class_0_n} and mean > 0: {class_0_p}\nClass 1 points with "
            f"mean < 0: {class_1_n} and mean > 0: {class_1_p}")

    mean_0_n = np.array(classes["0"]["negative"]).mean()
    mean_1_n = np.array(classes["1"]["negative"]).mean()
    mean_0_p = np.array(classes["0"]["positive"]).mean()
    mean_1_p = np.array(classes["1"]["positive"]).mean()

    if parameter_dict:
        fig.suptitle(
            f"On average class 0 has {mean_0_n} points with mean < 0 and {mean_0_p} points with mean > 0\nOn average "
            f"class 1 has {mean_1_n} points with mean < 0 and {mean_1_p} points with mean > 0 "
            f"\nn_points: {parameter_dict['n']}, overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}",
            y=1.0)

        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"_folds.jpg")
    else:
        fig.suptitle(
            f"On average class 0 has {mean_0_n} points with mean < 0 and {mean_0_p} points with mean > 0\nOn average "
            f"class 1 has {mean_1_n} points with mean < 0 and {mean_1_p} points with mean > 0",
            y=1.0)
        plt.close(fig)
        fig.savefig("folds.jpg")


# In[]:

def vis_hist_included(df, bins=10, parameter_dict=None, measure="mean"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    # Class 0 distribution
    df[df["class"] == 0].hist(measure,
                              bins=bins,
                              ax=ax1,
                              legend=True,
                              edgecolor="black",
                              fc=red)
    ax1.set_title("Class 0 distribution")
    ax1.legend(["Class 0"], loc='upper right')

    # Class 1 distribution
    df[df["class"] == 1].hist(measure,
                              bins=bins,
                              ax=ax3,
                              legend=True,
                              edgecolor="black",
                              fc=blue)
    ax3.set_title("Class 1 distribution")
    ax3.legend(["Class 1"], loc='upper right')

    # Overlap of the distributions 
    df[df["class"] == 0].hist(measure,
                              bins=bins,
                              ax=ax2,
                              legend=True,
                              edgecolor="black",
                              fc=red)

    df[df["class"] == 1].hist(measure,
                              bins=bins,
                              ax=ax2,
                              legend=True,
                              edgecolor="black",
                              fc=blue)

    ax2.set_title("Overlap")
    ax2.legend(["Class 0", "Class 1"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(
            f"Average accuracy change Distribution - noise included\nn_points: {parameter_dict['n']}, "
            f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"avg_mean_distribution1.jpg")
    else:
        fig.suptitle("Average accuracy change Distribution - noise included", fontsize=15)
        plt.close(fig)
        fig.savefig("avg_mean_distribution1.jpg")


def vis_hist_excluded(df, bins=10, parameter_dict=None, measure="mean"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    class_0_df = df[df["class_with_noise"] == 0]
    class_1_df = df[df["class_with_noise"] == 1]

    # Class 0 distribution
    class_0_df.hist(measure,
                    bins=bins,
                    ax=ax1,
                    legend=True,
                    edgecolor="black",
                    fc=red)
    ax1.set_title("Class 0 distribution")
    ax1.legend(["Class 0"], loc='upper right')

    # Class 1 distribution
    class_1_df.hist(measure,
                    bins=bins,
                    ax=ax3,
                    legend=True,
                    edgecolor="black",
                    fc=blue)
    ax3.set_title("Class 1 distribution")
    ax3.legend(["Class 1"], loc='upper right')

    # Overlap of the distributions 
    class_0_df.hist(measure,
                    bins=bins,
                    ax=ax2,
                    legend=True,
                    edgecolor="black",
                    fc=red)

    class_1_df.hist(measure,
                    bins=bins,
                    ax=ax2,
                    legend=True,
                    edgecolor="black",
                    fc=blue)

    ax2.set_title("Overlap")
    ax2.legend(["Class 0", "Class 1"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(
            f"Average accuracy change Distribution - noise excluded\nn_points: {parameter_dict['n']}, "
            f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"avg_mean_distribution2.jpg")
    else:
        fig.suptitle("Average accuracy change Distribution - noise excluded", fontsize=15)
        plt.close(fig)
        fig.savefig("avg_mean_distribution2.jpg")


def vis_hist_separate(df, bins=10, parameter_dict=None, measure="mean"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    class_0_df = df[df["class_with_noise"] == 0]
    class_1_df = df[df["class_with_noise"] == 1]
    noise_df = df[df["class_with_noise"] == 2]

    # Noise distribution
    noise_df.hist(measure,
                  legend=True,
                  color=(0.549, 0.549, 0.549),
                  edgecolor="black",
                  ax=ax1)
    ax1.set_title("Noise distribution")
    ax1.legend(["Noise"], loc='upper right')

    # Overlap of the distributions 
    class_0_df.hist(measure,
                    bins=bins,
                    ax=ax2,
                    legend=True,
                    edgecolor="black",
                    fc=red)

    class_1_df.hist(measure,
                    bins=bins,
                    ax=ax2,
                    legend=True,
                    edgecolor="black",
                    fc=blue)

    noise_df.hist(measure,
                  legend=True,
                  fc=(0, 0, 0, 0.5),
                  ax=ax2)

    ax2.set_title("Overlap")
    ax2.legend(["Class 0", "Class 1", "Noise"], loc='upper right')

    # Overlap of the distributions - normalized
    class_0_df.hist("mean",
                    bins=bins,
                    ax=ax3,
                    legend=True,
                    edgecolor="black",
                    fc=red,
                    weights=np.ones_like(class_0_df.index) / len(class_0_df.index))

    class_1_df.hist("mean",
                    bins=bins,
                    ax=ax3,
                    legend=True,
                    edgecolor="black",
                    fc=blue,
                    weights=np.ones_like(class_1_df.index) / len(class_1_df.index))

    noise_df.hist("mean",
                  legend=True,
                  edgecolor="black",
                  ax=ax3,
                  fc=(0, 0, 0, 0.5),
                  weights=np.ones_like(noise_df.index) / len(noise_df.index))

    ax3.set_title("Overlap but normalized")
    ax3.legend(["Class 0", "Class 1", "Noise"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(
            f"Average accuracy change Distribution - noise as a seperate class\nn_points: {parameter_dict['n']}, "
            f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"avg_mean_distribution3.jpg")
    else:
        fig.suptitle("Average accuracy change Distribution - noise as a seperate class", fontsize=15)
        plt.close(fig)
        fig.savefig("avg_mean_distribution3.jpg")


# In[]:

def vis_hist_included_folds(df, n_folds, bins=10, figsize=(15, 30), parameter_dict=None):
    fig, axs = plt.subplots(n_folds, 3, figsize=figsize)
    fig.tight_layout(pad=10.0)

    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    for index, ax in enumerate(axs):
        data = df[~df[index].isnull()]

        # Class 0 distribution
        data[data["class"] == 0].hist(index,
                                      bins=bins,
                                      ax=ax[0],
                                      legend=True,
                                      edgecolor="black",
                                      fc=red)
        ax[0].set_title("Class 0 distribution")
        ax[0].legend(["Class 0"], loc='upper right')

        # Class 1 distribution
        data[data["class"] == 1].hist(index,
                                      bins=bins,
                                      ax=ax[2],
                                      legend=True,
                                      edgecolor="black",
                                      fc=blue)
        ax[2].set_title("Class 1 distribution")
        ax[2].legend(["Class 1"], loc='upper right')

        # Overlap of the distributions
        data[data["class"] == 0].hist(index,
                                      bins=bins,
                                      ax=ax[1],
                                      legend=True,
                                      edgecolor="black",
                                      fc=red)

        data[data["class"] == 1].hist(index,
                                      bins=bins,
                                      ax=ax[1],
                                      legend=True,
                                      edgecolor="black",
                                      fc=blue)

        ax[1].set_title(f"Overlap, fold: {index}")
        ax[1].legend(["Class 0", "Class 1"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(
            f"Accuracy change Distribution for individual folds - noise included\nn_points: {parameter_dict['n']}, "
            f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"mean_distribution_with_noise_folds.jpg")
    else:
        fig.suptitle(f"Accuracy change Distribution for individual folds - noise included", fontsize=15)
        plt.close(fig)
        fig.savefig(f"mean_distribution_with_noise_folds.jpg")


def vis_hist_excluded_folds(df, n_folds, bins=10, figsize=(15, 30), parameter_dict=None):
    fig, axs = plt.subplots(n_folds, 3, figsize=figsize)
    fig.tight_layout(pad=10.0)

    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    for index, ax in enumerate(axs):
        data = df[~df[index].isnull()]

        # Class 0 distribution
        data[data["class_with_noise"] == 0].hist(index,
                                                 bins=bins,
                                                 ax=ax[0],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=red)
        ax[0].set_title("Class 0 distribution")
        ax[0].legend(["Class 0"], loc='upper right')

        # Class 1 distribution
        data[data["class_with_noise"] == 1].hist(index,
                                                 bins=bins,
                                                 ax=ax[2],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=blue)
        ax[2].set_title("Class 1 distribution")
        ax[2].legend(["Class 1"], loc='upper right')

        # Overlap of the distributions
        data[data["class_with_noise"] == 0].hist(index,
                                                 bins=bins,
                                                 ax=ax[1],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=red)

        data[data["class_with_noise"] == 1].hist(index,
                                                 bins=bins,
                                                 ax=ax[1],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=blue)

        ax[1].set_title(f"Overlap, fold: {index}")
        ax[1].legend(["Class 0", "Class 1"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(
            f"Accuracy change distribution for individual folds - noise excluded\nn_points: {parameter_dict['n']}, "
            f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
            f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"_mean_distribution_without_noise_folds.jpg")
    else:
        fig.suptitle(f"Accuracy change distribution for individual folds - noise excluded", fontsize=15)
        plt.close(fig)
        fig.savefig(f"mean_distribution_without_noise_folds.jpg")


def vis_hist_separate_folds(df, n_folds, bins=10, figsize=(15, 30), parameter_dict=None):
    fig, axs = plt.subplots(n_folds, 3, figsize=figsize)
    fig.tight_layout(pad=10.0)

    red = (0.984, 0.196, 0.196, 0.5)
    blue = (0.196, 0.984, 0.913, 0.5)

    for index, ax in enumerate(axs):
        data = df[~df[index].isnull()]

        # Noise distribution
        data[data["class_with_noise"] == 2].hist(index,
                                                 legend=True,
                                                 color=(0.549, 0.549, 0.549),
                                                 edgecolor="black",
                                                 ax=ax[0])
        ax[0].set_title("Noise distribution")
        ax[0].legend(["Noise"], loc='upper right')

        # Overlap of the distributions
        data[data["class_with_noise"] == 0].hist(index,
                                                 bins=bins,
                                                 ax=ax[1],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=red)

        data[data["class_with_noise"] == 1].hist(index,
                                                 bins=bins,
                                                 ax=ax[1],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=blue)

        data[data["class_with_noise"] == 2].hist(index,
                                                 legend=True,
                                                 fc=(0, 0, 0, 0.5),
                                                 ax=ax[1])

        ax[1].set_title(f"Overlap, fold:{index}")
        ax[1].legend(["Class 0", "Class 1", "Noise"], loc='upper right')

        # Overlap of the distributions - normalized
        data[data["class_with_noise"] == 0].hist(index,
                                                 bins=bins,
                                                 ax=ax[2],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=red,
                                                 weights=np.ones_like(data[data["class_with_noise"] == 0].index) / len(
                                                     data[data["class_with_noise"] == 0].index))

        data[data["class_with_noise"] == 1].hist(index,
                                                 bins=bins,
                                                 ax=ax[2],
                                                 legend=True,
                                                 edgecolor="black",
                                                 fc=blue,
                                                 weights=np.ones_like(data[data["class_with_noise"] == 1].index) / len(
                                                     data[data["class_with_noise"] == 1].index))

        data[data["class_with_noise"] == 2].hist(index,
                                                 legend=True,
                                                 edgecolor="black",
                                                 ax=ax[2],
                                                 fc=(0, 0, 0, 0.5),
                                                 weights=np.ones_like(data[data["class_with_noise"] == 2].index) / len(
                                                     data[data["class_with_noise"] == 2].index))

        ax[2].set_title("Overlap but normalized")
        ax[2].legend(["Class 0", "Class 1", "Noise"], loc='upper right')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(f"Mean accuracy change", fontsize=15)
    plt.ylabel("Number of points", fontsize=15)

    if parameter_dict:
        fig.suptitle(f"Accuracy change distribution - noise as a separate class\nn_points: {parameter_dict['n']}, "
                     f"overlap: {parameter_dict['overlap']}, noise_level: {parameter_dict['noise']},"
                     f" max_depth: {parameter_dict['tree']}", fontsize=15)
        plt.close(fig)
        fig.savefig(
            f"{parameter_dict['n']}_{parameter_dict['noise']}_{parameter_dict['overlap']}_{parameter_dict['tree']}"
            f"_mean_distribution_noise_class_folds.jpg")
    else:
        fig.suptitle("Accuracy change distribution - noise as a separate class", fontsize=15)
        plt.close(fig)
        fig.savefig("mean_distribution_noise_class_folds.jpg")


def create_vis(df, n_folds, parameter_dict=None, bins=10, figsize=(15, 30)):
    vis_overlap(df, parameter_dict=parameter_dict)

    print("Creating heat map...")
    heat_map(df, "Heat map with noise included - classes separated", parameter_dict=parameter_dict, sep=True)
    heat_map(df, "Heat map with noise included", parameter_dict=parameter_dict, sep=False)
    heat_map(df, "Heat map with noise as a separate class - classes separated", parameter_dict=parameter_dict, col="class_with_noise", sep=True)
    heat_map(df, "Heat map with noise as a separate class", parameter_dict=parameter_dict, col="class_with_noise", sep=False)



    print("Creating general histograms...")
    vis_hist_included(df, parameter_dict=parameter_dict)
    vis_hist_excluded(df, parameter_dict=parameter_dict)
    vis_hist_separate(df, parameter_dict=parameter_dict)

    if n_folds > 1:
        print("Creating histograms for each fold...")
        vis_folds(df, n_folds, parameter_dict=parameter_dict)
        vis_hist_included_folds(df, n_folds, parameter_dict=parameter_dict)
        vis_hist_excluded_folds(df, n_folds, parameter_dict=parameter_dict)
        vis_hist_separate_folds(df, n_folds, parameter_dict=parameter_dict)
