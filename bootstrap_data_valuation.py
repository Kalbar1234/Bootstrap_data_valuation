#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import scipy
import time


# In[ ]:


class bootstrap_data_valuation():
    folds = {}
    points_vector = {}
    subsets = {}
    subsets_performance = {}
    point_valuation = {}
    final_point_valuation = {}

    def __init__(self, data, model, target, n_bootstraps=100, n_folds=10):

        data["index"] = data.index
        self.data = data
        self.model = model
        self.target = target

        if n_folds == 1:
            self.kf = 1
            self.df_bootstrapping = pd.DataFrame(np.nan,
                                                 index=range(max(self.data.index) + 1),
                                                 columns=["mean_biased", "mean_unbiased"])
        else:
            self.kf = KFold(n_splits=n_folds, shuffle=True)
            columns_biased = [i for i in range(self.kf.get_n_splits())] + ["mean_biased"]
            columns_unbiased = [i for i in range(self.kf.get_n_splits())] + ["mean_unbiased"]

            self.df_biased = pd.DataFrame(np.nan,
                                          index=range(max(data.index) + 1),
                                          columns=columns_biased)

            self.df_unbiased = pd.DataFrame(np.nan,
                                            index=range(max(data.index) + 1),
                                            columns=columns_unbiased)

        self.n_bootstrap_subsets = n_bootstraps

    def create_subset(self, data):
        bootstrap_data = data.sample(len(data), replace=True)
        out_of_bag_data = data.drop(bootstrap_data.index)
        return bootstrap_data, out_of_bag_data

    # Remeber to update the rest of the code in terms of the updated return data
    def calculate_performance(self, model, train_data, test_data):
        Y_train = train_data[self.target]
        X_train = train_data.drop([self.target], axis=1)

        if ("index" in X_train.columns):
            X_train = X_train.drop("index", axis=1)

        X_test = test_data.drop([self.target], axis=1)
        Y_test = test_data[self.target]

        if ("index" in X_test.columns):
            X_test = X_test.drop("index", axis=1)

        model = model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        acc_test = accuracy_score(Y_test, Y_pred)

        return acc_test, Y_test, Y_pred

    def perform_kfold(self, k=1):
        fold = 0
        self.point_valuation = {}

        for train, test in self.kf.split(self.data):

            print(f"Fold: {fold}")
            self.point_valuation[fold] = {}
            self.subsets[fold] = {}
            subset_performance = []
            self.folds[fold] = {"train": self.data.iloc[train],
                                "test": self.data.iloc[test]}

            for point in self.data.iloc[train]["index"]:
                self.point_valuation[fold][point] = {"inside": [],
                                                     "outside_biased": [],
                                                     "outside_unbiased": []}

            data = self.data.loc[self.folds[fold]["train"].index.tolist()]
            start_time = time.time()

            for subset in range(self.n_bootstrap_subsets):

                if ((subset % (self.n_bootstrap_subsets * 0.10) == 0) or (subset % 500 == 0)):
                    print(f"Subset number: {subset}")

                bootstrap_data, oob_data = self.create_subset(data)
                self.subsets[fold][subset] = {"bootstrap": bootstrap_data,
                                              "out-of-bag": oob_data}

                acc, y_test, y_pred = self.calculate_performance(self.model, bootstrap_data, oob_data)

                subset_performance.append(acc)
                self.performance(fold, bootstrap_data, acc, k, data, y_test, y_pred)

            print("--- %s seconds ---" % (time.time() - start_time), "\n")
            self.subsets_performance[fold] = subset_performance
            fold += 1

        self.df_biased["mean_biased"] = self.df_biased.mean(axis=1)
        self.df_unbiased["mean_unbiased"] = self.df_unbiased.mean(axis=1)

        self.df_biased["index"] = self.df_biased.index
        self.df_unbiased["index"] = self.df_unbiased.index

        return [self.df_biased, self.df_unbiased]

    def bootstrapping(self, k):
        self.only_bootstrapping = {}
        self.point_valuation2 = {}
        for point in self.data.index:
            self.point_valuation2[point] = {"inside": [],
                                            "outside_biased": [],
                                            "outside_unbiased": []}
        start_time = time.time()
        for subset in range(self.n_bootstrap_subsets):
            if ((subset % (self.n_bootstrap_subsets * 0.10) == 0) or (subset % 500 == 0)):
                print(f"Subset number: {subset}")

            bootstrap_data, oob_data = self.create_subset(self.data)

            self.only_bootstrapping[subset] = {"bootstrap": bootstrap_data,
                                               "out-of-bag": oob_data}

            acc, y_test, y_pred = self.calculate_performance(self.model, bootstrap_data, oob_data)
            self.performance2(bootstrap_data, acc, k, self.data, y_test, y_pred)
        print("--- %s seconds ---" % (time.time() - start_time), "\n")

    def performance2(self, bootstrap_data, acc, k, data, y_test, y_pred):
        for point in self.data.index:
            if (point in bootstrap_data.index):
                #                 if (len(bootstrap_data[bootstrap_data["index"]==point])==k):
                self.point_valuation2[point]["inside"].append(acc)
            else:
                self.point_valuation2[point]["outside_biased"].append(acc)
                idx = y_test.index.tolist().index(point)

                # Remove point anyway
                y_test_all = y_test.drop(labels=[point])
                y_pred_all = np.delete(y_pred, idx)
                acc_all = accuracy_score(y_test_all, y_pred_all)
                self.point_valuation2[point]["outside_unbiased"].append(acc_all)

            #                 if(y_test.iloc[idx] == y_pred[idx]):
            #                     self.point_valuation2[point]["hit"].append(1)
            #                 else:
            #                     self.point_valuation2[point]["hit"].append(0)

            score_inside = self.point_valuation2[point]["inside"]

            score_outside_biased = self.point_valuation2[point]["outside_biased"]
            score_outside_unbiased = self.point_valuation2[point]["outside_unbiased"]
            #             hits = self.point_valuation2[point]['hit']

            try:
                inside = (sum(score_inside) / len(score_inside)) * 100

                outside_biased = (sum(score_outside_biased) / len(score_outside_biased)) * 100
                outside_unbiased = (sum(score_outside_unbiased) / len(score_outside_unbiased)) * 100
                #                 outside_hits = (sum(hits)/len(hits)) * 100

                quality_biased = inside - outside_biased
                #                 quality_hits =  inside - outside_hits
                quality_unbiased = inside - outside_unbiased

                self.point_valuation2[point]["difference_biased"] = quality_biased
                #                 self.point_valuation2[point]["difference_hits"] = quality_hits
                self.point_valuation2[point]["difference_unbiased"] = quality_unbiased

                self.df_bootstrapping.at[point, "mean_biased"] = quality_biased
                #                 self.df_bootstrapping.at[point, "mean_hits"] = quality_hits
                self.df_bootstrapping.at[point, "mean_unbiased"] = quality_unbiased
            except:
                pass

    def performance(self, fold, bootstrap_data, acc, k, data, y_test, y_pred):
        for point in self.folds[fold]["train"].index:

            if (point in bootstrap_data.index):
                if (len(bootstrap_data[bootstrap_data["index"] == point]) == k):
                    self.point_valuation[fold][point]["inside"].append(acc)
            else:
                self.point_valuation[fold][point]["outside_biased"].append(acc)
                idx = y_test.index.tolist().index(point)

                # Remove point anyway
                y_test_unbiased = y_test.drop(labels=[point])
                y_pred_unbiased = np.delete(y_pred, idx)
                acc_unbiased = accuracy_score(y_test_unbiased, y_pred_unbiased)
                self.point_valuation[fold][point]["outside_unbiased"].append(acc_unbiased)

            score_inside = self.point_valuation[fold][point]["inside"]
            score_outside_biased = self.point_valuation[fold][point]["outside_biased"]
            score_outside_unbiased = self.point_valuation[fold][point]["outside_unbiased"]

            try:
                inside = (sum(score_inside) / len(score_inside)) * 100

                outside_biased = (sum(score_outside_biased) / len(score_outside_biased)) * 100
                outside_unbiased = (sum(score_outside_unbiased) / len(score_outside_unbiased)) * 100

                quality_biased = inside - outside_biased
                quality_unbiased = inside - outside_unbiased

                self.point_valuation[fold][point]["mean_biased"] = quality_biased
                self.point_valuation[fold][point]["mean_unbiased"] = quality_unbiased

                self.df_biased.at[point, fold] = quality_biased
                self.df_unbiased.at[point, fold] = quality_unbiased
            except:
                pass

    def data_valuation(self):
        if self.kf == 1:
            self.bootstrapping(1)
            return self.df_bootstrapping
        else:
            dfs = self.perform_kfold()
            unbiased_df = dfs[1]
            return unbiased_df
