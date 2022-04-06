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
        self.kf = KFold(n_splits=n_folds,shuffle=True)
        self.n_bootstrap_subsets = n_bootstraps
        
        self.df = pd.DataFrame(np.nan,
                               index=range(max(data.index)),
                               columns=range(self.kf.get_n_splits()))

    def create_subset(self, data):
        bootstrap_data = data.sample(len(data), replace=True)
        out_of_bag_data = data.drop(bootstrap_data.index)
        return bootstrap_data, out_of_bag_data
    
    def update_vector(self, bootstrap_subset, subset_n, fold):
        points = bootstrap_subset["index"]
        for point in points:
            self.points_vector[fold][point][subset_n] = 1

    def calculate_performance(self, model, train_data, test_data):
        Y_train = train_data[self.target]
        X_train = train_data.drop([self.target],axis=1)

        if ("index" in X_train.columns):
            X_train = X_train.drop("index",axis=1)

        X_test = test_data.drop([self.target],axis=1)
        Y_test = test_data[self.target]

        if ("index" in X_test.columns):
            X_test = X_test.drop("index",axis=1)


        model = model.fit(X_train,Y_train) 
        Y_pred = model.predict(X_test)
        acc_test = accuracy_score(Y_test, Y_pred)

        return acc_test
        
    def perform_kfold(self):
        fold=0
        self.point_valuation = {}
        for train, test in self.kf.split(self.data):
            print(fold)
            self.point_valuation[fold] = {}
            self.subsets[fold] = {}
            subset_performance = []
            self.folds[fold] = {"train":self.data.iloc[train],
                                "test":self.data.iloc[test]}

            for point_idx in self.data.iloc[train]["index"]:
                self.point_valuation[fold][point_idx] = {"inside":[],
                                                         "outside":[]}
                
            data = self.data.loc[self.folds[fold]["train"].index.tolist()]
            
            for subset in range(self.n_bootstrap_subsets):
                bootstrap_data, oob_data = self.create_subset(data)
                self.subsets[fold][subset] = {"bootstrap": bootstrap_data,
                                              "out-of-bag": oob_data}
                acc = self.calculate_performance(self.model, bootstrap_data, oob_data)

                subset_performance.append(acc)
                self.performance(fold, bootstrap_data, acc)
                
            self.subsets_performance[fold] = subset_performance
            fold+=1
                
    def performance(self, fold, bootstrap_data, acc):
        for point in self.folds[fold]["train"].index:
            if (point in bootstrap_data.index):
                self.point_valuation[fold][point]["inside"].append(acc)
            else:
                self.point_valuation[fold][point]["outside"].append(acc)
                
            score_inside = self.point_valuation[fold][point]["inside"]
            score_outside = self.point_valuation[fold][point]["outside"]
            
            try:
                inside = (sum(score_inside)/len(score_inside)) * 100
                outside = (sum(score_outside)/len(score_outside)) * 100
                quality =  inside - outside
                self.point_valuation[fold][point]["avg_change_in_acc"] = quality
                self.df.at[point,fold] = quality
            except:
                pass

    def get_quality(self):
        self.perform_kfold()
        for point in self.data.index:
            self.final_point_valuation[point] = []

        for fold in self.point_valuation.keys():
            for point in self.point_valuation[fold].keys():
                self.final_point_valuation[point].append(self.point_valuation[fold][point]["avg_change_in_acc"])
    
    def transform_df(self):        
        self.df["mean"] = self.df.mean(numeric_only=True, axis=1)
        self.df = self.df.sort_values(["mean"], ascending=False)
        self.df["index"] = self.df.index
        
        return self.df
        
    def run_test2(self, threshold=-2, criteria=True, noise_reduction=True):
        print("Threshold: ",threshold)
        noise_to_remove = []
        noise_df = pd.DataFrame()
        threshold = threshold
        for fold in self.folds.keys():
            print(f"\nFold: {fold}")
            df = self.df[~self.df[fold].isnull()]

            if noise_reduction:
                idx_to_drop = df[df[fold]<threshold]["index"]
                print(f"Number of noisy datapoints: {len(idx_to_drop)}")
            else:
                idx_to_drop = df[df[fold]>threshold]["index"]
                print(f"Number of noisy datapoints: {len(idx_to_drop)}")
                
            train_data_fold = self.folds[fold]["train"]
            train_data_reduced_fold = self.folds[fold]["train"].drop(idx_to_drop)
            test_data_fold = self.folds[fold]["test"]

            try:
                print(f"Training on whole data: {len(train_data_fold)}")

                baseline_acc = self.calculate_performance(self.model,
                                      train_data_fold,
                                      test_data_fold)

                print(f"Training on reduced data: {len(train_data_reduced_fold)}")

                reduction_acc = self.calculate_performance(self.model,
                                      train_data_reduced_fold,
                                      test_data_fold)
                
                change = round((reduction_acc - baseline_acc)*100,5)
                print(f"Baseline accuracy: {round(baseline_acc,5)}")
                print(f"Accuracy after reduction: {round(reduction_acc,5)}")
                print(f"Change: {change}%")
                
                if ((change > 0)&(criteria)):
                    print("Adding")
                    noise_to_remove += list(idx_to_drop)
                    print("Printing: ",noise_to_remove)
                if(not criteria):
                    print("Adding anyways")
                    noise_to_remove += list(idx_to_drop)
                    print("Printing: ",noise_to_remove)
                    

            except:
                print("No datapoints found")
        
        print("\nLength of the whole list: ", len(noise_to_remove))
        print("Length of the list without duplicates: ", len(set(noise_to_remove)))
        noise_df["index"] = list(set(noise_to_remove))
        return noise_df

