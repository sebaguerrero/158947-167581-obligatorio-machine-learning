#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils import (
    get_patches,
    plot_variance,
    plot_pca_classes,
    plot_3d_pca,
    view_sample_images,
    plot_pairplot,
)


def run_pca_iteration(
    X_train_std,
    y_train,
    X_test,
    y_test,
    scaler,
    n_components_list,
    ratio,
    version=1,
    show=True,
):
    results = []

    for n_components in n_components_list:
        suffix = f"{ratio}_v{version}_{n_components}"
        print(f"\n{'=' * 50}")
        print(f"Running with n_components = {n_components}")
        print(f"Suffix: {suffix}")
        print(f"{'=' * 50}")

        pca = PCA(n_components=n_components, whiten=True).fit(X_train_std)

        explained_var = pca.explained_variance_ratio_
        acumulated_var = np.cumsum(explained_var)
        if show:
            print(f"Explained variance ratio: {explained_var.sum():.4f}")
            plot_variance(explained_var, acumulated_var, show=show)

        X_train_pca = pca.transform(X_train_std)

        if show:
            plot_pca_classes(X_train_pca, y_train, show=show)
            plot_3d_pca(X_train_pca, y_train, show=show)
            plot_pairplot(X_train_pca, y_train, show=show)

        model = GaussianNB()

        model.fit(X_train_pca, y_train)
        y_train_pred = model.predict(X_train_pca)

        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        if show:
            print("\nTraining Classification Report:")
            print(classification_report(y_train, y_train_pred))

        x_test_std = scaler.transform(X_test)
        X_test_pca = pca.transform(x_test_std)

        y_test_pred = model.predict(X_test_pca)

        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        if show:
            print("\nTest Classification Report:")
            print(classification_report(y_test, y_test_pred))

        try:
            kaggle_patches, kaggle_filenames = get_patches(
                zip_path="/content/Test.zip",
                patches_path="content/test_patches",
                return_filenames=True,
            )

            if show:
                print(
                    f"Kaggle patches: {kaggle_patches.shape=}, {kaggle_patches.dtype=}, "
                    f"{kaggle_patches.min()=}, {kaggle_patches.max()=}"
                )

                view_sample_images(kaggle_patches, 25, figsize=(5, 5))

            pgm_kaggle_files_id = []
            for filename in tqdm(kaggle_filenames, desc="Processing filenames"):
                pgm_kaggle_files_id.append(filename[5:-4])

            X_kag = np.array(
                [im.flatten() for im in tqdm(kaggle_patches, desc="Construyendo X")]
            )

            X_kag_std = scaler.transform(X_kag)
            X_pca_kag = pca.transform(X_kag_std)
            y_kag = model.predict(X_pca_kag)

            y_kag_dict = {
                pgm_kaggle_files_id[i]: y_kag[i]
                for i in range(len(pgm_kaggle_files_id))
            }

            kaggle_hat = pd.DataFrame(
                list(y_kag_dict.items()), columns=["id", "target_feature"]
            )

            kaggle_hat["id"] = kaggle_hat["id"].astype(str)
            kaggle_hat["target_feature"] = kaggle_hat["target_feature"].astype(int)

            kaggle_hat.sort_values(by="id", inplace=True)

            if show:
                print(f"\nKaggle submission preview:")
                print(kaggle_hat.head())

            os.makedirs(
                f"./submissions/submission_{ratio}_v{version}_PCA", exist_ok=True
            )
            path = f"./submissions/submission_{ratio}_v{version}_PCA/submission_PCA_{suffix}.csv"
            kaggle_hat.to_csv(path, index=False)
            print(f"Saved submission to: {path}")

        except Exception as e:
            print(f"Warning: Could not process Kaggle data: {e}")

        result = {
            "n_components": n_components,
            "suffix": suffix,
            "explained_variance_ratio": explained_var.sum(),
            "train_f1": train_report["macro avg"]["f1-score"],
            "test_f1": test_report["macro avg"]["f1-score"],
        }
        results.append(result)
        if show:
            print(f"\nIteration {suffix} completed!")
            print(f"Explained Variance: {result['explained_variance_ratio']:.4f}")
            print(f"Train F1: {result['train_f1']:.4f}")
            print(f"Test F1: {result['test_f1']:.4f}")

    return results
