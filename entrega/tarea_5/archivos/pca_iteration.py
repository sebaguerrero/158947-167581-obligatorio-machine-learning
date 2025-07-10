#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    ShuffleSplit,
    GridSearchCV,
    StratifiedShuffleSplit,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from skimage import feature
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import joblib

from utils import (
    get_patches,
    plot_variance,
    plot_pca_classes,
    plot_3d_pca,
    plot_pairplot,
)


def get_models_and_param_grids():
    models = {
        "MLPClassifier": MLPClassifier(
            random_state=0,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.3,
        ),
        # "AdaBoost": AdaBoostClassifier(
        #     estimator=DecisionTreeClassifier(), random_state=0
        # ),
        # "GradientBoosting": GradientBoostingClassifier(random_state=0),
        # "XGBoost": XGBClassifier(
        #     eval_metric="logloss", random_state=0, n_jobs=-1, verbosity=0
        # ),
        # "DecisionTree": DecisionTreeClassifier(random_state=0),
        # "Bagging": BaggingClassifier(
        #     estimator=DecisionTreeClassifier(), random_state=0, n_jobs=-1
        # ),
        # "RandomForest": RandomForestClassifier(
        #     random_state=0, max_features="sqrt", n_jobs=-1
        # ),
        # "KNeighbors": KNeighborsClassifier(metric="euclidean"),
        # "LogisticRegression": LogisticRegression(max_iter=2000, random_state=0),
        # "GaussianNB": GaussianNB(),
    }

    param_grids = {
        "MLPClassifier": {
            "max_iter": [1000, 1500, 2000],
            "solver": ["adam"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "hidden_layer_sizes": [(50,), (100,), (50, 50), [60, 120, 15]],
            "alpha": [0.0001, 0.001],
            "activation": ["relu"],
        },
        # "AdaBoost": {
        #     "estimator__max_depth": [1, 2],
        #     "n_estimators": [50, 100, 200],
        #     "learning_rate": [0.1, 0.5, 1.0],
        # },
        # "GradientBoosting": {
        #     "n_estimators": [100, 200],
        #     "learning_rate": [0.05, 0.1],
        #     "max_depth": [None, 1, 3, 5],
        # },
        # "XGBoost": {
        #     "n_estimators": [30, 50, 100, 200, 300],
        #     "learning_rate": [0.001, 0.05, 0.1, 0.2],
        #     "max_depth": [None, 1, 3, 5, 10],
        # },
        # "DecisionTree": {
        #     "max_depth": [None, 10, 20, 30],
        #     "min_samples_split": [2, 5, 10],
        # },
        # "Bagging": {
        #     "estimator__max_depth": [None, 10, 20],
        #     "n_estimators": [20, 50, 100],
        # },
        # "RandomForest": {
        #     "n_estimators": [100, 200, 200],
        #     "max_depth": [None, 5, 10, 20],
        #     "min_samples_split": [2, 5, 10],
        # },
        # "GaussianNB": {},
        # "KNeighbors": {
        #     "n_neighbors": [3, 5, 7, 10, 15],
        #     "weights": ["uniform", "distance"],
        # },
        # "LogisticRegression": {
        #     "C": [0.1, 1, 10, 100],
        #     "solver": ["liblinear", "lbfgs"],
        # },
    }

    return models, param_grids


def calculate_roc_metrics(y_true, y_pred, y_pred_proba=None):
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    g_mean = np.sqrt(tpr * tnr)

    auc_roc = None
    if y_pred_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Error")
            auc_roc = None

    return {
        "auc_roc": auc_roc,
        "tpr": tpr,
        "fpr": fpr,
        "g_mean": g_mean,
        "confusion_matrix": cm,
    }


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", show=False, save_path=None):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        if save_path:
            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(
                fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})"
            )
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (FPR)", color="black")
            ax.set_ylabel("True Positive Rate (TPR)", color="black")
            ax.set_title(title, color="black")

            leg = ax.legend(loc="lower right")
            for text in leg.get_texts():
                text.set_color("black")
            leg.get_frame().set_edgecolor("black")

            ax.grid(True)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"ROC curve saved to: {save_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    except Exception as e:
        print(f"Error")


def plot_confusion_matrix(
    y_true, y_pred, title="Confusion Matrix", show=False, save_path=None
):
    if save_path:
        plt.style.use("ggplot")

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Face", "Back"]
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

        ax.set_title(title, color="black")
        ax.set_xlabel("Predicted Label", color="black")
        ax.set_ylabel("True Label", color="black")

        ax.grid(False)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def format_metrics_dict(y_true, y_pred, y_pred_proba=None):
    metrics = {
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True
        ),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    if y_pred_proba is not None:
        if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
            proba_for_roc = y_pred_proba[:, 1]
        else:
            proba_for_roc = y_pred_proba
        roc_metrics = calculate_roc_metrics(y_true, y_pred, proba_for_roc)
        metrics.update(
            {
                "auc_roc": roc_metrics["auc_roc"],
                "tpr": roc_metrics["tpr"],
                "fpr": roc_metrics["fpr"],
                "g_mean": roc_metrics["g_mean"],
            }
        )
    else:
        # Use predictions for ROC metrics (less accurate but still useful)
        roc_metrics = calculate_roc_metrics(y_true, y_pred)
        metrics.update(
            {
                "auc_roc": roc_metrics["auc_roc"],
                "tpr": roc_metrics["tpr"],
                "fpr": roc_metrics["fpr"],
                "g_mean": roc_metrics["g_mean"],
            }
        )

    return metrics


def create_standard_result_structure(
    experiment_type,
    model_name,
    n_components,
    ratio,
    version,
    run_suffix,
    explained_variance_ratio=None,
    train_metrics=None,
    test_metrics=None,
    model_info=None,
    submission_paths=None,
    plot_paths=None,
    additional_info=None,
):
    result = {
        "experiment_info": {
            "type": experiment_type,
            "model_name": model_name,
            "n_components": n_components,
            "ratio": ratio,
            "version": version,
            "suffix": run_suffix,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "pca_info": {
            "n_components": n_components,
            "explained_variance_ratio": explained_variance_ratio,
        },
        "metrics": {
            "train": train_metrics if train_metrics else {},
            "test": test_metrics if test_metrics else {},
        },
        "model_info": model_info if model_info else {},
        "paths": {
            "submissions": submission_paths if submission_paths else {},
            "plots": plot_paths if plot_paths else {},
        },
        "additional_info": additional_info if additional_info else {},
    }

    return result


def generate_standard_plots(
    X_train_pca,
    y_train,
    explained_variance_ratio,
    cumulative_variance,
    plots_dir,
    run_suffix,
    n_components,
    ratio,
    version,
    show=False,
    save_results=True,
):
    """
    Generate standard PCA plots (variance, classes, 3D, pairplot).

    Parameters:
    -----------
    X_train_pca : array-like
        PCA-transformed training data
    y_train : array-like
        Training labels
    explained_variance_ratio : array-like
        Explained variance ratio from PCA
    cumulative_variance : array-like
        Cumulative variance from PCA
    plots_dir : str
        Directory to save plots
    run_suffix : str
        Suffix for plot filenames
    model_name : str
        Name of the model
    n_components : int
        Number of PCA components
    ratio : str
        Ratio parameter
    version : int
        Version number
    show : bool
        Whether to display plots
    save_results : bool
        Whether to save plots

    Returns:
    --------
    dict : Dictionary with paths to saved plots
    """
    plot_paths = {}

    if save_results:
        # Variance plot
        variance_path = f"{plots_dir}/{run_suffix}_variance_plot.png"
        plot_paths["variance"] = variance_path

        plot_variance(
            explained_variance_ratio,
            cumulative_variance,
            subtitle=format_plot_title_no_model(
                "Variance Plot", n_components, ratio, version
            ),
            show=show,
            save_path=variance_path,
        )
        print(f"Variance plot saved to: {variance_path}")

        # PCA classes plot
        classes_path = f"{plots_dir}/{run_suffix}_pca_classes.png"
        plot_paths["classes"] = classes_path

        plot_pca_classes(
            X_train_pca,
            y_train,
            subtitle=format_plot_title_no_model(
                "PCA Classes", n_components, ratio, version
            ),
            show=show,
            save_path=classes_path,
        )
        print(f"PCA classes plot saved to: {classes_path}")

        # 3D PCA plot
        if n_components >= 3:
            if save_results:
                pca_3d_path = f"{plots_dir}/{run_suffix}_pca_3d.png"
                plot_paths["3d"] = pca_3d_path
            else:
                pca_3d_path = None

            plot_3d_pca(
                X_train_pca,
                y_train,
                subtitle=format_plot_title_no_model(
                    "3D PCA", n_components, ratio, version
                ),
                show=show,
                save_path=pca_3d_path,
            )
            print(f"3D PCA plot saved to: {pca_3d_path}")

        # Pairplot
        if n_components >= 5:
            pairplot_path = f"{plots_dir}/{run_suffix}_pairplot.png"
            plot_paths["pairplot"] = pairplot_path

            plot_pairplot(
                X_train_pca,
                y_train,
                subtitle=format_plot_title_no_model(
                    "Pairplot", n_components, ratio, version
                ),
                show=show,
                save_path=pairplot_path,
            )
            print(f"Pairplot saved to: {pairplot_path}")

    return plot_paths


def setup_pca_transformation(X_train_std, n_components, show=True, whiten=True):
    pca = PCA(n_components=n_components, whiten=whiten)
    X_train_pca = pca.fit_transform(X_train_std)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"PCA with {n_components} components:")
    print(f"  Explained variance ratio: {explained_variance_ratio[:5]}...")
    print(f"  Cumulative variance: {cumulative_variance[-1]:.4f}")

    return pca, X_train_pca, explained_variance_ratio, cumulative_variance


def train_and_evaluate_model(
    model, X_train, y_train, X_test, y_test, model_name, show=True
):
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_proba = None
    y_test_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_train_pred_proba = model.predict_proba(X_train)
        y_test_pred_proba = model.predict_proba(X_test)

    train_metrics = format_metrics_dict(y_train, y_train_pred, y_train_pred_proba)
    test_metrics = format_metrics_dict(y_test, y_test_pred, y_test_pred_proba)

    print(f"  Train F1: {train_metrics['f1']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    if train_metrics["auc_roc"] is not None:
        print(f"  Train AUC-ROC: {train_metrics['auc_roc']:.4f}")
        print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")

    return model, train_metrics, test_metrics


def perform_grid_search(model, param_grid, X_train, y_train, model_name, show=True):
    print(f"Performing grid search for {model_name}...")

    cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splitter,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def perform_grid_search_stratified_shuffle(
    model, param_grid, X_train, y_train, model_name, show=True
):
    print(f"Performing grid search for {model_name}...")

    cv_splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splitter,
        n_jobs=1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def perform_grid_search_stratified_k_folds(
    model, param_grid, X_train, y_train, model_name, show=True
):
    print(f"Performing grid search for {model_name}...")

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splitter,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def perform_randomized_search(
    model, param_grid, X_train, y_train, model_name, show=True
):
    print(f"Performing randomized search for {model_name}...")

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        scoring="f1",
        cv=cv_splitter,
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    randomized_search.fit(X_train, y_train)

    print(f"  Best score: {randomized_search.best_score_:.4f}")
    print(f"  Best params: {randomized_search.best_params_}")

    return (
        randomized_search.best_estimator_,
        randomized_search.best_params_,
        randomized_search.best_score_,
    )


def ensure_directories(run_name):
    run_dir = f"./results/{run_name}"
    metrics_dir = f"{run_dir}/metrics"
    plots_dir = f"{run_dir}/plots"
    submissions_dir = f"{run_dir}/submissions"

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    return metrics_dir, plots_dir, submissions_dir, run_dir


def generate_kaggle_submission(
    models,
    scaler,
    pca,
    submissions_dir,
    run_suffix,
    zip_path="/content/Test.zip",
    patches_path="content/test_patches",
    show=True,
    single_model_mode=False,
    hog_changes=False,
):
    try:
        if not isinstance(models, dict):
            model_name = type(models).__name__
            models = {model_name: models}

        print(f"Loading Kaggle test data from {zip_path}...")

        kaggle_patches, kaggle_filenames = get_patches(
            zip_path=zip_path,
            patches_path=patches_path,
            return_filenames=True,
        )

        pgm_kaggle_files_id = [filename[5:-4] for filename in kaggle_filenames]

        print("Building HOG features for Kaggle data...")

        if hog_changes:
            hog_params = {
                "orientations": 9,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
                "visualize": False,
                "transform_sqrt": True,
                "feature_vector": True,
            }

            X_kag = np.array(
                [
                    feature.hog(im, **hog_params)
                    for im in tqdm(
                        kaggle_patches, desc="Building HOG features", disable=not show
                    )
                ]
            )

        else:
            X_kag = np.array(
                [
                    feature.hog(im)
                    for im in tqdm(
                        kaggle_patches, desc="Building HOG features", disable=not show
                    )
                ]
            )

        X_kag_std = scaler.transform(X_kag)
        X_pca_kag = pca.transform(X_kag_std)

        submission_paths = {}

        print(f"Generating submissions for {len(models)} models...")

        for model_name, model in models.items():
            print(f"  Generating submission for {model_name}...")

            y_kaggle_predictions = model.predict(X_pca_kag)

            y_kaggle_dict = {
                pgm_kaggle_files_id[i]: y_kaggle_predictions[i]
                for i in range(len(pgm_kaggle_files_id))
            }

            kaggle_hat = pd.DataFrame(
                list(y_kaggle_dict.items()), columns=["id", "target_feature"]
            )

            kaggle_hat["id"] = kaggle_hat["id"].astype(str)
            kaggle_hat["target_feature"] = kaggle_hat["target_feature"].astype(int)
            kaggle_hat.sort_values(by="id", inplace=True)

            if single_model_mode or len(models) == 1:
                submission_path = f"{submissions_dir}/submission_{run_suffix}.csv"
            else:
                submission_path = (
                    f"{submissions_dir}/submission_{run_suffix}_{model_name}.csv"
                )

            os.makedirs(os.path.dirname(submission_path), exist_ok=True)
            kaggle_hat.to_csv(submission_path, index=False)
            submission_paths[model_name] = submission_path

            if len(models) == 1:
                print(f"Submission saved: {submission_path}")
            else:
                print(f"  Saved: {submission_path}")

        if show and len(models) == 1:
            print("\nKaggle submission preview:")
            print(kaggle_hat.head())

        return {
            "submission_paths": submission_paths,
            "data_shape": X_kag.shape,
            "num_predictions": len(pgm_kaggle_files_id),
            "models_processed": list(models.keys()),
        }

    except Exception as e:
        print(f"Warning: Could not generate Kaggle submissions: {e}")
        return {"submission_paths": {}, "error": str(e), "models_processed": []}


def format_plot_title_no_model(
    plot_type, n_components, ratio, version, dataset_type=""
):
    title_parts = []
    if dataset_type:
        title_parts.append(dataset_type)
    title_parts.append(plot_type)
    title_parts.append(f"Ratio: {ratio}")
    title_parts.append(f"v{version}")
    title_parts.append(f"{n_components} PCA Components")

    return " - ".join(title_parts)


def format_plot_title(
    plot_type, model_name, n_components, ratio, version, dataset_type=""
):
    title_parts = []
    if dataset_type:
        title_parts.append(dataset_type)
    title_parts.append(plot_type)
    title_parts.append(f"Ratio: {ratio}")
    title_parts.append(f"v{version}")
    title_parts.append(f"{model_name}")
    title_parts.append(f"{n_components} PCA Components")

    return " - ".join(title_parts)


def save_metrics_to_files(metrics_dict, base_path, suffix=""):
    try:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        suffix_str = f"_{suffix}" if suffix else ""

        json_path = f"{base_path}{suffix_str}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2, default=str)

        print(f"Metrics saved to: {json_path}")

        return json_path

    except Exception as e:
        print(f"Error saving metrics to files: {e}")
        return None


def generate_final_submission(
    X_full,
    y_full,
    scaler,
    n_components,
    ratio,
    version=1,
    show=True,
    save_results=False,
    hog_changes=False
):
    models = {
        "MLPClassifier": MLPClassifier(
            random_state=0,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.3,
            activation="relu",
            alpha=0.001,
            hidden_layer_sizes=(100,),
            learning_rate_init=0.01,
            max_iter=1000,
            solver="adam",
        ),
        # "XGBGBoost": XGBClassifier(
        #     n_estimators=300, max_depth=3, learning_rate=0.2, eval_metric="logloss", random_state=0, n_jobs=-1, verbosity=0
        # )
    }

    model_name = list(models.keys())[0]
    suffix = f"{ratio}_v{version}_{n_components}_{model_name}_FULL"

    print(f"\n{'=' * 60}")
    print("GENERATING FINAL SUBMISSION WITH FULL DATASET")
    print(f"n_components = {n_components}")
    print(f"Full dataset size: {X_full.shape[0]} samples")
    print(f"Models: {list(models.keys())}")
    print(f"Suffix: {suffix}")
    print(f"{'=' * 60}")

    if save_results:
        run_name = f"final_submission_{ratio}_v{version}"
        metrics_dir, plots_dir, submissions_dir, _ = ensure_directories(run_name)

        run_suffix = f"{ratio}_v{version}_{n_components}_FULL"

    X_full_std = scaler.fit_transform(X_full)

    pca, X_full_pca, explained_var, acumulated_var = setup_pca_transformation(
        X_full_std, n_components, show=show, whiten=True
    )

    if save_results:
        suffix = f"{ratio}_v{version}_{n_components}"
        generate_standard_plots(
            X_full_pca,
            y_full,
            explained_var,
            acumulated_var,
            plots_dir,
            suffix,
            n_components,
            ratio,
            version,
            show=show,
            save_results=save_results,
        )

    trained_models = {}
    model_predictions = {}
    model_probabilities = {}

    for model_name, model in tqdm(models.items(), desc="Training Models", unit="model"):
        model.fit(X_full_pca, y_full)
        trained_models[model_name] = model

        y_pred = model.predict(X_full_pca)
        y_pred_proba = model.predict_proba(X_full_pca)

        model_predictions[model_name] = y_pred
        model_probabilities[model_name] = y_pred_proba

    first_model_name = list(models.keys())[0]
    y_full_pred = model_predictions[first_model_name]
    y_full_pred_proba = model_probabilities[first_model_name]
    run_model_name = first_model_name if len(models) == 1 else f"{first_model_name}"

    full_report = classification_report(y_full, y_full_pred, output_dict=True)
    print(classification_report(y_full, y_full_pred))

    full_roc_metrics = calculate_roc_metrics(
        y_full,
        y_full_pred,
        y_full_pred_proba[:, 1]
        if y_full_pred_proba.shape[1] == 2
        else y_full_pred_proba,
    )

    final_cm_save_path = (
        f"{plots_dir}/{run_suffix}_confusion_matrix.png" if save_results else None
    )
    plot_confusion_matrix(
        y_full,
        y_full_pred,
        title=format_plot_title(
            "Confusion Matrix",
            run_model_name,
            n_components,
            ratio,
            version,
            "Final Model",
        ),
        show=show,
        save_path=final_cm_save_path,
    )

    if y_full_pred_proba.shape[1] == 2:
        final_roc_save_path = (
            f"{plots_dir}/{run_suffix}_roc_curve.png" if save_results else None
        )
        plot_roc_curve(
            y_full,
            y_full_pred_proba[:, 1],
            title=format_plot_title(
                "ROC Curve",
                run_model_name,
                n_components,
                ratio,
                version,
                "Final Model",
            ),
            show=show,
            save_path=final_roc_save_path,
        )

    kaggle_suffix_final = f"{ratio}_v{version}_{n_components}_FULL"
    kaggle_results_final = generate_kaggle_submission(
        models=trained_models,
        scaler=scaler,
        pca=pca,
        submissions_dir=submissions_dir,
        run_suffix=kaggle_suffix_final,
        show=show,
        single_model_mode=True,
        hog_changes=hog_changes
    )

    individual_submissions = kaggle_results_final["submission_paths"]

    submission_path = (
        list(individual_submissions.values())[0] if individual_submissions else None
    )

    result = {
        "n_components": n_components,
        "suffix": suffix,
        "run_model_name": run_model_name,
        "num_models": len(models),
        "model_names": list(models.keys()),
        "individual_submissions": individual_submissions,
        "explained_variance_ratio": explained_var.sum(),
        "full_dataset_f1": full_report["macro avg"]["f1-score"],
        "full_dataset_auc_roc": full_roc_metrics["auc_roc"],
        "full_dataset_tpr": full_roc_metrics["tpr"],
        "full_dataset_fpr": full_roc_metrics["fpr"],
        "full_dataset_g_mean": full_roc_metrics["g_mean"],
        "submission_path": submission_path,
        "dataset_size": X_full.shape[0],
    }

    print("\nFINAL SUBMISSION SUMMARY:")
    print(f"Components: {result['n_components']}")
    print(f"Models used: {result['model_names']}")
    print(f"Dataset size: {result['dataset_size']}")
    print(f"Explained Variance: {result['explained_variance_ratio']:.4f}")
    print(f"F1 Score: {result['full_dataset_f1']:.4f}")
    if result["full_dataset_auc_roc"] is not None:
        print(f"AUC-ROC: {result['full_dataset_auc_roc']:.4f}")
    print(f"TPR (True Positive Rate): {result['full_dataset_tpr']:.4f}")
    print(f"FPR (False Positive Rate): {result['full_dataset_fpr']:.4f}")
    print(f"G-Mean: {result['full_dataset_g_mean']:.4f}")

    if result["individual_submissions"]:
        print("\nIndividual Model Submissions:")
        for model_name, path in result["individual_submissions"].items():
            print(f"  {model_name}: {path}")

    if submission_path:
        print(f"\nMain submission: {submission_path}")

    if save_results:
        save_metrics_to_files(
            result, f"{metrics_dir}/final_submission_{run_suffix}", "final"
        )

    return result


def run_single_model_grid_search(
    X_train_pca,
    y_train,
    X_test,
    y_test,
    scaler,
    pca,
    n_components,
    ratio,
    version,
    model_name,
    show=False,
    save_results=True,
    base_run_dir=None,
    hog_changes=False
):
    all_models, all_param_grids = get_models_and_param_grids()

    if model_name not in all_models:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(all_models.keys())}"
        )

    model = all_models[model_name]
    param_grid = all_param_grids[model_name]

    del all_models, all_param_grids
    gc.collect()

    print(f"\n{'=' * 60}")
    print(f"SEARCH FOR {model_name}")
    print(f"n_components: {n_components}")
    print(
        f"Parameter combinations: {len(list(param_grid.values())[0]) if param_grid else 1}"
    )
    print(f"{'=' * 60}")

    if save_results:
        if base_run_dir:
            model_run_dir = f"{base_run_dir}/{model_name}"
            metrics_dir = f"{model_run_dir}/metrics"
            plots_dir = f"{model_run_dir}/plots"
            submissions_dir = f"{model_run_dir}/submissions"

            os.makedirs(model_run_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(submissions_dir, exist_ok=True)
        else:
            run_name = f"single_model_{ratio}_v{version}_{n_components}_{model_name}"
            metrics_dir, plots_dir, submissions_dir, _ = ensure_directories(run_name)

        suffix = f"{ratio}_v{version}_{n_components}_{model_name}"
        base_save_path = f"{metrics_dir}/single_model_{suffix}"

    best_model, best_params, best_score = perform_grid_search_stratified_shuffle(
        model, param_grid, X_train_pca, y_train, model_name, show=show
    )
    # best_model, best_params, best_score = perform_grid_search(
    #     model, param_grid, X_train_pca, y_train, model_name, show=show
    # )
    # best_model, best_params, best_score = perform_grid_search_stratified_k_folds(
    #     model, param_grid, X_train_pca, y_train, model_name, show=show
    # )
    # best_model, best_params, best_score = perform_randomized_search(
    #     model, param_grid, X_train_pca, y_train, model_name, show=show
    # )

    joblib.dump(
        best_model,
        f"{metrics_dir}/{suffix}_best_model.joblib",
        compress=3,
    )

    gc.collect()

    if save_results:
        explained_var = pca.explained_variance_ratio_

    X_test_transformed = scaler.transform(X_test)
    X_test_transformed = pca.transform(X_test_transformed)

    trained_model, train_metrics, test_metrics = train_and_evaluate_model(
        best_model,
        X_train_pca,
        y_train,
        X_test_transformed,
        y_test,
        model_name,
        show=show,
    )

    gc.collect()

    if save_results:
        train_cm_save_path = f"{plots_dir}/{suffix}_train_confusion_matrix.png"
        plot_confusion_matrix(
            y_train,
            train_metrics["y_pred"],
            title=format_plot_title(
                "Confusion Matrix", model_name, n_components, ratio, version, "Training"
            ),
            show=show,
            save_path=train_cm_save_path,
        )

        if (
            train_metrics["y_pred_proba"] is not None
            and train_metrics["y_pred_proba"].shape[1] == 2
        ):
            train_roc_save_path = f"{plots_dir}/{suffix}_train_roc_curve.png"
            plot_roc_curve(
                y_train,
                train_metrics["y_pred_proba"][:, 1],
                title=format_plot_title(
                    "ROC Curve", model_name, n_components, ratio, version, "Training"
                ),
                show=show,
                save_path=train_roc_save_path,
            )

        test_cm_save_path = (
            f"{plots_dir}/{suffix}_test_confusion_matrix.png" if save_results else None
        )
        plot_confusion_matrix(
            y_test,
            test_metrics["y_pred"],
            title=format_plot_title(
                "Confusion Matrix", model_name, n_components, ratio, version, "Test"
            ),
            show=show,
            save_path=test_cm_save_path,
        )

        if (
            test_metrics["y_pred_proba"] is not None
            and test_metrics["y_pred_proba"].shape[1] == 2
        ):
            test_roc_save_path = (
                f"{plots_dir}/{suffix}_test_roc_curve.png" if save_results else None
            )
            plot_roc_curve(
                y_test,
                test_metrics["y_pred_proba"][:, 1],
                title=format_plot_title(
                    "ROC Curve", model_name, n_components, ratio, version, "Test"
                ),
                show=show,
                save_path=test_roc_save_path,
            )

        gc.collect()

    if save_results:
        kaggle_suffix = f"{ratio}_v{version}_{n_components}_{model_name}_SingleGrid"
        kaggle_results = generate_kaggle_submission(
            models=best_model,
            scaler=scaler,
            pca=pca,
            submissions_dir=submissions_dir,
            run_suffix=kaggle_suffix,
            show=show,
            single_model_mode=True,
            hog_changes=hog_changes
        )

        submission_path = kaggle_results["submission_paths"].get(
            type(best_model).__name__
        )

        gc.collect()
    else:
        submission_path = None

    result = create_standard_result_structure(
        experiment_type="single_model_grid_search",
        model_name=model_name,
        n_components=n_components,
        ratio=ratio,
        version=version,
        run_suffix=suffix,
        explained_variance_ratio=explained_var.sum(),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_info={
            "best_params": best_params,
            "best_cv_score": best_score,
            "trained_model": best_model,
        },
        submission_paths={"main": submission_path} if submission_path else None,
    )

    if save_results:
        save_metrics_to_files(result, base_save_path)

    return result


def run_all_models_separately(
    X_train_std,
    y_train,
    X_test,
    y_test,
    scaler,
    n_components_list,
    ratio,
    version=1,
    model_names=None,
    show=False,
    save_results=True,
    hog_changes=False
):
    all_models, _ = get_models_and_param_grids()
    if model_names is None:
        model_names = list(all_models.keys())

    print(f"Running separate grid searches for models: {model_names}")
    print(f"PCA components to test: {n_components_list}")

    if save_results:
        base_run_name = f"all_models_{ratio}_v{version}_StratifiedShuffleSplit"
        base_run_dir = f"./results/{base_run_name}"
        os.makedirs(base_run_dir, exist_ok=True)

    all_results = []
    model_results = {model_name: [] for model_name in model_names}

    generated_plots_tracker = set()

    for n_components in tqdm(
        n_components_list, desc="PCA Components", unit="components"
    ):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING n_components = {n_components}")
        print(f"{'=' * 80}")

        pca, X_train_pca, _, _ = setup_pca_transformation(
            X_train_std, n_components, show=show, whiten=True
        )

        for model_name in tqdm(
            model_names,
            desc=f"Models (n_components={n_components})",
            unit="model",
            leave=False,
        ):
            try:
                result = run_single_model_grid_search(
                    X_train_pca=X_train_pca,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    scaler=scaler,
                    pca=pca,
                    n_components=n_components,
                    ratio=ratio,
                    version=version,
                    model_name=model_name,
                    show=False,
                    save_results=save_results,
                    base_run_dir=base_run_dir if save_results else None,
                )
                all_results.append(result)
                model_results[model_name].append(result)

            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue

    flattened_results = []
    for result in all_results:
        flat_result = {
            "model_name": result["experiment_info"]["model_name"],
            "n_components": result["experiment_info"]["n_components"],
            "ratio": result["experiment_info"]["ratio"],
            "version": result["experiment_info"]["version"],
            "explained_variance": result["pca_info"]["explained_variance_ratio"],
            "train_f1": result["metrics"]["train"].get("f1", None),
            "train_auc_roc": result["metrics"]["train"].get("auc_roc", None),
            "train_g_mean": result["metrics"]["train"].get("g_mean", None),
            "test_f1": result["metrics"]["test"].get("f1", None),
            "test_auc_roc": result["metrics"]["test"].get("auc_roc", None),
            "test_g_mean": result["metrics"]["test"].get("g_mean", None),
        }
        flattened_results.append(flat_result)

    results_df = pd.DataFrame(flattened_results)
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'=' * 80}")

    display_columns = [
        "model_name",
        "n_components",
        "test_f1",
        "test_auc_roc",
        "test_g_mean",
    ]
    available_columns = [col for col in display_columns if col in results_df.columns]
    if available_columns:
        print(results_df[available_columns].to_string(index=False))
    else:
        print("No results to display")

    if "test_f1" in results_df.columns and not results_df["test_f1"].isna().all():
        best_idx = results_df["test_f1"].idxmax()
        best_result = results_df.iloc[best_idx]
        print(
            f"\nBest model: {best_result['model_name']} with {best_result['n_components']} components"
        )
        print(f"Best test F1: {best_result['test_f1']:.4f}")
        if best_result["test_auc_roc"] is not None:
            print(f"Best test AUC-ROC: {best_result['test_auc_roc']:.4f}")
        if best_result["test_g_mean"] is not None:
            print(f"Best test G-Mean: {best_result['test_g_mean']:.4f}")

        print("\nComprehensive metrics for best model:")
        train_f1 = best_result.get("train_f1", "N/A")
        train_auc_roc = best_result.get("train_auc_roc", "N/A")
        train_g_mean = best_result.get("train_g_mean", "N/A")

        print(
            f"Training F1: {train_f1:.4f}"
            if train_f1 != "N/A" and train_f1 is not None
            else "Training F1: N/A"
        )
        print(
            f"Training AUC-ROC: {train_auc_roc:.4f}"
            if train_auc_roc != "N/A" and train_auc_roc is not None
            else "Training AUC-ROC: N/A"
        )
        print(
            f"Training G-Mean: {train_g_mean:.4f}"
            if train_g_mean != "N/A" and train_g_mean is not None
            else "Training G-Mean: N/A"
        )
        print(f"Test F1: {best_result['test_f1']:.4f}")
        print(
            f"Test AUC-ROC: {best_result['test_auc_roc']:.4f}"
            if best_result["test_auc_roc"] is not None
            else "Test AUC-ROC: N/A"
        )
        if best_result["test_g_mean"] is not None:
            print(f"Test G-Mean: {best_result['test_g_mean']:.4f}")
        else:
            print("Test G-Mean: N/A")

    if save_results and flattened_results:
        for model_name in model_names:
            model_specific_results = [
                r for r in flattened_results if r["model_name"] == model_name
            ]
            if model_specific_results:
                model_dir = f"{base_run_dir}/{model_name}"
                model_metrics_dir = f"{model_dir}/metrics"
                os.makedirs(model_metrics_dir, exist_ok=True)

                model_results_df = pd.DataFrame(model_specific_results)

                if (
                    "test_f1" in model_results_df.columns
                    and not model_results_df["test_f1"].isna().all()
                ):
                    best_model_idx = model_results_df["test_f1"].idxmax()
                    best_model_result = model_results_df.iloc[best_model_idx]

                    model_best_info = {
                        "model_name": model_name,
                        "n_components": int(best_model_result["n_components"]),
                        "test_f1": float(best_model_result["test_f1"]),
                        "test_auc_roc": float(best_model_result["test_auc_roc"])
                        if best_model_result["test_auc_roc"] is not None
                        else None,
                        "test_g_mean": float(best_model_result["test_g_mean"])
                        if best_model_result["test_g_mean"] is not None
                        else None,
                        "train_f1": float(best_model_result["train_f1"])
                        if best_model_result["train_f1"] is not None
                        else None,
                        "train_auc_roc": float(best_model_result["train_auc_roc"])
                        if best_model_result["train_auc_roc"] is not None
                        else None,
                        "train_g_mean": float(best_model_result["train_g_mean"])
                        if best_model_result["train_g_mean"] is not None
                        else None,
                    }
                else:
                    model_best_info = {}

                model_summary_data = {
                    "experiment_summary": {
                        "experiment_type": f"single_model_all_components_{model_name}",
                        "model_name": model_name,
                        "ratio": ratio,
                        "version": version,
                        "n_components_tested": n_components_list,
                        "total_experiments": len(model_specific_results),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    },
                    "results_table": model_results_df[available_columns].to_dict(
                        "records"
                    )
                    if available_columns
                    else [],
                    "best_model_info": model_best_info,
                    "all_results_summary": model_specific_results,
                }

                model_summary_path = f"{model_metrics_dir}/summary_all_results_{model_name}_{ratio}_v{version}.json"
                try:
                    import json

                    with open(model_summary_path, "w") as f:
                        json.dump(model_summary_data, f, indent=2, default=str)
                    print(
                        f"Individual summary saved for {model_name}: {model_summary_path}"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not save individual summary for {model_name}: {e}"
                    )

                model_final_summary = {
                    "experiment_type": f"final_models_{model_name}_individual",
                    "model_name": model_name,
                    "ratio": ratio,
                    "version": version,
                    "n_components_tested": n_components_list,
                    "total_experiments": len(model_specific_results),
                    "dataset_size": X_train_std.shape[0],
                    "best_model": model_name,
                    "best_n_components": model_best_info.get("n_components", "N/A"),
                    "best_test_f1": model_best_info.get("test_f1", "N/A"),
                    "best_test_auc_roc": model_best_info.get("test_auc_roc", "N/A"),
                    "best_test_g_mean": model_best_info.get("test_g_mean", "N/A"),
                    "all_results": model_results[model_name],
                }

                model_final_summary_path = f"{model_metrics_dir}/final_models_summary_{model_name}_{ratio}_v{version}"
                save_metrics_to_files(
                    model_final_summary, model_final_summary_path, "summary"
                )
                print(
                    f"Final summary saved for {model_name}: {model_final_summary_path}"
                )

        global_summary_data = {
            "experiment_summary": {
                "experiment_type": "all_models_separately_global",
                "ratio": ratio,
                "version": version,
                "n_components_tested": n_components_list,
                "models_tested": model_names,
                "total_experiments": len(all_results),
                "timestamp": pd.Timestamp.now().isoformat(),
            },
            "results_table": results_df[available_columns].to_dict("records")
            if available_columns
            else [],
            "best_model_info": {},
            "all_results_summary": flattened_results,
        }

        if "test_f1" in results_df.columns and not results_df["test_f1"].isna().all():
            global_summary_data["best_model_info"] = {
                "model_name": best_result["model_name"],
                "n_components": int(best_result["n_components"]),
                "test_f1": float(best_result["test_f1"]),
                "test_auc_roc": float(best_result["test_auc_roc"])
                if best_result["test_auc_roc"] is not None
                else None,
                "test_g_mean": float(best_result["test_g_mean"])
                if best_result["test_g_mean"] is not None
                else None,
                "train_f1": float(train_f1)
                if train_f1 != "N/A" and train_f1 is not None
                else None,
                "train_auc_roc": float(train_auc_roc)
                if train_auc_roc != "N/A" and train_auc_roc is not None
                else None,
                "train_g_mean": float(train_g_mean)
                if train_g_mean != "N/A" and train_g_mean is not None
                else None,
            }

        global_summary_path = (
            f"{base_run_dir}/summary_all_results_{ratio}_v{version}.json"
        )
        try:
            import json

            with open(global_summary_path, "w") as f:
                json.dump(global_summary_data, f, indent=2, default=str)
            print(f"\nGlobal summary saved to JSON: {global_summary_path}")
        except Exception as e:
            print(f"Warning: Could not save global summary JSON: {e}")

        global_final_summary = {
            "experiment_type": "final_models_all_global",
            "ratio": ratio,
            "version": version,
            "n_components_tested": n_components_list,
            "models_tested": model_names,
            "total_experiments": len(all_results),
            "dataset_size": X_train_std.shape[0],
            "best_model": best_result.get("model_name", "N/A")
            if "test_f1" in results_df.columns
            and not results_df["test_f1"].isna().all()
            else "N/A",
            "best_n_components": best_result.get("n_components", "N/A")
            if "test_f1" in results_df.columns
            and not results_df["test_f1"].isna().all()
            else "N/A",
            "best_test_f1": best_result.get("test_f1", "N/A")
            if "test_f1" in results_df.columns
            and not results_df["test_f1"].isna().all()
            else "N/A",
            "best_test_auc_roc": best_result.get("test_auc_roc", "N/A")
            if "test_f1" in results_df.columns
            and not results_df["test_f1"].isna().all()
            else "N/A",
            "best_test_g_mean": best_result.get("test_g_mean", "N/A")
            if "test_f1" in results_df.columns
            and not results_df["test_f1"].isna().all()
            else "N/A",
            "all_results": all_results,
        }

        global_final_summary_path = (
            f"{base_run_dir}/final_models_summary_{ratio}_v{version}"
        )
        save_metrics_to_files(
            global_final_summary, global_final_summary_path, "summary"
        )
        print(f"Global final summary saved to: {global_final_summary_path}")

        if "test_f1" in results_df.columns and not results_df["test_f1"].isna().all():
            best_idx = results_df["test_f1"].idxmax()
            best_result = results_df.iloc[best_idx]
            best_model_name = best_result["model_name"]
            best_n_components = int(best_result["n_components"])

            print(
                f"\nBest model: {best_model_name} with {best_n_components} components"
            )

            best_model_instance = None
            for result in all_results:
                if (
                    result["experiment_info"]["model_name"] == best_model_name
                    and result["experiment_info"]["n_components"] == best_n_components
                ):
                    best_model_instance = result["model_info"]["trained_model"]
                    break

            if best_model_instance:
                if len(n_components_list) == 1 and len(model_names) == 1:
                    scaler_path = f"{base_run_dir}/scaler_{ratio}_v{version}.joblib"
                    pca_path = f"{base_run_dir}/pca_{best_n_components}_{ratio}_v{version}.joblib"
                    model_path = f"{base_run_dir}/best_model_{best_model_name}_{best_n_components}_{ratio}_v{version}.joblib"

                    joblib.dump(scaler, scaler_path)
                    print(f"Scaler saved to: {scaler_path}")

                    single_pca = PCA(n_components=best_n_components, whiten=True)
                    single_pca.fit(X_train_std)
                    joblib.dump(single_pca, pca_path)
                    print(f"PCA saved to: {pca_path}")

                    joblib.dump(best_model_instance, model_path)
                    print(f"Best model saved to: {model_path}")
                else:
                    print(
                        f"Skipping artifact saving - tested {len(model_names)} models and {len(n_components_list)} components"
                    )
            else:
                print("Warning: Could not find best model instance to save")

    return all_results


def generate_pca_plots_once(
    X_train_pca,
    y_train,
    explained_variance_ratio,
    cumulative_variance,
    plots_dir,
    n_components,
    ratio,
    version,
    show=False,
    save_results=True,
    generated_plots_tracker=None,
):
    if generated_plots_tracker is None:
        generated_plots_tracker = set()

    pca_config_key = (n_components, ratio, version)

    if pca_config_key in generated_plots_tracker:
        print(f"PCA plots for {n_components} components already generated, skipping...")
        return {}

    plot_paths = {}

    if save_results:
        pca_suffix = f"{ratio}_v{version}_{n_components}_PCA"

        variance_path = f"{plots_dir}/{pca_suffix}_variance_plot.png"
        plot_paths["variance"] = variance_path

        plot_variance(
            explained_variance_ratio,
            cumulative_variance,
            subtitle=format_plot_title_no_model(
                "Variance Plot", n_components, ratio, version
            ),
            show=show,
            save_path=variance_path,
        )
        print(f"PCA variance plot saved to: {variance_path}")

        classes_path = f"{plots_dir}/{pca_suffix}_pca_classes.png"
        plot_paths["classes"] = classes_path

        plot_pca_classes(
            X_train_pca,
            y_train,
            subtitle=format_plot_title_no_model(
                "PCA Classes", n_components, ratio, version
            ),
            show=show,
            save_path=classes_path,
        )
        if save_results and show:
            print(f"PCA classes plot saved to: {classes_path}")

        if n_components >= 3:
            pca_3d_path = f"{plots_dir}/{pca_suffix}_pca_3d.png"
            plot_paths["3d"] = pca_3d_path

            plot_3d_pca(
                X_train_pca,
                y_train,
                subtitle=format_plot_title_no_model(
                    "3D PCA", n_components, ratio, version
                ),
                show=show,
                save_path=pca_3d_path,
            )
            if save_results and show:
                print(f"PCA 3D plot saved to: {pca_3d_path}")

        if n_components >= 5:
            pairplot_path = f"{plots_dir}/{pca_suffix}_pairplot.png"
            plot_paths["pairplot"] = pairplot_path

            plot_pairplot(
                X_train_pca,
                y_train,
                subtitle=format_plot_title_no_model(
                    "Pairplot", n_components, ratio, version
                ),
                show=show,
                save_path=pairplot_path,
            )
            if save_results and show:
                print(f"PCA pairplot saved to: {pairplot_path}")

    generated_plots_tracker.add(pca_config_key)

    return plot_paths
