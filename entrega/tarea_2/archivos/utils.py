import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
import numpy as np
import zipfile
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")

def get_patches(zip_path, patches_path, return_filenames=False):
    suffix = ".pgm"

    if not os.path.exists(patches_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
                zip_ref.extract(file, patches_path)

    all_files = os.listdir(patches_path)
    pgm_files = [filename for filename in all_files if filename.endswith(suffix)]

    images = []
    for filename in tqdm(pgm_files, desc="Loading images"):
        img_path = os.path.join(patches_path, filename)
        with open(img_path, "rb") as pgmf:
            image = plt.imread(pgmf)
        images.append(image)

    patches = np.array(images)

    if return_filenames:
        return patches, pgm_files
    else:
        return patches


def view_sample_images(images, count=15, figsize=(5, 3)):
    # Visualizamos una muestra
    samples = sample(range(images.shape[0]), count)
    fig, ax = plt.subplots(nrows=figsize[1], ncols=figsize[0], figsize=figsize)
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[samples[i]], cmap="gray")
        axi.axis("off")
    plt.tight_layout()
    plt.show()


def plot_variance(
    explained_variance, acumulated_variance, subtitle=None, show=False, save_path=None
):
    
    print("Tarea 2")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Varianza explicada
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance, color="skyblue")
    ax1.set_title("Varianza explicada por componente", color="black")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Proporción de varianza")
    ax1.grid(True)

    # Varianza acumulada
    ax2.bar(
        range(1, len(acumulated_variance) + 1), acumulated_variance, color="lightgreen"
    )
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95%")
    ax2.set_title("Varianza acumulada", color="black")
    ax2.set_xlabel("Número de componentes")
    ax2.set_ylabel("Proporción acumulada de varianza")
    leg = ax2.legend()
    for text in leg.get_texts():
        text.set_color("black")
    leg.get_frame().set_edgecolor("black")
    ax2.grid(True)

    if subtitle:
        plt.suptitle(subtitle, color="black")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pca_classes(X_train_pca, y_train, subtitle=None, show=False, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Gráfico 2D de PCA con clases
    colors = ["tab:red", "tab:blue"]
    labels = ["Back", "Face"]

    for class_value in [0, 1]:
        ax.scatter(
            X_train_pca[y_train == class_value, 0],
            X_train_pca[y_train == class_value, 1],
            alpha=0.4,
            label=labels[class_value],
            c=colors[class_value],
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA con clases", color="black")
    if subtitle:
        fig.suptitle(subtitle, color="black")
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_color("black")
    leg.get_frame().set_edgecolor("black")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_pca(X_train_pca, y_train, subtitle=None, show=False, save_path=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["tab:red", "tab:blue"]
    labels = ["Back", "Face"]

    for class_value in [0, 1]:
        ax.scatter(
            X_train_pca[y_train == class_value, 0],
            X_train_pca[y_train == class_value, 1],
            X_train_pca[y_train == class_value, 2],
            alpha=0.7,
            label=labels[class_value],
            color=colors[class_value],
        )

    ax.view_init(elev=15, azim=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    pc3_center = (X_train_pca[:, 2].max() + X_train_pca[:, 2].min()) / 2

    pc3_x = X_train_pca[:, 0].max()
    pc3_y = X_train_pca[:, 1].min()
    pc3_z = pc3_center

    default_label_color = mpl.rcParams["axes.labelcolor"]
    ax.text(pc3_x, pc3_y, pc3_z, "PC3", fontsize=12, color=default_label_color)
    ax.set_title("PCA 3D Visualization - First 3 Components", color="black")
    if subtitle:
        fig.suptitle(subtitle, color="black")
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_color("black")
    leg.get_frame().set_edgecolor("black")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pairplot(X_train_pca, y_train, subtitle=None, show=False, save_path=None):
    plt.figure(figsize=(12, 10))

    df = pd.DataFrame(X_train_pca[:, :5], columns=[f"PC{i + 1}" for i in range(5)])

    df["label"] = y_train
    df["label"] = df["label"].map({0: "Back", 1: "Face"})

    palette = {
        "Back": "#e24a33",
        "Face": "#348abd",
    }

    g = sns.pairplot(
        df,
        vars=["PC1", "PC2", "PC3", "PC4", "PC5"],
        hue="label",
        palette=palette,
        diag_kind="kde",
        height=2,
        plot_kws={"alpha": 0.4},
    )

    if not g._legend:
        g.add_legend()

    leg = g._legend
    if leg:
        for text in leg.get_texts():
            text.set_color("black")
        leg.get_frame().set_edgecolor("black")

    if subtitle:
        g.fig.suptitle(subtitle, color="black", y=1.02)

    if save_path:
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(g.fig)
