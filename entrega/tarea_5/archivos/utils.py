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

# def load_patches_from_zip(zip_path, to_float=True):
#     """
#     Lee los .pgm de `zip_path` y devuelve un array (N, 64, 64).

#     Args:
#         zip_path (str): ruta del archivo .zip
#         to_float (bool): si True, convierte a float32
#     """
#     with zipfile.ZipFile(zip_path) as z:
#         # Filtramos solo archivos .pgm y preservamos orden alfabético
#         pgm_files = sorted([f for f in z.namelist() if f.endswith(".pgm")])

#         patches = []
#         for fname in tqdm(pgm_files, desc="Leyendo parches"):
#             with z.open(fname) as f:
#                 patch = iio.imread(f)          # uint8 0-255
#                 if to_float:
#                     patch = patch.astype(np.float32)
#                 patches.append(patch)

#     return np.stack(patches)


# Función para extraer porciones de una imagen
# def extract_patches(img, N, scale=1.0, patch_size=(64, 64)):
#     # Calcula el tamaño del parche extraído basado en el factor de escala dado
#     extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))

#     # Inicializa un objeto PatchExtractor con el tamaño de parche calculado,
#     # el número máximo de parches, y una semilla de estado aleatorio
#     extractor = PatchExtractor(
#         patch_size=extracted_patch_size, max_patches=N, random_state=0
#     )

#     # Extrae parches de la imagen dada
#     # img[np.newaxis] se utiliza la entrada de PatchExtractor es un conjunto de imágenes
#     patches = extractor.transform(img[np.newaxis])

#     # Si el factor de escala no es 1, redimensiona cada parche extraído
#     # al tamaño del parche original
#     if scale != 1:
#         patches = np.array([resize(patch, patch_size) for patch in patches])

#     # Devuelve la lista de parches extraídos (y posiblemente redimensionados)
#     return patches


# def load_background_patches():
#     # Tomamos algunas imágenes de sklearn
#     imgs = [
#         "text",
#         "coins",
#         "moon",
#         "page",
#         "clock",
#         "immunohistochemistry",
#         "chelsea",
#         "coffee",
#         "hubble_deep_field",
#     ]

#     images = []
#     for name in imgs:
#         img = getattr(data, name)()
#         if len(img.shape) == 3 and img.shape[2] == 3:  # Chequeamos si la imagen es RGB
#             img = color.rgb2gray(img)
#         images.append(resize(img, (100, 100)))

#     # Imagenes caseras adicionales
#     for i in range(31):
#         filename = "generar_fondos/pictures/" + str(i) + ".jpg"
#         img = plt.imread(filename)
#         img = color.rgb2gray(img)
#         images.append(resize(img, (100, 100)))

#     # Extraemos las imágenes de fondo
#     patches = np.vstack(
#         [
#             extract_patches(im, 64, scale)
#             for im in tqdm(images, desc="Procesando imágenes")
#             for scale in [0.1, 0.25, 0.5, 0.75, 1]
#         ]
#     )

#     # Fix: converimos float [0,1] a uint8 [0,255]
#     patches = (patches * 255).astype(np.uint8)

#     return patches


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
    # Create a new figure to avoid interference with other plots
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
    plt.close(fig)  # Always close to prevent overlap


def plot_pca_classes(X_train_pca, y_train, subtitle=None, show=False, save_path=None):
    # Create a new figure to avoid interference with other plots
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

    # Calculate optimal PC3 label position based on data range
    pc3_center = (X_train_pca[:, 2].max() + X_train_pca[:, 2].min()) / 2

    # Position PC3 label: full positive PC1, full negative PC2, centered on PC3
    pc3_x = X_train_pca[:, 0].max()  # Full positive of PC1
    pc3_y = X_train_pca[:, 1].min()  # Full negative of PC2
    pc3_z = pc3_center  # Centered on PC3 (Z) axis

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
    plt.close(fig)  # Always close to prevent overlap


def plot_pairplot(X_train_pca, y_train, subtitle=None, show=False, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))

    # Crear DataFrame con las primeras 5 componentes
    df = pd.DataFrame(X_train_pca[:, :5], columns=[f"PC{i + 1}" for i in range(5)])

    # Agregar etiquetas de clase
    df["label"] = y_train
    df["label"] = df["label"].map({0: "Back", 1: "Face"})

    # Colores manuales para que Back = rojo, Face = azul (como en la imagen)
    palette = {
        "Back": "#e24a33",  # red-like (index 3)
        "Face": "#348abd",  # blue (index 0)
    }

    # Crear el pairplot
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
