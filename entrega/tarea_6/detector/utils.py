from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from skimage import feature, color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
from itertools import product
import os
import time
from ipywidgets import interact, IntSlider
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# FUNCIONES DE CARGA DE IMÁGENES
# =============================================

def load_image(base_dir, image_path, target_size=500):
    """
    Carga una imagen y la redimensiona al target_size.
    
    Args:
        base_dir: Directorio base donde se encuentran las imágenes
        image_path: Ruta relativa a la imagen
        target_size: Tamaño objetivo (por defecto 500x500)
        
    Returns:
        Tupla con:
        - El nombre del archivo de la imagen.
        - La imagen en formato numpy array (escala de grises, 500x500).
        - La cantidad real de rostros.
    """
    try:        
        full_path = os.path.join(base_dir, image_path)
        
        # Verificar que el archivo existe
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No se encontró la imagen en la ruta: {full_path}")

        # Cargar imagen usando matplotlib
        imagen = mpimg.imread(full_path)
        
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            # Imagen en color, convertir a escala de grises
            if imagen.shape[2] == 4:  # RGBA
                imagen = color.rgba2rgb(imagen)
            imagen = color.rgb2gray(imagen)
        
        # Asegurar que esté en el rango correcto
        if imagen.max() <= 1.0:
            imagen = (imagen * 255).astype(np.uint8)
        else:
            imagen = imagen.astype(np.uint8)
        
        #print(f"✅ Imagen cargada: {full_path}")
        #print(f"📏 Dimensiones originales: {imagen.shape}")
        print(f"🔍 Tipo de dato: {imagen.dtype}")
        print(f"🔍 Rango de valores: {imagen.min()} - {imagen.max()}")

        # Redimensionar a target_size x target_size manteniendo proporciones
        imagen_resized, scale_factor, offsets = preprocess_image_optimal(
            imagen, target_size=target_size, keep_scale=True
        )
        
        print(f"📐 Dimensiones: {imagen_resized.shape}")
        #print(f"🔍 Factor de escala aplicado: {scale_factor:.3f}")
        
        # Extraer cantidad de rostros desde el nombre del archivo
        try:
            filename = os.path.basename(image_path)
            num_faces = int(filename.split('-')[0])
            print(f"✅ Cantidad real de rostros: {num_faces}")
        except ValueError:
            print(f"❌ Error al extraer cantidad de rostros del nombre: {filename}")
            return None  # Fallar si no se puede extraer la cantidad de rostros
        
        return filename, imagen_resized, num_faces
        
    except Exception as e:
        print(f"❌ Error al cargar imagen: {e}")
        print(f"💡 Ruta intentada: {image_path}")
        print(f"💡 Asegúrate de que la ruta sea correcta")
        return None

def load_multiple_images(base_dir, image_paths, target_size=500):
    """
    Carga múltiples imágenes desde image_paths y las redimensiona al target_size.
    
    Args:
        base_dir: Directorio base donde se encuentran las imágenes
        image_paths: Lista de rutas a las imágenes
        target_size: Tamaño objetivo (por defecto 500x500)
        
    Returns:
        Lista de tuplas con:
            - El nombre del archivo de la imagen.
            - La imagen en formato numpy array (escala de grises, 500x500).
            - La cantidad real de rostros.
    """
    images_with_faces = []

    print(f"🖼️ Cargando {len(image_paths)} imágenes...")
    
    for i, path in enumerate(image_paths):
        print(f"\n📥 Cargando imagen {i+1}/{len(image_paths)}: {path}")
        
        #print(f"🔍 Ruta completa: {os.path.join(base_dir, path)}")
        result = load_image(base_dir, path, target_size=target_size)
        if result is not None:
            filename, img, num_faces = result
            images_with_faces.append((filename, img, num_faces))
            # print(f"✅ Imagen {i+1} cargada exitosamente con {num_faces} rostros reales")
        else:
            print(f"❌ Error cargando imagen {i+1}")
            return None  # Fallar si cualquier imagen no se puede cargar
    
    print(f"\n🎉 Todas las {len(images_with_faces)} imágenes cargadas exitosamente!")
    return images_with_faces

def browse_images(directory="", show_preview=False):
    """
    Explora imágenes disponibles en directory y las retorna ordenadas alfabéticamente.
    
    Args:
        directory: Directorio a explorar (relativo a this)
        show_preview: Si mostrar preview de las imágenes encontradas
        
    Returns:
        Lista de rutas de imágenes encontradas, ordenadas alfabéticamente por nombre de archivo.
    """
    try:
        # Determinar directorio base
        if directory:
            base_dir = os.path.join(directory, "test_images")
        else:
            base_dir = "./test_images"
        
        if not os.path.exists(base_dir):
            print(f"❌ Directorio no encontrado: {base_dir}")
            return []
        
        # Buscar archivos de imagen
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_files = []
        
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    # Convertir a ruta relativa desde test_images
                    rel_path = os.path.relpath(full_path, base_dir)
                    image_files.append(rel_path)
        
        # Ordenar las imágenes alfabéticamente por nombre de archivo
        image_files.sort(key=lambda x: os.path.basename(x).lower())
        
        print(f"🔍 Encontradas {len(image_files)} imágenes en {base_dir}")
        
        if image_files:
            print("\n📋 IMÁGENES ENCONTRADAS:")
            print("-" * 50)
            for i, img_path in enumerate(image_files[:10]):  # Mostrar máximo 10
                print(f"{i+1:2d}. {img_path}")
            
            if len(image_files) > 10:
                print(f"... y {len(image_files) - 10} más")
            
            if show_preview and len(image_files) <= 4:
                print(f"\n🖼️ PREVIEW DE IMÁGENES:")
                fig, axes = plt.subplots(1, min(len(image_files), 4), figsize=(15, 4))
                if len(image_files) == 1:
                    axes = [axes]
                
                for i, img_path in enumerate(image_files[:4]):
                    img = load_image(base_dir, img_path, mount_drive=False, target_size=500)
                    if img is not None:
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(os.path.basename(img_path), fontsize=10)
                        axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()
        
        return image_files
        
    except Exception as e:
        print(f"❌ Error explorando: {e}")
        return []

# =============================================
# FUNCIONES DE UTILIDAD
# =============================================

def preprocess_image_optimal(image, target_size=500, keep_scale=False):
    """
    Preprocesa imagen manteniendo proporciones para imágenes no cuadradas.

    Args:
        image: Imagen de entrada como numpy array.
        target_size: Tamaño objetivo para la imagen cuadrada.
        keep_scale: Si True, mantiene la proporción original de la imagen y usa target_size como altura.
                    Si False, aplica el comportamiento estándar de redimensionar y agregar padding negro.
    Returns:
        Tupla con:
        - Imagen procesada como numpy array.
        - Factor de escala aplicado.
        - Tupla con offsets (x, y) para centrar la imagen (solo si keep_scale es False).
    """
    h, w = image.shape[:2]

    if keep_scale:
        # Mantener proporciones y usar target_size como altura
        scale_factor = target_size / h
        new_h, new_w = target_size, int(w * scale_factor)
        resized = resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)
        return resized.astype(np.uint8), scale_factor, None
    else:
        # Mantener proporciones y agregar padding negro
        scale_factor = target_size / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)

        padded = np.zeros((target_size, target_size, image.shape[2] if len(image.shape) == 3 else 1))
        if len(image.shape) == 2:
            padded = np.zeros((target_size, target_size))

        offset_y = (target_size - new_h) // 2
        offset_x = (target_size - new_w) // 2

        if len(image.shape) == 3:
            padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        else:
            padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        if len(padded.shape) == 3:
            padded = color.rgb2gray(padded)

        return padded.astype(np.uint8), scale_factor, (offset_x, offset_y)

def show_image(image, title=None, cmap='gray', figsize=(8, 8)):
    """Muestra una imagen con título opcional."""
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_with_rectangle(image, true_scale,title=None, cmap='gray', figsize=(8, 8)):
    """Muestra una imagen con título opcional y un recuadro."""
    
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)

    if title:
        plt.title(title)

    Ni, Nj = (int(true_scale * s) for s in (64,64))

    width = image.shape[1] / 2 - Nj/2
    height = image.shape[0] / 2 - Ni / 2

    ax.add_patch(plt.Rectangle((width, height), Nj, Ni, edgecolor='red', alpha=1, lw=1, facecolor='none'))

def show_images(images, max_per_row):
    """
    Display a list of images in a grid format.

    Parameters:
    images (list of np.array): List of images to display.
    max_per_row (int): Maximum number of images to display per row.

    Returns:
    None
    """
    num_images = len(images)
    num_rows = (num_images + max_per_row - 1) // max_per_row  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, max_per_row, figsize=(max_per_row * 3, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes for easier indexing

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused axes

    plt.tight_layout()
    plt.show()

def compare_image_sets(path_a, path_b, path_c):
    """
    Muestra lado a lado las imágenes correspondientes de tres carpetas
    (A, B y C) para comparación visual con un deslizador interactivo.

    Cada carpeta debe contener las mismas imágenes (mismo orden / nombre)
    y la misma cantidad de archivos .jpg (o modificar la extensión).

    Parameters
    ----------
    path_a, path_b, path_c : str o Path
        Rutas a las carpetas que contienen las imágenes de cada set.
    """
    dir_A = Path(path_a)
    dir_B = Path(path_b)
    dir_C = Path(path_c)

    files_A = sorted(dir_A.glob('*.jpg'))
    files_B = sorted(dir_B.glob('*.jpg'))
    files_C = sorted(dir_C.glob('*.jpg'))

    n = len(files_A)
    assert n == len(files_B) == len(files_C), "Los tres sets deben tener la misma cantidad de imágenes"

    def show_triplet(idx):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        img_A = mpimg.imread(files_A[idx])
        img_B = mpimg.imread(files_B[idx])
        img_C = mpimg.imread(files_C[idx])

        axes[0].imshow(img_A)
        axes[0].set_title("Modelo A")
        axes[0].axis("off")

        axes[1].imshow(img_B)
        axes[1].set_title("Modelo B")
        axes[1].axis("off")

        axes[2].imshow(img_C)
        axes[2].set_title("Modelo C")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    interact(
        show_triplet,
        idx=IntSlider(value=0, min=0, max=n-1, step=1, description="Imagen")
    );
# =============================================
# FUNCIONES DE DETECCIÓN
# =============================================

def non_max_suppression(indices, sizes, overlap_thresh, scores=None, use_scores=False):
    if len(indices) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    if indices.dtype.kind == "i":
        indices = indices.astype("float")

    pick = []

    x1 = np.array([indices[i, 0] for i in range(indices.shape[0])])
    y1 = np.array([indices[i, 1] for i in range(indices.shape[0])])
    Ni = np.array([sizes[i, 0] for i in range(sizes.shape[0])])
    Nj = np.array([sizes[i, 1] for i in range(sizes.shape[0])])
    x2 = x1 + Ni
    y2 = y1 + Nj
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if use_scores and scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return indices[pick].astype("int"), sizes[pick]

def sliding_window(img, patch_size=(64,64), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Nj, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = resize(patch, patch_size)
            yield (i, j), patch

def global_multiscale_detection(image, clf, scaler, pca, test_scales,
                                patch_size=(64, 64), 
                                threshold=0.1, 
                                step=2,
                                overlap_thresh=0.3, 
                                batch_size=512, 
                                max_jobs=3,
                                hog_params = {
                                    "feature_vector": True
                                },
                                plot=True, 
                                base_dir="./",
                                save_result=False,
                                result_dir="detections",
                                filename=None):
    """
    Detección multiescala optimizada con mejoras de performance.
    
    Args:
        image: Imagen de entrada para detección
        clf: Clasificador entrenado
        scaler: Normalizador de características
        pca: Transformación PCA
        test_scales: Lista de escalas a evaluar
        patch_size: Tamaño del patch (altura, ancho)
        threshold: Umbral de probabilidad para detección
        step: Paso de la ventana deslizante
        overlap_thresh: Umbral para supresión de no-máximos
        batch_size: Tamaño del lote para procesamiento por lotes
        max_jobs: Número máximo de trabajos paralelos
        hog_params: Parámetros para la extracción de características HOG
        plot: Si mostrar resultados gráficamente
        base_dir: Directorio base para guardar resultados
        save_result: Si guardar la imagen de resultados
        filename: Nombre de la imagen

    Returns:
        tuple: (indices_filtrados, tamaños_filtrados) después de NMS
    """
    import gc
    start_time = time.time()

    global_indices = []
    global_scores = []
    global_sizes = []

    if filename is not None:
        print(f"🔍 Iniciando detección multiescala para {filename}...")
    else:
        print(f"🔍 Iniciando detección multiescala...")

    def process_scale(scale):
        Ni, Nj = (int(scale * patch_size[0]), int(scale * patch_size[1]))
        indices = []
        patches = []
        for (i, j), patch in sliding_window(image, scale=scale, istep=step, jstep=step):
            indices.append((i, j))
            patches.append(patch)
        if not patches:
            return np.empty((0,2)), np.empty((0,)), []
        indices = np.array(indices)
        selected_indices = []
        selected_scores = []
        # --- Procesar por lotes para limitar memoria ---
        for start in range(0, len(patches), batch_size):
            end = min(start + batch_size, len(patches))
            batch_patches = patches[start:end]
            batch_indices = indices[start:end]
            batch_hog = np.array([feature.hog(p, **hog_params) for p in batch_patches])
            batch_hog = scaler.transform(batch_hog)
            batch_hog = pca.transform(batch_hog)
            probas = clf.predict_proba(batch_hog)[:, 1]
            mask = probas >= threshold
            selected_indices.append(batch_indices[mask])
            selected_scores.append(probas[mask])
            # Liberar memoria del batch
            del batch_patches, batch_hog, probas, mask
            gc.collect()
        if selected_indices:
            selected_indices = np.concatenate(selected_indices, axis=0)
            selected_scores = np.concatenate(selected_scores, axis=0)
        else:
            selected_indices = np.empty((0,2), dtype=int)
            selected_scores = np.empty((0,))
        selected_sizes = [(Ni, Nj)] * len(selected_indices)
        # Liberar memoria de la escala
        del patches, indices
        gc.collect()
        return selected_indices, selected_scores, selected_sizes

    # --- Limitar el número de procesos paralelos ---
    results = Parallel(n_jobs=max_jobs)(delayed(process_scale)(scale) for scale in test_scales)

    for selected_indices, selected_scores, sizes in results:
        global_indices.extend(selected_indices)
        global_scores.extend(selected_scores)
        global_sizes.extend(sizes)
        # Liberar memoria tras cada escala
        gc.collect()

    global_indices = np.array(global_indices)
    global_scores = np.array(global_scores)
    global_sizes = np.array(global_sizes)

    filtered_indices, filtered_sizes = non_max_suppression(
        global_indices,
        global_sizes,
        overlap_thresh=overlap_thresh,
        scores=global_scores,
        use_scores=True
    )

    processing_time = time.time() - start_time
    minutes, seconds = divmod(processing_time, 60)
    print(f"⏱️ Tiempo total: {int(minutes)}m {seconds:.2f}s")
    print(f"🎯 Detecciones finales después de NMS: {len(filtered_indices)}")

    if plot or save_result:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        for (i, j), (Ni, Nj) in zip(filtered_indices, filtered_sizes):
            ax.add_patch(plt.Rectangle(
                (j, i), Nj, Ni,
                edgecolor='blue', alpha=0.7, lw=2,
                facecolor='none'
            ))

        if save_result and filename is not None:      
            output_dir = os.path.join(base_dir, result_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Crear params.txt si no existe
            params_path = os.path.join(output_dir, "params.txt")
            if not os.path.exists(params_path):
                with open(params_path, "w") as f:
                    f.write(f"threshold={threshold}\n")
                    f.write(f"overlap_thresh={overlap_thresh}\n")
                    f.write(f"test_scales={test_scales}\n")

            output_path = os.path.join(output_dir, f"{filename}.jpg")
            fig.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Imagen guardada en: {output_path}")

        elif save_result and not filename:
            print("⚠️ Warning: save_result=True pero no se proporcionó filename")

        if plot:
            plt.show()
        else:
            plt.close(fig)

    return filtered_indices, filtered_sizes

def grid_search(clf, 
                scaler, 
                pca, 
                dataset, 
                thresholds, 
                steps, 
                overlaps, 
                test_scales,
                patch_size=(64, 64), 
                batch_size=512, 
                max_jobs=3,
                hog_params = {
                    "feature_vector": True
                },
                verbose=True,
                base_dir="./",
                early_stop_f1=0.3,
                n_images_max=3):
    """
    Realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros de detección.
    Args:
        clf: Clasificador entrenado
        scaler: Normalizador de características
        pca: Transformación PCA
        dataset: Lista de tuplas (filename, image, true_faces)
        thresholds: Lista de umbrales a evaluar
        steps: Lista de pasos de la ventana deslizante a evaluar
        overlaps: Lista de umbrales de superposición a evaluar
        test_scales: Lista de escalas a evaluar
        patch_size: Tamaño del patch (altura, ancho)
        batch_size: Tamaño del lote para procesamiento por lotes
        max_jobs: Número máximo de trabajos paralelos
        hog_params: Parámetros para la extracción de características HOG
        base_dir: Directorio base para guardar resultados
        early_stop_f1: si el promedio parcial de F1 cae por debajo de este valor, se interrumpe.
        n_images_max: cantidad máxima de imágenes a evaluar por combinación.
    Returns:
        df_grid: DataFrame ordenado por F1
        best_config: Diccionario con la mejor combinación
    """
  
    grid_results = []

    for threshold, step, overlap in product(thresholds, steps, overlaps):
        if verbose:
            print(f"\n🔍 Evaluando: th={threshold}, step={step}, ov={overlap}")

        total_f1 = 0
        total_precision = 0
        total_recall = 0
        total_error = 0

        for i, (filename, image, true_faces) in enumerate(dataset):
            if i >= n_images_max:
                break

            indices, sizes = global_multiscale_detection(
                image=image,
                clf=clf,
                scaler=scaler,
                pca=pca,
                test_scales=test_scales,
                threshold=threshold,
                step=step,
                overlap_thresh=overlap,
                patch_size=patch_size,
                batch_size=batch_size,
                max_jobs=max_jobs,
                hog_params=hog_params,
                plot=False,
                save_result=False,
                base_dir=base_dir,
                filename=filename
            )

            detected_faces = len(indices)
            correct = min(detected_faces, true_faces)

            precision = correct / detected_faces if detected_faces > 0 else 0
            recall = correct / true_faces if true_faces > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            abs_error = abs(detected_faces - true_faces)

            total_f1 += f1
            total_precision += precision
            total_recall += recall
            total_error += abs_error

            # Early stopping si la combinación es mala
            avg_f1_partial = total_f1 / (i + 1)
            if avg_f1_partial < early_stop_f1:
                if verbose:
                    print(f"⛔ Early stop: F1 parcial ({avg_f1_partial:.2f}) < {early_stop_f1}")
                break

        n_used = min(n_images_max, i + 1)

        grid_results.append({
            "threshold": threshold,
            "step": step,
            "overlap": overlap,
            "avg_F1": round(total_f1 / n_used, 3),
            "avg_Precision": round(total_precision / n_used, 3),
            "avg_Recall": round(total_recall / n_used, 3),
            "avg_AbsError": round(total_error / n_used, 2)
        })

    df_grid = pd.DataFrame(grid_results)
    df_grid = df_grid.sort_values(by="avg_F1", ascending=False)

    print("\n📊 Top combinaciones (modo rápido):")
    print(df_grid.head(10))

    best_row = df_grid.iloc[0]
    return df_grid, best_row.to_dict()

def plot_f1_heatmap(df_results, fix_overlap=0.2):
    """
    Dibuja un heatmap de F1-score promedio en función de threshold y step,
    manteniendo fijo un valor de overlap.

    Args:
        df_results (pd.DataFrame): Resultados del grid search.
        fix_overlap (float): Valor de overlap a mantener fijo para el gráfico.
    """
    # Filtrar combinaciones con el overlap deseado
    df_plot = df_results[df_results["overlap"] == fix_overlap]

    # Crear tabla pivote para heatmap
    pivot = df_plot.pivot(index="threshold", columns="step", values="avg_F1")

    # Graficar
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'F1-score promedio'})
    plt.title(f"F1-score por combinación (overlap = {fix_overlap})")
    plt.ylabel("threshold")
    plt.xlabel("step")
    plt.tight_layout()
    plt.show()

def plot_top_combinations_bar(df_results, top_n=5):
    """
    Muestra un gráfico de barras con las top-n combinaciones por F1-score promedio.
    Incluye el error absoluto promedio como anotación.

    Args:
        df_results (pd.DataFrame): Resultados del grid search.
        top_n (int): Número de combinaciones a mostrar.
    """
    # Seleccionar top-n por F1
    top_df = df_results.sort_values(by="avg_F1", ascending=False).head(top_n)

    # Crear etiquetas combinadas
    labels = [
        f"th={row.threshold}, st={row.step}, ov={row.overlap}"
        for _, row in top_df.iterrows()
    ]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, top_df["avg_F1"], color='skyblue')

    # Agregar etiquetas con error absoluto
    for bar, err in zip(bars, top_df["avg_AbsError"]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"err={err:.2f}",
                 ha='center', va='bottom', fontsize=9)

    plt.title(f"Top {top_n} combinaciones por F1-score promedio")
    plt.ylabel("F1-score promedio")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()