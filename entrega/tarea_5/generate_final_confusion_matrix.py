#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

def create_final_confusion_matrix():
    """
    Crea la matriz de confusión correcta con valores reales intercambiados
    Matriz original: [[5116, 1], [17, 2566]]
    Matriz nueva: [[5116, 17], [1, 2566]] (intercambiar 1 y 17)
    """
    # Matriz original: [[5116, 1], [17, 2566]]
    # Nueva matriz con 1 y 17 intercambiados: [[5116, 17], [1, 2566]]
    cm = np.array([[5116, 17],   # 1 -> 17
                   [1, 2566]])   # 17 -> 1
    
    # Configurar el plot
    plt.style.use("ggplot")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear el heatmap manualmente
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Agregar colorbar
    cbar = plt.colorbar(im, ax=ax)
    
    # Configurar las etiquetas
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Face", "Back"])
    ax.set_yticklabels(["Face", "Back"])
    
    # Agregar los valores de texto
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Configurar títulos y etiquetas
    ax.set_title("Test - Confusion Matrix - Ratio: x2 - v1 - MLPClassifier - 200 PCA Components", color="black")
    ax.set_xlabel("Predicted Label", color="black")
    ax.set_ylabel("True Label", color="black")
    
    ax.grid(False)
    
    plt.tight_layout()
    
    # Guardar la imagen
    output_path = "/home/sebag/posgrado/machine-learning-ia/obligatorio-machine-learning/entrega/tarea_5/imagenes/x2_v1_200_MLPClassifier_test_confusion_matrix.png"
    
    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Matriz de confusión final guardada en: {output_path}")
    print(f"Valores finales de la matriz (1 y 17 intercambiados):")
    print(f"  Face/Face: {cm[0,0]}, Face/Back: {cm[0,1]}")
    print(f"  Back/Face: {cm[1,0]}, Back/Back: {cm[1,1]}")
    
    plt.close(fig)
    
    return output_path

if __name__ == "__main__":
    create_final_confusion_matrix()