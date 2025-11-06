import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import numpy as np
import nltk 
# ==============================================================================
# --- Gráficos de Análisis Exploratorio (EDA) ---
# ==============================================================================

def plot_class_distribution(y_data, class_names, save_path):
    """
    Crea un gráfico de pastel (pie chart) de la distribución de clases.
    Muestra el porcentaje y el nombre de las clases.
    """
    print("Graficando distribución de clases...")
    class_counts = pd.Series(y_data).value_counts(normalize=True) * 100
    class_labels = [f"{class_names[i]} ({class_counts[i]:.1f}%)" for i in class_counts.index]
    
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Distribución de Clases en el Dataset')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_text_length_histogram(texts_list, title, save_path):
    """
    Crea un histograma de la longitud de los textos (por # de palabras).
    """
    print(f"Graficando histograma: {title}...")
    lengths = [len(text.split()) for text in texts_list]
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50, kde=True)
    plt.title(f'Distribución de Longitud de Texto ({title})')
    plt.xlabel('Número de Palabras')
    plt.ylabel('Frecuencia')
    plt.savefig(save_path)
    plt.close()

def plot_most_common_words(texts_list, title, save_path, n=20):
    """
    Crea un gráfico de barras de las 'n' palabras más comunes (Unigramas).
    """
    print(f"Graficando palabras comunes: {title}...")
    all_words = ' '.join(texts_list).split()
    word_counts = Counter(all_words)
    
    common_words = pd.DataFrame(word_counts.most_common(n), columns=['Palabra', 'Frecuencia'])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frecuencia', y='Palabra', data=common_words, palette='viridis')
    plt.title(f'{n} Palabras Más Comunes ({title})')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_word_cloud(texts_list, title, save_path):
    """Crea una nube de palabras."""
    print(f"Generando Nube de Palabras: {title}...")
    all_text = ' '.join(texts_list)
    if not all_text:
        print(f"Advertencia: No hay texto para la nube de palabras '{title}'")
        return
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_zipf_law(texts_list, save_path):
    """
    Grafica la Ley de Zipf (frecuencia vs rango en escala log-log).
    """
    print("Graficando Ley de Zipf...")
    all_words = ' '.join(texts_list).split()
    word_counts = Counter(all_words)
    
    # Obtener frecuencias y ordenarlas de mayor a menor
    frequencies = sorted(word_counts.values(), reverse=True)
    # Generar rangos (1, 2, 3...)
    ranks = range(1, len(frequencies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Ley de Zipf (Frecuencia vs. Rango)')
    plt.xlabel('Rango (log)')
    plt.ylabel('Frecuencia (log)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_word_frequency_distribution(texts_list, save_path):
    """
    Grafica cuántas palabras aparecen 1 vez, 2 veces, etc.
    Esencial para ver los "Hapax Legomena" (palabras de un solo uso).
    """
    print("Graficando distribución de frecuencias de palabras...")
    all_words = ' '.join(texts_list).split()
    word_counts = Counter(all_words)
    
    # Obtener solo las frecuencias (cuántas veces apareció cada palabra)
    frequencies = list(word_counts.values())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(frequencies, bins=50, log_scale=(False, True)) 
    plt.title('Distribución de Frecuencias de Palabras')
    plt.xlabel('Frecuencia de la Palabra (Cuántas veces aparece)')
    plt.ylabel('Conteo de Palabras (log scale)')
    plt.xlim(0, 50) 
    plt.savefig(save_path)
    plt.close()

def plot_vocabulary_growth(texts_list, save_path):
    """
    Grafica la curva de crecimiento del vocabulario (Ley de Heaps).
    """
    print("Graficando curva de crecimiento del vocabulario...")
    seen_words = set()
    vocab_size_over_time = []
    
    for doc in texts_list:
        seen_words.update(doc.split())
        vocab_size_over_time.append(len(seen_words))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(texts_list) + 1), vocab_size_over_time)
    plt.title('Curva de Crecimiento del Vocabulario')
    plt.xlabel('Documentos Procesados')
    plt.ylabel('Tamaño de Vocabulario Único')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_most_common_ngrams(texts_list, n_gram, title, save_path, n=20):
    """
    Crea un gráfico de barras de los 'n' n-gramas más comunes (ej. bigramas).
    """
    print(f"Graficando {n}-gramas comunes: {title}...")
    all_ngrams = []
    for text in texts_list:
        tokens = text.split()
        all_ngrams.extend(nltk.ngrams(tokens, n_gram))
    
    ngram_counts = Counter(all_ngrams)
    
    # Formatear n-gramas (tuplas) a strings para graficar
    common_ngrams_data = []
    for ng, freq in ngram_counts.most_common(n):
        common_ngrams_data.append((' '.join(ng), freq))

    common_ngrams_df = pd.DataFrame(common_ngrams_data, columns=['N-grama', 'Frecuencia'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frecuencia', y='N-grama', data=common_ngrams_df, palette='viridis')
    plt.title(f'{n} {n_gram}-gramas Más Comunes ({title})')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pos_tag_distribution(texts_list, save_path):
    """
    Grafica la distribución de las etiquetas PoS (ej. _V, _N, <MENTION>).
    """
    print("Graficando distribución de etiquetas PoS...")
    tags = []
    for text in texts_list:
        for token in text.split():
            if '_' in token:
                tag = token.split('_')[-1]
            elif token.startswith('<') and token.endswith('>'):
                tag = token
            else:
                tag = 'OTHER' # Palabras sin etiqueta
            tags.append(tag)
    
    tag_counts = Counter(tags)
    tag_df = pd.DataFrame(tag_counts.most_common(20), columns=['Etiqueta', 'Frecuencia'])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frecuencia', y='Etiqueta', data=tag_df, palette='magma')
    plt.title('Distribución de Etiquetas PoS y Tokens Especiales')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_comparative_word_clouds_by_class(X_clean_list, y_flat_list, 
                                          class_names_dict, save_path_prefix):
    """
    Crea nubes de palabras separadas para cada clase (ej. positivo vs negativo).
    'class_names_dict' debe ser un mapeo, ej: {0: 'negative', 1: 'positive'}
    """
    print("Generando nubes de palabras comparativas...")
    df = pd.DataFrame({'text': X_clean_list, 'label': y_flat_list})
    
    for label_id, label_name in class_names_dict.items():
        # Filtrar texto para esta clase
        class_text_list = df[df['label'] == label_id]['text'].tolist()
        
        # Generar nube de palabras para esta clase
        plot_word_cloud(
            class_text_list, 
            title=f'Nube de Palabras - Clase: {label_name.upper()}', 
            save_path=f"{save_path_prefix}_clase_{label_name}.png"
        )

# ==============================================================================
# --- Gráficos de Evaluación de Modelos ---
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    """
    Crea un heatmap de la matriz de confusión.
    """
    print(f"Graficando Matriz de Confusión: {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar para mostrar porcentajes de 'recall'
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión (Normalizada por Fila) - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_path):
    """
    Grafica las curvas de Accuracy y Loss de entrenamiento vs validación.
    """
    print("Graficando historial de entrenamiento de la Red Neuronal...")
    history_df = pd.DataFrame(history.history)
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Épocas')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Épocas')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def plot_zipf_law(all_tokens_list, title, save_path):
    """
    Grafica la Ley de Zipf (frecuencia vs rango en escala log-log).
    MODIFICADO: Acepta una lista plana de todos los tokens.
    """
    print(f"Graficando Ley de Zipf: {title}...")
    word_counts = Counter(all_tokens_list)
    
    # Obtener frecuencias y ordenarlas de mayor a menor
    frequencies = sorted(word_counts.values(), reverse=True)
    # Generar rangos (1, 2, 3...)
    ranks = range(1, len(frequencies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'Ley de Zipf ({title})')
    plt.xlabel('Rango (log)')
    plt.ylabel('Frecuencia (log)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


