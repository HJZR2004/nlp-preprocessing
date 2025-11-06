import numpy as np
import pandas as pd 
import os
import sys
import pickle
import limpieza
import vectorizar
import modelo
import graph
from keras.utils import to_categorical 
from sklearn.utils import class_weight
import nltk
from tqdm import tqdm

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


import vectorizar 
import modelo 

def define_experiments():
    """
    Define todas las combinaciones de (Limpieza + Vectorización + Modelo)
    que se ejecutarán.
    
    Cada 'name' DEBE ser único para no sobrescribir archivos.
    """
        
    # Vectorización (Tamaño del Vocabulario)
    vec_params_5k = {'max_features': 5000}
    vec_params_2k = {'max_features': 2000}
    vec_params_10k = {'max_features': 10000}

    # G1: Enriquecido + Lema + Stopwords IN (Línea Base)
    clean_enr_lema_in = {"pipeline_type": "enriquecido", "remove_stopwords": False, "stemming": False, "lemmatization": True}
    
    # G3: Enriquecido + Lema + Stopwords OUT
    clean_enr_lema_out = {"pipeline_type": "enriquecido", "remove_stopwords": True, "stemming": False, "lemmatization": True}
    
    # G4: Enriquecido + Stemming + Stopwords IN
    clean_enr_stem_in = {"pipeline_type": "enriquecido", "remove_stopwords": False, "stemming": True, "lemmatization": False}

    # G4: Enriquecido + Stemming + Stopwords OUT
    clean_enr_stem_out = {"pipeline_type": "enriquecido", "remove_stopwords": True, "stemming": True, "lemmatization": False}
    
    # G5: Simple + Lema + Stopwords OUT
    clean_sim_lema_out = {"pipeline_type": "simple", "remove_stopwords": True, "stemming": False, "lemmatization": True}

    # --- Lista de Experimentos ---
    experiments = []

    # =========================================
    # G1: Baseline (Enriquecido) - (IDs 1, 2, 3)
    # =========================================

    # --- Experimento 1: (Línea Base) Enriquecido + TF-IDF + Regresión Logística ---
    experiments.append({
        "name": "Enriquecido_TFIDF_LR",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })
    
    # --- Experimento 2: Enriquecido + BOW + Regresion logistica ---
    experiments.append({
        "name": "Enriquecido_BOW_LR",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_bow,
        "vec_name": "BoW",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })
    
    # --- Experimento 3: Enriquecido + OHE + Regresion Logistica ---
    experiments.append({
        "name": "Enriquecido_OHE_LR",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_ohe,
        "vec_name": "OHE",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })

    # ================================
    # G2: Variación de Modelos - (IDs 4, 5, 6, 7, 8)
    # ================================

    # --- Experimento 4: Enriquecido + TF-IDF + Naive Bayes ---
    experiments.append({
        "name": "Enriquecido_TFIDF_NB",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # --- Experimento 5: Enriquecido + TF-IDF + Random Forest ---
    experiments.append({
        "name": "Enriquecido_TFIDF_RF",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_random_forest,
        "model_name": "RandomForest",
        "is_nn": False
    })

    # --- Experimento 6: Enriquecido + TF-IDF + Red Neuronal ---
    experiments.append({
        "name": "Enriquecido_TFIDF_NN",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_nn_model,
        "model_name": "NeuralNetwork",
        "is_nn": True
    })

    # --- Experimento 7: Enriquecido + BoW + Naive Bayes ---
    experiments.append({
        "name": "Enriquecido_BOW_NB",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_bow,
        "vec_name": "BoW",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # --- Experimento 8: Enriquecido + OHE + Red Neuronal ---
    experiments.append({
        "name": "Enriquecido_OHE_NN",
        "clean_params": clean_enr_lema_in,
        "vec_func": vectorizar.vectorize_ohe,
        "vec_name": "OHE",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_nn_model,
        "model_name": "NeuralNetwork",
        "is_nn": True
    })

    # =========================================
    # G3: Impacto de Stopwords - (IDs 9, 10, 11)
    # =========================================

    # --- Experimento 9: Enriquecido (Sin Stopwords) + TF-IDF + Regresión Logística ---
    experiments.append({
        "name": "EnriquecidoNoStop_TFIDF_LR",
        "clean_params": clean_enr_lema_out,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })
    
    # --- Experimento 10: Enriquecido (Sin Stopwords) + TF-IDF + Naive Bayes ---
    # (Este es el que te faltaba en tu código original)
    experiments.append({
        "name": "EnriquecidoNoStop_TFIDF_NB",
        "clean_params": clean_enr_lema_out,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # --- Experimento 11: Enriquecido (Sin Stopwords) + BoW + Random Forest ---
    # (Tu código tenía TF-IDF + RF, la tabla especificaba BoW + RF)
    experiments.append({
        "name": "EnriquecidoNoStop_BOW_RF",
        "clean_params": clean_enr_lema_out,
        "vec_func": vectorizar.vectorize_bow,
        "vec_name": "BoW",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_random_forest,
        "model_name": "RandomForest",
        "is_nn": False
    })

    # =========================================
    # G4: Impacto de Stemming - (IDs 12, 13, 14)
    # =========================================

    # --- Experimento 12: Enriquecido (Stemming) + TF-IDF + Regresión Logística ---
    experiments.append({
        "name": "EnriquecidoStem_TFIDF_LR",
        "clean_params": clean_enr_stem_in, # Con Stemming, Con Stopwords
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })

    # --- Experimento 13: Enriquecido (Stemming, Sin Stopwords) + TF-IDF + Regresión Logística ---
    experiments.append({
        "name": "EnriquecidoStemNoStop_TFIDF_LR",
        "clean_params": clean_enr_stem_out, # Con Stemming, Sin Stopwords
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })

    # --- Experimento 14: Enriquecido (Stemming, Sin Stopwords) + TF-IDF + Naive Bayes ---
    experiments.append({
        "name": "EnriquecidoStemNoStop_TFIDF_NB",
        "clean_params": clean_enr_stem_out, # Con Stemming, Sin Stopwords
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # =========================================
    # G5: Pipeline Simple - (IDs 15, 16, 17, 18)
    # =========================================

    # --- Experimento 15: (Línea Base Simple) Simple + TF-IDF + Regresión Logística ---
    experiments.append({
        "name": "Simple_TFIDF_LR",
        "clean_params": clean_sim_lema_out, # Simple, Sin Stopwords, Con Lema
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })

    # --- Experimento 16: Simple + TF-IDF + Naive Bayes ---
    experiments.append({
        "name": "Simple_TFIDF_NB",
        "clean_params": clean_sim_lema_out,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # --- Experimento 17: Simple + TF-IDF + Random Forest ---
    experiments.append({
        "name": "Simple_TFIDF_RF",
        "clean_params": clean_sim_lema_out,
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_random_forest,
        "model_name": "RandomForest",
        "is_nn": False
    })

    # --- Experimento 18: Simple + BoW + Naive Bayes ---
    experiments.append({
        "name": "Simple_BOW_NB",
        "clean_params": clean_sim_lema_out,
        "vec_func": vectorizar.vectorize_bow,
        "vec_name": "BoW",
        "vec_params": vec_params_5k,
        "model_func": modelo.train_naive_bayes,
        "model_name": "NaiveBayes",
        "is_nn": False
    })

    # =========================================
    # G6: Variación de Features - (IDs 19, 20)
    # =========================================

    # --- Experimento 19: Enriquecido + TF-IDF (2k) + Regresión Logística ---
    experiments.append({
        "name": "Enriquecido_TFIDF-2k_LR",
        "clean_params": clean_enr_lema_in, # Baseline Enriquecido
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_2k, # Menos features
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })

    # --- Experimento 20: Enriquecido + TF-IDF (10k) + Regresión Logística ---
    experiments.append({
        "name": "Enriquecido_TFIDF-10k_LR",
        "clean_params": clean_enr_lema_in, # Baseline Enriquecido
        "vec_func": vectorizar.vectorize_tfidf,
        "vec_name": "TF-IDF",
        "vec_params": vec_params_10k, # Más features
        "model_func": modelo.train_logistic_regression,
        "model_name": "LogisticRegression",
        "is_nn": False
    })
    
    print(f"Se han definido {len(experiments)} experimentos.")
    return experiments

# --- 2. FUNCIÓN PRINCIPAL DEL MOTOR DE EXPERIMENTOS ---
def run_experiment_engine():
    """
    Carga datos una vez y luego itera sobre todas las configuraciones 
    de experimentos definidas.
    """
    
    # --- 0. Configuración Global ---
    config = {
        "graficos_dir": "graficos",
        "modelos_dir": "modelos_serializados",
        "class_names": ['negative', 'positive', 'neutral', 'obj-neutral', 'obj']
    }
    os.makedirs(config["graficos_dir"], exist_ok=True)
    os.makedirs(config["modelos_dir"], exist_ok=True)
    
    config["num_classes"] = len(config["class_names"])
    
    # --- 1. Cargar Datos Crudos (SOLO UNA VEZ) ---
    print("Paso 1: Cargando datos crudos (una sola vez)...")
    X_train_raw, y_train = limpieza.load_data('raw-train.txt')
    X_test_raw, y_test = limpieza.load_data('raw-test.txt')
    
    # Preparar Y (SOLO UNA VEZ)
    config["y_train_flat"] = y_train.ravel()
    config["y_test_flat"] = y_test.ravel()
    config["y_train_cat"] = to_categorical(y_train, num_classes=config["num_classes"])
    config["y_test_cat"] = to_categorical(y_test, num_classes=config["num_classes"])
    
    # Calcular Class Weights (SOLO UNA VEZ)
    class_weights = class_weight.compute_class_weight('balanced',
        classes=np.unique(config["y_train_flat"]), y=config["y_train_flat"]
    )
    config["class_weights_dict"] = dict(enumerate(class_weights))
    
    # --- 2. Definir Experimentos ---
    experiments_to_run = define_experiments()
    
    # Log de resultados
    master_results_log = []

    # --- 3. BUCLE DE EXPERIMENTOS ---
    print(f"Iniciando {len(experiments_to_run)} experimentos...")
    
    for exp in tqdm(experiments_to_run, desc="Total de Experimentos"):
        
        # Extraer parámetros del experimento
        exp_name = exp["name"]
        clean_params = exp["clean_params"]
        vec_name = exp["vec_name"]
        vec_func = exp["vec_func"]
        vec_params = exp["vec_params"]
        model_name = exp["model_name"]
        model_func = exp["model_func"]
        is_nn = exp["is_nn"]

        print(f"\n========================================================")
        print(f"INICIANDO: {exp_name.upper()}")
        print(f"Limpieza: {clean_params} | Vector: {vec_name} | Modelo: {model_name}")
        print(f"========================================================")

        # --- A. Limpieza (Se ejecuta cada vez, como pediste) ---
        print(f"Limpiando con parámetros: {clean_params}")
        X_train_clean = []
        for txt_array in tqdm(X_train_raw, desc=f"Limpiando Train ({exp_name})", file=sys.stdout, leave=False):
            X_train_clean.append(limpieza.full_text_pipeline(txt_array[0], **clean_params))

        X_test_clean = []
        for txt_array in tqdm(X_test_raw, desc=f"Limpiando Test  ({exp_name})", file=sys.stdout, leave=False):
            X_test_clean.append(limpieza.full_text_pipeline(txt_array[0], **clean_params))

        # --- B. Vectorización ---
        print(f"Vectorizando con {vec_name}...")
        X_train_vec, X_test_vec, vectorizer = vec_func(
            X_train_clean, X_test_clean, **vec_params
        )
        
        # Guardar el vectorizador
        vec_path = os.path.join(config["modelos_dir"], f"{exp_name}_vectorizer.pkl")
        with open(vec_path, 'wb') as f: pickle.dump(vectorizer, f)

        # --- C. Entrenamiento y Evaluación ---
        
        log_entry = {"Experimento": exp_name, "Vectorizador": vec_name, "Modelo": model_name}

        if is_nn:
            # --- Modelo de Red Neuronal ---
            print(f"\n--- Entrenando: {model_name} ---")
            nn_model, history = model_func( # model_func es 'train_nn_model'
                X_train_vec, config["y_train_cat"], X_test_vec, config["y_test_cat"],
                num_features=X_train_vec.shape[1], 
                num_classes=config["num_classes"],
                class_weights_dict=config["class_weights_dict"]
            )
            
            # Evaluar y obtener métricas
            nn_metrics = modelo.evaluate_nn_model(nn_model, X_test_vec, config["y_test_cat"], config["class_names"])
            log_entry.update(nn_metrics)
            
            # Guardar gráficos y modelo
            graph.plot_training_history(history, save_path=f"{config['graficos_dir']}/hist_{exp_name}.png")
            y_pred_nn_probs = nn_model.predict(X_test_vec.toarray())
            y_pred_nn = np.argmax(y_pred_nn_probs, axis=1)
            graph.plot_confusion_matrix(config["y_test_flat"], y_pred_nn, config["class_names"], 
                                        exp_name, save_path=f"{config['graficos_dir']}/cm_{exp_name}.png")

            nn_path = os.path.join(config["modelos_dir"], f"{exp_name}_model.keras") 
            nn_model.save(nn_path)

        else:
            # --- Modelos Sklearn (NB, LR, RF) ---
            print(f"\n--- Entrenando: {model_name} ---")
            model_obj = model_func(X_train_vec, config["y_train_flat"]) # model_func es 'train_logistic_regression', etc.
            
            # Evaluar y obtener métricas
            metrics = modelo.evaluate_model(model_obj, X_test_vec, config["y_test_flat"], config["class_names"])
            log_entry.update(metrics)
            
            # Guardar gráfico y modelo
            graph.plot_confusion_matrix(config["y_test_flat"], model_obj.predict(X_test_vec), config["class_names"], 
                                        exp_name, save_path=f"{config['graficos_dir']}/cm_{exp_name}.png")
            
            model_path = os.path.join(config["modelos_dir"], f"{exp_name}_model.pkl")
            with open(model_path, 'wb') as f: pickle.dump(model_obj, f)

        # Guardar el resultado de este experimento
        master_results_log.append(log_entry)

    # --- 4. Guardar Reporte Final de Experimentos ---
    print("\n========================================================")
    print("FIN DE TODOS LOS EXPERIMENTOS.")
    print("========================================================")
    
    results_df = pd.DataFrame(master_results_log)
    results_df.to_csv("master_experiment_results.csv", index=False)
    
    print("Resultados guardados en 'master_experiment_results.csv'")
    print("Mejores modelos (ordenados por f1_macro):")
    print(results_df.sort_values(by='f1_macro', ascending=False))

if __name__ == "__main__":
    # Verificación de NLTK
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/sentiwordnet')
    except LookupError as e:
        print(f"Error: Paquete de NLTK no encontrado ({e}).")
        print("Ejecutando descargas de NLTK... (Esto solo pasará una vez)")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('sentiwordnet', quiet=True)
        
    run_experiment_engine()