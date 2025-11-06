# vectorizar.py

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def vectorize_tfidf(X_train_clean, X_test_clean, max_features=5000):
    """
    Crea, entrena y aplica un vectorizador TF-IDF.
    Mide la importancia de una palabra (frecuencia).
    """
    print(f"Vectorizando con TF-IDF (max_features={max_features})...")
    
    # 1. Inicializar el vectorizador
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    
    # 2. Entrenar (fit) y transformar (transform) los datos de entrenamiento
    X_train_vec = tfidf_vectorizer.fit_transform(X_train_clean)
    
    # 3. Solo transformar (transform) los datos de prueba
    X_test_vec = tfidf_vectorizer.transform(X_test_clean)

    return X_train_vec, X_test_vec, tfidf_vectorizer

def vectorize_bow(X_train_clean, X_test_clean, max_features=5000):
    """
    Crea, entrena y aplica un vectorizador Bag-of-Words (Conteo).
    Mide la frecuencia de una palabra.
    """
    print(f"Vectorizando con Bag-of-Words (Conteo, max_features={max_features})...")
    
    # 1. Inicializar
    # binary=False (por defecto) significa que cuenta las ocurrencias (ej. 1, 2, 3...)
    bow_vectorizer = CountVectorizer(max_features=max_features, binary=False)
    
    # 2. Entrenar y transformar
    X_train_vec = bow_vectorizer.fit_transform(X_train_clean)
    
    # 3. Transformar
    X_test_vec = bow_vectorizer.transform(X_test_clean)
    
    return X_train_vec, X_test_vec, bow_vectorizer

def vectorize_ohe(X_train_clean, X_test_clean, max_features=5000):
    """
    Crea, entrena y aplica un vectorizador One-Hot Encoding (Binario).
    Mide solo la presencia (1) o ausencia (0) de una palabra.
    """
    print(f"Vectorizando con One-Hot Encoding (Binario, max_features={max_features})...")
    
    # 1. Inicializar
    # ¡binary=True es la clave! Convierte el conteo en un indicador binario (0 o 1).
    ohe_vectorizer = CountVectorizer(max_features=max_features, binary=True)
    
    # 2. Entrenar y transformar
    X_train_vec = ohe_vectorizer.fit_transform(X_train_clean)
    
    # 3. Transformar
    X_test_vec = ohe_vectorizer.transform(X_test_clean)
    
    return X_train_vec, X_test_vec, ohe_vectorizer