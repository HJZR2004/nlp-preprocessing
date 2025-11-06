import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Input 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers

# --- Modelo 1: Naive Bayes ---
def train_naive_bayes(X_train_vec, y_train):
    print("Entrenando Modelo: Naive Bayes Multinomial...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

# --- Modelo 1.B: Naive Bayes para OHE ---
def train_bernoulli_nb(X_train_vec, y_train):
    """
    Entrena Naive Bayes de Bernoulli (para características binarias/OHE).
    """
    print("Entrenando Modelo: Naive Bayes Bernoulli...")
    model = BernoulliNB()
    model.fit(X_train_vec, y_train)
    return model

# --- Modelo 2: Regresión Logística ---
def train_logistic_regression(X_train_vec, y_train):
    print("Entrenando Modelo: Regresión Logística (con Class Weight)...")
    model = LogisticRegression(solver='liblinear', max_iter=1000, 
                               class_weight='balanced')
    model.fit(X_train_vec, y_train)
    return model


# --- Modelo 3: Random Forest ---
def train_random_forest(X_train_vec, y_train):
    """
    Entrena un Clasificador Random Forest.
    """
    print("Entrenando Modelo: Random Forest (con Class Weight)...")
    model = RandomForestClassifier(
        n_estimators=100,         # Número de árboles (buen default)
        class_weight='balanced',  # Maneja el desbalance de clases
        n_jobs=-1,                # Usa todos los núcleos de la CPU
        random_state=42           # Para resultados reproducibles
    )
    model.fit(X_train_vec, y_train)
    return model




# --- Modelo 3: Red Neuronal (Arquitectura Equilibrada) ---
def build_nn_model(input_dim, num_classes):
    """
    Define una arquitectura "embudo" más equilibrada.
    Es menos agresiva y está mejor regularizada.
    """
    print("Construyendo Modelo: Red Neuronal Densa (Arquitectura Equilibrada)...")
    l2_rate = 0.001 # Tasa de regularización más estándar y menos agresiva

    
    # 1. Capa de entrada
    inputs = Input(shape=(input_dim,))
    
    # 2. Primera capa oculta (más pequeña que 1024)
    x = Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    x = Dropout(0.5)(x) # Dropout consistente

    # 3. Segunda capa oculta
    x = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = Dropout(0.5)(x) # Dropout consistente

    # 4. Tercera capa oculta (para profundidad)
    x = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = Dropout(0.5)(x) # Dropout consistente

    # 5. Capa de salida
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Crear el modelo
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_nn_model(X_train_vec, y_train_cat, X_test_vec, y_test_cat, 
                   num_features, num_classes, class_weights_dict):
    """
    Construye y entrena la Red Neuronal con Early Stopping y Class Weights.
    """
    model = build_nn_model(input_dim=num_features, num_classes=num_classes)
    
    X_train_dense = X_train_vec.toarray()
    X_test_dense = X_test_vec.toarray()
    
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, 
                                  restore_best_weights=True)

    print("Entrenando Red Neuronal con Early Stopping y Class Weights...")
    
    # --- CAMBIO AQUÍ ---
    # Aumenta las épocas; EarlyStopping se encargará de parar
    history = model.fit(X_train_dense, y_train_cat,
              epochs=100, # Ponemos 100, pero EarlyStopping parará antes
              batch_size=32,
              validation_data=(X_test_dense, y_test_cat),
              callbacks=[early_stopper],
              class_weight=class_weights_dict)
    
    # Devuelve el objeto History, no el diccionario .history
    return model, history

# --- Evaluación ---
def evaluate_model(model, X_test_vec, y_test, class_names):
    """
    Evalúa un modelo de scikit-learn.
    Imprime el reporte Y DEVUELVE un diccionario de métricas.
    """
    y_pred = model.predict(X_test_vec)
    
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Devuelve las métricas clave para el log
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro')
    }

def evaluate_nn_model(model, X_test_vec, y_test_cat, class_names):
    """
    Evalúa un modelo de Keras (Red Neuronal).
    Imprime el reporte Y DEVUELVE un diccionario de métricas.
    """
    X_test_dense = X_test_vec.toarray()
    
    y_pred_probs = model.predict(X_test_dense)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # y_test_cat es one-hot, necesitamos convertirlo a etiquetas
    y_test_labels = np.argmax(y_test_cat, axis=1)
    
    print("\n--- Reporte de Clasificación (Red Neuronal) ---")
    print(classification_report(y_test_labels, y_pred, target_names=class_names))
    
    # Devuelve las métricas clave para el log
    return {
        'accuracy': accuracy_score(y_test_labels, y_pred),
        'f1_macro': f1_score(y_test_labels, y_pred, average='macro'),
        'precision_macro': precision_score(y_test_labels, y_pred, average='macro'),
        'recall_macro': recall_score(y_test_labels, y_pred, average='macro')
    }