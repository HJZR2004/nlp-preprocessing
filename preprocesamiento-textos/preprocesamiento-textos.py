#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de preprocesamiento de texto para PLN (Análisis de Sentimientos).
Limpia y normaliza tuits en un solo archivo, usando solo funciones.
"""

import numpy as np
import pandas as pd
import string
import re
# import html  <-- ELIMINADO

# --- NLTK Imports ---
import nltk
# Descomenta las siguientes líneas la primera vez que ejecutes el script
# para descargar los paquetes necesarios:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # Necesario para wordnet

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================================================================
# 1. CONSTANTES Y MAPS
# ==============================================================================

CONTRACTION_MAP = {
    "don't": "do not", "can't": "can not", "won't": "will not",
    "shan't": "shall not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
    "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "mightn't": "might not",
    "mustn't": "must not", "let's": "let us", "that's": "that is",
    "who's": "who is", "what's": "what is", "here's": "here is",
    "there's": "there is", "when's": "when is", "where's": "where is",
    "why's": "why is",
}

SLANG_MAP = {
    "brb": "be right back", "lol": "laughing out loud", "omg": "oh my god",
    "ttyl": "talk to you later", "idk": "i do not know",
    "smh": "shaking my head", "btw": "by the way", "imo": "in my opinion",
    "fyi": "for your information", "lmk": "let me know", "lmao": "laughing",
    "rofl": "laughing", "thx": "thanks", "ty": "thank you",
    "np": "no problem", "wyd": "what are you doing", "ikr": "i know right",
    "tbh": "to be honest", "afk": "away from keyboard",
    "bff": "best friends forever", "dm": "direct message",
    "ftw": "for the win", "gg": "good game", "irl": "in real life",
    "jk": "just kidding", "nvm": "never mind", "ppl": "people",
    "sry": "sorry", "wbu": "what about you", "yw": "you are welcome",
    "xoxo": "hugs and kisses", "rt": "retweet", "fav": "favorite",
    "u": "you", "r": "are", "ll": "will", "ve": "have", "re": "are"
}

# ==============================================================================
# 2. OBJETOS GLOBALES (Para eficiencia)
# ==============================================================================
# --- Inicializadores de NLTK (se crean una sola vez) ---
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# --- Traductor de Puntuación (se crea una sola vez) ---
# Excluye '<' y '>' para proteger nuestros tokens (ej. <url>)
PUNCT_TO_REMOVE = string.punctuation.replace('<', '').replace('>', '')
PUNCT_TRANSLATOR = str.maketrans('', '', PUNCT_TO_REMOVE)

# --- Patrones Regex Pre-compilados (mejora de rendimiento) ---
# Compilar las regex una vez es mucho más rápido que dentro de un bucle.

# Para contracciones
CONTRACTION_PATTERN = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), 
                                 flags=re.IGNORECASE | re.DOTALL)

# Para slang (usando \b para "word boundary" y así "u" no reemplace "user")
SLANG_PATTERN = re.compile(r'\b({})\b'.format('|'.join(re.escape(key) for key in SLANG_MAP.keys())), 
                           flags=re.IGNORECASE)

# Patrones para 'sentence_regex'
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
EMOJI_PATTERN = re.compile(
    "["u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+", flags=re.UNICODE)
EMOTICON_PATTERN = re.compile(r"""(?:
    (?:[:;=8][\-']?[\)\]\(\[dDpP\/\\S\|oO\*])|
    (?:O[:;=8][\-']?[\)])|
    (?:>[:;=8][\-']?[\)])|
    (?:[:=]3)|(?:<3+)|(?:\b[xX]D\b)|(?::'\()
)""", flags=re.VERBOSE)
ORDINAL_PATTERN = re.compile(r'\b\d+(st|nd|rd|th)\b', flags=re.IGNORECASE)
DECIMAL_PATTERN = re.compile(r'\b\d+\.\d+\b')
NUMBER_PATTERN = re.compile(r'\d+')
EMPHASIS_PATTERN = re.compile(r'([!?.]){2,}')
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')
MULTIPLE_SPACE_PATTERN = re.compile(r'\s+')

# ==============================================================================
# 3. FUNCIONES HELPER (para re.sub)
# ==============================================================================

def _helper_expand_match(match):
    """
    Función helper global. re.sub la llamará para cada contracción.
    (Esta reemplaza la función anidada 'expand_match').
    """
    match_str = match.group(0)
    first_char = match_str[0]
    expanded_contraction = CONTRACTION_MAP.get(match_str.lower())
    
    if not expanded_contraction:
        return match_str  # No debería pasar, pero por seguridad
    
    # Preservar mayúscula (ej. "I'm" -> "I am", no "i am")
    if first_char.isupper():
        expanded_contraction = expanded_contraction[0].upper() + expanded_contraction[1:]
    
    # Manejar "I'm" -> "I am"
    if match_str.lower() == "i'm":
        return "I am"
    
    return expanded_contraction

def _helper_normalize_slang(match):
    """
    Función helper global. re.sub la llamará para cada palabra de slang.
    """
    match_str = match.group(0)
    return SLANG_MAP.get(match_str.lower(), match_str)

# ==============================================================================
# 4. FUNCIONES DE LIMPIEZA PRINCIPALES
# ==============================================================================

def vectorize(file_name):
    """
    Carga los datos desde un archivo.
    ¡CORRECCIÓN! Ya no convierte X a minúsculas aquí.
    """
    df = pd.read_csv(file_name, sep='\t', header=None, encoding='utf-8')
    
    # X se queda con mayúsculas/minúsculas para que 'expand_contractions' funcione
    X = df[3]
    X = np.array(X).reshape(-1, 1)

    # Y sí se puede procesar
    y = df[2].str.lower()
    y = y.map({'negative': 0, 'positive': 1, 'neutral': 2, 'objective-or-neutral': 3, 'objective': 4})
    y = np.array(y).reshape(-1, 1)
    return X, y

def normalize_slang(text):
    """
    Normaliza slang usando la regex pre-compilada.
    (Versión mejorada de tu función original).
    """
    return SLANG_PATTERN.sub(_helper_normalize_slang, text)

def expand_contractions(text):
    """
    Expande contracciones usando la regex pre-compilada y la función helper.
    (Tu función original, pero ahora sin anidamiento).
    """
    return CONTRACTION_PATTERN.sub(_helper_expand_match, text)

def remove_punctuation(text):
    """Quita puntuación usando el traductor pre-compilado."""
    return text.translate(PUNCT_TRANSLATOR)

def sentence_lemmatizer(text, remove_stopwords=False):
    """Lematiza texto, usando los objetos NLTK globales."""
    words = text.split()
    if remove_stopwords:
        lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOP_WORDS]
    else:
        lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def sentence_regex(text):
    """
    Limpia el texto usando todos los patrones de regex estructurales.
    (Tu función original, pero usando patrones globales pre-compilados).
    """
    text = URL_PATTERN.sub('<url>', text)
    text = EMAIL_PATTERN.sub('<email>', text)
    text = MENTION_PATTERN.sub('<mention>', text)
    text = HASHTAG_PATTERN.sub('<hashtag>', text)
    text = EMOJI_PATTERN.sub(r' <emoji> ', text)
    text = EMOTICON_PATTERN.sub(r' <emoji> ', text)
    text = ORDINAL_PATTERN.sub('<number>', text)
    text = DECIMAL_PATTERN.sub('<number>', text)
    text = NUMBER_PATTERN.sub('<number>', text)
    text = EMPHASIS_PATTERN.sub(r' <emphasis> ', text)
    text = REPEATED_CHAR_PATTERN.sub(r'\1\1', text)
    text = MULTIPLE_SPACE_PATTERN.sub(' ', text)
    
    return text.strip()

# =C============================================================================
# 5. EJECUCIÓN DEL SCRIPT (main)
# ==============================================================================

def main():
    """Función principal del script."""
    # Asegúrate de haber descargado los paquetes de NLTK
    try:
        stopwords.words('english')
        LEMMATIZER.lemmatize('test')
    except LookupError:
        print("Error: Paquetes de NLTK no encontrados.")
        print("Por favor, ejecuta las descargas de NLTK (líneas 27-29).")
        return # Salir si NLTK no está listo
        
    print("Cargando y procesando datos...")
    X_train_raw, y_train = vectorize('raw-train.txt')
    X_test_raw, y_test = vectorize('raw-test.txt')

    print("--- RAW (Últimos 10) ---")
    print(X_train_raw[-10:])

    # Procesamiento de Train
    X_train_clean = []
    for i in range(len(X_train_raw)):
        text = X_train_raw[i][0] # 1. Obtener el texto
        
        # --- ORDEN DEL PIPELINE (CRUCIAL) ---
        
        # 2. Limpiar ruido (ej. \xa0)
        text = text.replace(u'\xa0', u' ')
        
        # 3. Expandir contracciones (ej. "I'm" -> "I am")
        #    Debe ir ANTES de minúsculas y ANTES de quitar puntuación.
        text = expand_contractions(text)
        
        # 4. Normalizar slang (ej. "u" -> "you")
        #    Debe ir ANTES de minúsculas.
        text = normalize_slang(text)
        
        # 5. Aplicar Regex (URLs, @, #, emojis, números)
        text = sentence_regex(text)
        
        # 6. Convertir a minúsculas (¡CORRECCIÓN! Ahora es seguro)
        text = text.lower()
        
        # 7. Quitar puntuación (protege <tokens>)
        text = remove_punctuation(text)
        
        # 8. Lematizar (y NO quitar stopwords)
        text = sentence_lemmatizer(text, remove_stopwords=False)
        
        # 9. Limpieza final de espacios (la lematización puede añadir)
        text = MULTIPLE_SPACE_PATTERN.sub(' ', text).strip()
        
        X_train_clean.append([text])
    
    # Procesamiento de Test
    X_test_clean = []
    for i in range(len(X_test_raw)):
        text = X_test_raw[i][0]
        # text = html.unescape(text) <-- ELIMINADO
        text = text.replace(u'\xa0', u' ')
        text = expand_contractions(text)
        text = normalize_slang(text)
        text = sentence_regex(text)
        text = text.lower()
        text = remove_punctuation(text)
        text = sentence_lemmatizer(text, remove_stopwords=False)
        text = MULTIPLE_SPACE_PATTERN.sub(' ', text).strip()
        X_test_clean.append([text])

    # Convertir listas a arrays de numpy
    X_train = np.array(X_train_clean)
    X_test = np.array(X_test_clean)
    
    print("\n--- LIMPIO (Últimos 10) ---")
    print(X_train[-10:])
    
    print("\n--- Formas Finales ---")
    print(f"X_train (limpio): {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test (limpio): {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print("\nProcesamiento completado.")

    
if __name__ == "__main__":
    main()