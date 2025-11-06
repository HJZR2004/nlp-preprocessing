import pandas as pd
import string
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


import unicodedata
import nltk
from nltk.corpus import wordnet


from nltk.corpus import sentiwordnet as swn

import nltk
nltk.download('wordnet')

# ==============================================================================
# 1. CONSTANTES Y MAPS 
# ==============================================================================

CONTRACTION_MAP = {
    "don't": "do not",
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "let's": "let us",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "there's": "there is",
    "when's": "when is",
    "where's": "where is",
    "why's": "why is"
}

SLANG_MAP = {
    "brb": "be right back",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "idk": "i do not know",
    "smh": "shaking my head",
    "btw": "by the way",
    "imo": "in my opinion",
    "fyi": "for your information",
    "lmk": "let me know",
    "lmao": "laughing",
    "rofl": "laughing",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "wyd": "what are you doing",
    "ikr": "i know right",
    "tbh": "to be honest",
    "afk": "away from keyboard",
    "bff": "best friends forever",
    "dm": "direct message",
    "ftw": "for the win",
    "gg": "good game",
    "irl": "in real life",
    "jk": "just kidding",
    "nvm": "never mind",
    "ppl": "people",
    "sry": "sorry",
    "wbu": "what about you",
    "yw": "you are welcome",
    "xoxo": "hugs and kisses",
    "rt": "retweet",
    "fav": "favorite",
    "u": "you",
    "r": "are",
    "ll": "will",
    "ve": "have",
    "re": "are"
}


# Mapa de Emoticonos a Sentimiento
EMOTICON_SENTIMENT_MAP = {
    # Positivos
    ':-)': ' <SENT_POS> ', ':)': ' <SENT_POS> ', ':D': ' <SENT_POS> ', ':o)': ' <SENT_POS> ',
    ':]': ' <SENT_POS> ', ':3': ' <SENT_POS> ', ':c)': ' <SENT_POS> ', ':>': ' <SENT_POS> ',
    '=]': ' <SENT_POS> ', '8)': ' <SENT_POS> ', '=)': ' <SENT_POS> ', ':^D': ' <SENT_POS> ',
    '<3': ' <SENT_POS> ', 'XD': ' <SENT_POS> ',
    # Negativos
    ':-(': ' <SENT_NEG> ', ':(': ' <SENT_NEG> ', ':-c': ' <SENT_NEG> ', ':c': ' <SENT_NEG> ',
    ':-<': ' <SENT_NEG> ', ':<': ' <SENT_NEG> ', ':-[': ' <SENT_NEG> ', ':-||': ' <SENT_NEG> ',
    ':@': ' <SENT_NEG> ', ":'(": ' <SENT_NEG> ',
}


EMOLEX_MAP_POS = {'blessed', 'happy', 'joy', 'love', 'wonderful', 'amazing', 'good', 'great', 'excellent'}
EMOLEX_MAP_NEG = {'fail', 'sad', 'angry', 'hate', 'terrible', 'bad', 'worst', 'poor', 'kill', 'death'}


# ==============================================================================
# 2. PATRONES REGEX PRE-COMPILADOS (Modificados)
# ==============================================================================

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

CONTRACTION_PATTERN = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), 
                                 flags=re.IGNORECASE | re.DOTALL)
SLANG_PATTERN = re.compile(r'\b({})\b'.format('|'.join(re.escape(key) for key in SLANG_MAP.keys())), 
                           flags=re.IGNORECASE)

# --- Patrones Regex (Modificados) ---
ALLCAPS_PATTERN = re.compile(r'\b([A-Z]{2,})\b')
HASHTAG_PATTERN = re.compile(r'#(\w+)') # Captura la palabra del hashtag
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')

# --- Patrones de Normalización (Modificados) ---
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
MENTION_PATTERN = re.compile(r'@\w+')
# (Patrones de Emoji/Emoticon eliminados de aquí, se manejan en una función separada)
ORDINAL_PATTERN = re.compile(r'\b\d+(st|nd|rd|th)\b', flags=re.IGNORECASE)
DECIMAL_PATTERN = re.compile(r'\b\d+\.\d+\b')
NUMBER_PATTERN = re.compile(r'\d+')
EMPHASIS_PATTERN = re.compile(r'([!?.]){2,}')
MULTIPLE_SPACE_PATTERN = re.compile(r'\s+')
EMOTICON_PATTERN = re.compile(r'(:-\)|:\)|:D|:o\)|:\]|:3|:c\)|:>|=\]|8\)|=\)|:\^D|<3|XD|:-\(|:\(|:-c|:c|:-<|:<|:-\[|:-\|\||:@|:\'\()')




# ==============================================================================
# 3. FUNCIONES HELPER (Modificadas)
# ==============================================================================

def _helper_expand_match(match):
    # (Tu función sin cambios)
    match_str = match.group(0)
    first_char = match_str[0]
    expanded_contraction = CONTRACTION_MAP.get(match_str.lower())
    if not expanded_contraction:
        return match_str
    if first_char.isupper():
        expanded_contraction = expanded_contraction[0].upper() + expanded_contraction[1:]
    if match_str.lower() == "i'm":
        return "I am"
    return expanded_contraction

def _helper_normalize_slang(match):
    # (Tu función sin cambios)
    match_str = match.group(0)
    return SLANG_MAP.get(match_str.lower(), match_str)

def get_wordnet_pos(nltk_tag):
    """
    NUEVA FUNCIÓN HELPER
    Mapea tags de NLTK (PoS) a tags que WordNet Lemmatizer entiende.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def replace_emoticons_with_sentiment(text):
    """
    Reemplaza Emoticonos en el texto con sus etiquetas de sentimiento.
    """
    # Ordenar por longitud (más largos primero) para evitar reemplazos parciales (ej. ':-)' antes que ':)' )
    sorted_emoticons = sorted(EMOTICON_SENTIMENT_MAP.keys(), key=len, reverse=True)
    for emoticon in sorted_emoticons:
        text = text.replace(emoticon, EMOTICON_SENTIMENT_MAP[emoticon])
    return text

def get_sentiment_tag(lema, wn_tag):
    """
    Obtiene una etiqueta de sentimiento para un lema usando EmoLex y SentiWordNet.
    """
    # 1. Comprobar léxicos rápidos (EmoLex) primero
    if lema in EMOLEX_MAP_POS:
        return '<SENT_POS>'
    if lema in EMOLEX_MAP_NEG:
        return '<SENT_NEG>'


    # Traducir el OBJETO (wordnet.ADJ) a la LETRA ('a')
    if wn_tag == wordnet.ADJ:
        swn_tag = 'a'
    elif wn_tag == wordnet.ADV:
        swn_tag = 'r'
    elif wn_tag == wordnet.VERB:
        swn_tag = 'v'
    else: 
        swn_tag = 'n'
    synsets = list(swn.senti_synsets(lema, swn_tag))
    
    if not synsets:
        return '<SENT_NEU>' 
    
    s = synsets[0]
    pos_score = s.pos_score()
    neg_score = s.neg_score()
    
    # 3. Determinar etiqueta basada en puntuaciones
    if (pos_score > neg_score) and (pos_score >= 0.25):
        return '<SENT_POS>'
    elif (neg_score > pos_score) and (neg_score >= 0.25):
        return '<SENT_NEG>'
    else:
        return '<SENT_NEU>'
    

# ==============================================================================
# 4. FUNCIONES DE LIMPIEZA PRINCIPALES (Modificadas)
# ==============================================================================

def load_data(file_name):
    """Carga datos desde el archivo TXT."""
    df = pd.read_csv(file_name, sep='\t', header=None, encoding='utf-8')
    X = df[3].values.reshape(-1, 1)
    y = df[2].str.lower()
    y = y.map({'negative': 0, 'positive': 1, 'neutral': 2, 'objective-or-neutral': 3, 'objective': 4})
    y = y.values.reshape(-1, 1)
    return X, y

def normalize_slang(text):
    """Normaliza slang usando la regex pre-compilada."""
    return SLANG_PATTERN.sub(_helper_normalize_slang, text)

def expand_contractions(text):
    """Expande contracciones usando la regex pre-compilada."""
    return CONTRACTION_PATTERN.sub(_helper_expand_match, text)


def get_zipf_tokens(text):
    """
    Crea una lista de tokens "semi-limpios" (lemas puros)
    específicamente para graficar la Ley de Zipf.
    
    Este pipeline:
    1. Limpia ruido (contracciones, regex, etc.).
    2. Lematiza con PoS.
    3. DEVUELVE solo los lemas (palabras), eliminando puntuación y etiquetas.
    """
    
    # --- Pasos 1-4: Limpieza inicial (reutilizados) ---
    text = unicodedata.normalize('NFC', text)
    text = text.replace(u'\xa0', u' ')
    text = replace_emoticons_with_sentiment(text) # Reemplaza :) con <SENT_POS>
    text = expand_contractions(text)
    text = normalize_slang(text)
    text = clean_with_regex(text) # Reemplaza #tag con 'tag <HASHTAG>'
    
    # --- Paso 5: Tokenización ---
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # --- Paso 6: Lematización y FILTRADO para Zipf ---
    pos_tags = nltk.pos_tag(tokens)
    
    zipf_tokens = []
    for word, tag in pos_tags:
        
        # Lematizar
        wn_tag = get_wordnet_pos(tag)
        lema = LEMMATIZER.lemmatize(word, pos=wn_tag)
        
        # --- FILTRADO ESPECIAL PARA ZIPF ---
        
        # a. Ignorar tokens de puntuación
        if lema in string.punctuation:
            continue
            
        # b. Ignorar TODOS los tokens especiales (los que creamos)
        if lema.startswith('<') and lema.endswith('>'):
            continue
            

        # d. Guardar solo el lema (la palabra pura)
        zipf_tokens.append(lema)
            
    return zipf_tokens



def clean_with_regex(text):
    """
    Limpia el texto usando regex (versión única y limpia).
    """
    text = ALLCAPS_PATTERN.sub(r' \1 <ALLCAPS> ', text)
    text = REPEATED_CHAR_PATTERN.sub(r'\1\1 <ELONGATED> ', text)
    text = HASHTAG_PATTERN.sub(r' \1 <HASHTAG> ', text)
    text = URL_PATTERN.sub(' <URL> ', text)
    text = EMAIL_PATTERN.sub(' <EMAIL> ', text)
    text = MENTION_PATTERN.sub(' <MENTION> ', text)
    text = ORDINAL_PATTERN.sub(' <NUMBER> ', text)
    text = DECIMAL_PATTERN.sub(' <NUMBER> ', text)
    text = NUMBER_PATTERN.sub(' <NUMBER> ', text)
    text = EMPHASIS_PATTERN.sub(r' <EMPHASIS> ', text)
    text = MULTIPLE_SPACE_PATTERN.sub(' ', text)
    return text.strip()



# ==============================================================================
# 5. LOGICA DE PIPELINES (SIMPLE O ENRIQUECIDO)
# ==============================================================================
def _pipeline_simple(text, remove_stopwords=False, stemming=False, lemmatization=True):
    """
    Pipeline de limpieza 'simple': Rápido, sin PoS, sin SentiWordNet.
    """
    # Pasos 1-4: Normalización básica
    text = unicodedata.normalize('NFC', text)
    text = text.replace(u'\xa0', u' ')
    text = replace_emoticons_with_sentiment(text) # Sigue siendo útil
    text = expand_contractions(text)
    text = normalize_slang(text)
    text = clean_with_regex(text) # Normaliza URLs, @, #, etc.
    
    # Paso 5: Minúsculas y Tokenización
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Paso 6: Loop de limpieza simple (rápido)
    simple_tokens = []
    for word in tokens:
        # a. Ignorar puntuación
        if word in string.punctuation:
            continue
            
        # b. Conservar tokens especiales
        if word.startswith('<') and word.endswith('>'):
            simple_tokens.append(word)
            continue
            
        # c. Manejar Stopwords
        if remove_stopwords and word in STOP_WORDS:
            continue
            
        # d. Aplicar Stemming O Lemmatization (no ambas)
        if stemming:
            token = STEMMER.stem(word)
        elif lemmatization:
            token = LEMMATIZER.lemmatize(word) # Lematización simple (sin PoS)
        else:
            token = word # Dejar la palabra como está
        
        simple_tokens.append(token)
            
    # 7. Re-ensamblaje
    final_text = ' '.join(simple_tokens)
    final_text = MULTIPLE_SPACE_PATTERN.sub(' ', final_text).strip()
    return final_text


def _pipeline_enriquecido(text, remove_stopwords=False):
    """
    Pipeline 'enriquecido': Lento, PoS, SentiWordNet, Corrección Ortográfica.
    (Esta era tu 'full_text_pipeline' anterior, ahora con parámetros).
    """
    
    # Pasos 1-4
    text = unicodedata.normalize('NFC', text)
    text = text.replace(u'\xa0', u' ')
    text = replace_emoticons_with_sentiment(text)
    text = expand_contractions(text)
    text = normalize_slang(text)
    text = clean_with_regex(text)
    
    # Paso 5
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Paso 6: Lematización PoS y Enriquecimiento
    pos_tags = nltk.pos_tag(tokens)
    
    enriched_tokens = []
    
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lema = LEMMATIZER.lemmatize(word, pos=wn_tag)
        
        # a. Ignorar puntuación
        if lema in string.punctuation:
            continue
            
        # b. Conservar tokens especiales
        if lema.startswith('<') and lema.endswith('>'):
            enriched_tokens.append(lema)
            continue
        
        # c. ¡NUEVO! Manejar Stopwords
        if remove_stopwords and lema in STOP_WORDS:
            continue

        # e. Enriquecimiento sentimental
        sentiment_tag = get_sentiment_tag(lema, wn_tag)
        enriched_tokens.append(lema)
        enriched_tokens.append(sentiment_tag)
            
    # 7. Re-ensamblaje
    final_text = ' '.join(enriched_tokens)
    final_text = MULTIPLE_SPACE_PATTERN.sub(' ', final_text).strip()
    return final_text

# ==============================================================================
# 6. FUNCION DE PIPELINE PRINCIPAL
# ==============================================================================

def full_text_pipeline(text, 
                       pipeline_type='enriquecido',  
                       remove_stopwords=False,       
                       stemming=False,             
                       lemmatization=True):
    """
    Función principal de limpieza que enruta al pipeline solicitado.
    
    Parámetros:
    - text (str): El texto crudo.
    - pipeline_type (str): 'enriquecido' (PoS, SentiWordNet) o 'simple' (rápido).
    - remove_stopwords (bool): Si es True, elimina stop words.
    - stemming (bool): Si es True (y pipeline='simple'), usa stemming.
    - lemmatization (bool): Si es True (y pipeline='simple'), usa lematización simple.
    """
    
    if pipeline_type == 'enriquecido':
        # El pipeline 'enriquecido' usa PoS-Lemmatization por defecto.
        # Ignora los parámetros 'stemming' y 'lemmatization'.
        return _pipeline_enriquecido(text, 
                                     remove_stopwords=remove_stopwords)
    
    elif pipeline_type == 'simple':
        # El pipeline 'simple' no puede hacer 'spell_check' (requiere PoS).
        return _pipeline_simple(text, 
                                remove_stopwords=remove_stopwords, 
                                stemming=stemming, 
                                lemmatization=lemmatization)
    else:
        raise ValueError("pipeline_type debe ser 'enriquecido' o 'simple'")
    
