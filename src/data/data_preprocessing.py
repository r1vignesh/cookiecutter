import os
import re
import numpy as np
import pandas as pd
import nltk
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def download_nltk_resources() -> None:
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in str(text).split() if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text: str) -> str:
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    return " ".join(text.split()).strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if len(str(df.text.iloc[i]).split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def normalized_sentence(sentence: str) -> str:
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        raise

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Failed to save data to {path}: {e}")
        raise

def main() -> None:
    try:
        download_nltk_resources()
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data preprocessing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
