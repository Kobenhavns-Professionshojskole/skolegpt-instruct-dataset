import os
from datetime import datetime

import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from .config import config

load_dotenv()

auth_key = os.environ["DEEPL_API_KEY"]


def translate_dataset(df: pl.DataFrame, save_freq: int) -> pl.DataFrame:
    """Translation pipeline wiht DeepL."""
    df = translate_system_prompts(df)
    df = translate_examples(df, save_freq=save_freq)
    return df


def translate_with_deepl(text: str, target_lang: str):
    """
    Translates text using the DeepL API.

    Args:
        text (str or list of str): The text to be translated.
        target_lang (str): The target language code (e.g., "DE" for German).
        auth_key (str): Your DeepL API authentication key.

    Returns:
        str: The translated text or an error message.
    """
    # URL for the DeepL API
    url = "https://api.deepl.com/v2/translate"

    # Define the request headers
    headers = {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "Content-Type": "application/json",
    }

    # Convert single string input to a list if needed
    if isinstance(text, str):
        text = [text]

    # Define the data payload
    data = {"text": text, "target_lang": target_lang}

    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        # Return the translated text
        translated_text = response.json()["translations"][0]["text"]
        return translated_text
    else:
        # Return an error message if the request was not successful
        return f"Error: {response.status_code} - {response.text}"


def translate_system_prompts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Translates unique system prompts in a DataFrame to Danish using DeepL.

    This function iterates over unique system prompts found in the specified column of the DataFrame.
    Each prompt is translated to Danish, and these translations are then used to update the original DataFrame.

    Args:
    - dataframe (pl.DataFrame): A DataFrame containing a column 'system_prompt' with prompts to be translated.

    Returns:
    - pl.DataFrame: A DataFrame with the 'system_prompt' column updated with Danish translations.
    """

    # Extract unique system prompts from the DataFrame
    unique_system_prompts = df["system_prompt"].unique().to_list()

    # Dictionary to hold translations
    translated_system_prompts = {}

    # Translate each unique system prompt to Danish
    for system_prompt in unique_system_prompts:
        translated_system_prompts.update(
            {system_prompt: translate_with_deepl(system_prompt, "DA")}
        )

    # Update the DataFrame with translated prompts
    df = df.with_columns(pl.col("system_prompt").replace(translated_system_prompts))

    return df


def translate_examples(df: pl.DataFrame, save_freq: int) -> pl.DataFrame:
    date = datetime.now().isoformat()[:19]

    n_translations = 0
    translated_rows = []
    try:
        for r in tqdm(df.to_dicts(), total=len(df)):
            r["question"] = translate_with_deepl(r["question"], "DA")
            r["response"] = translate_with_deepl(r["response"], "DA")
            translated_rows.append(r)
            n_translations += 1
            if (n_translations % save_freq) == 0:
                pl.DataFrame(translated_rows).write_parquet(
                    config.data_dir / f"save_{date}.parquet"
                )
    except:
        breakpoint()

    return pl.DataFrame(translated_rows)
