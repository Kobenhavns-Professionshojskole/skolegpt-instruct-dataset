import textwrap
import time
from pathlib import Path

import datasets
import pandas as pd
import polars as pl

from .config import config


def sample_and_print_example(df):
    """Sample random example and print it."""

    s = df.sample(1)
    print("ID")
    print(s.get_column("id")[0])
    print()
    print("System Prompt:\n".upper())
    print(textwrap.fill(s.get_column("system_prompt")[0]))
    print()
    print("Question:\n".upper())
    print(textwrap.fill(s.get_column("question")[0]))
    print()
    print("Response:\n".upper())
    print(textwrap.fill(s.get_column("response")[0]))
    return s


def analyse_pre_and_postfixes(df: pl.DataFrame):
    """Analyse frequency of common pre- and postfixes."""

    # Analysing common question/instruction pre- and postfixes
    print("--- Prefixes ---")
    for prefix in config.common_prefixes:
        freq = (
            df["question"]
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("^{prefix}".format(prefix=prefix.lower()))
            .sum()
        )
        norm_freq = round(freq / len(df) * 100, 5)
        print(f"Normalized Freq. {norm_freq}% | Freq.: {freq} | Term: '{prefix}' ")

    print("\n--- Postfixes ---")
    for postfix in config.common_postfixes:
        freq = (
            df["question"]
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("{prefix}$".format(prefix=postfix.lower()))
            .sum()
        )
        norm_freq = round(freq / len(df) * 100, 5)
        print(f"Normalized Freq. {norm_freq}% | Freq.: {freq} | Term: '{postfix}' ")


def print_elapsed_time(step, start_time):
    elapsed = time.time() - start_time
    print(f"{step} completed in {elapsed:.2f} seconds")


def return_filter_char_list(df: pl.DataFrame) -> list[str]:
    filter_chars = """
        ビクター・シュムは、シンガポールのエネルギーアナリストであり、プルドーベイ油田の閉鎖は「トレーダーにとって驚きの種をまいた」と述べた。
        นักวิเคราะห์พลังงานในสิงคโปรค์กล่าวว่าการปิดตัวของแหล่งน้ำมันอ่าว
        ได้สร้างความประหลาดใจต่อผู้ค้า
        アーノルド夫人の要求に関する調査で過失が申し立てられたムーア氏の民事訴訟において
        もは告発された
        過失が申し立てられた
        知府
        អ្នកជាប់ចោទត្រូវបានគេចោទប្រកាន់ពីការរំលោភសិទ្ធិកម្មសិទ្ធិបញ្ញាលើតួអង្គវិរៈបុរសម៉
        ាវ៉ុលនិងកម្មសិទ្ធបញ្ញាដែលបង្កើតដោយស្ទេនលី។
        彼は秘密を守るために、国家安全保障局とリンクした秘密の任務を受け、スパイ活動を行った。
        អ្នកជាប់ចោទត្រូវបានគេចោទប្រកាន់ពីការរំលោភសិទ្ធិកម្មសិទ្ធិបញ
        ្ញាលើតួអង្គវិរៈបុរសម៉ាវ៉ុលនិងកម្មសិទ្ធបញ្ញាដែលបង្កើតដោយស្ទេនលី។
        では番目のファイルは
        ઘાસના ઢોળેલા ખેતરમાં ઘેટાચરાવવાનું ટોળું
        Парче дърво може да отвори вратата
        তো শেহতীয়াকৈ কুছ মীঠা হো যায়ে অলপ মীঠা খাই লোৱা যাওঁক
        শীৰ্ষকেৰে প্ৰচাৰ কৰা বিজ্ঞাপন সমূহে বহুল জনপ্ৰিয়তা পাইছে। পৰিয়ালৰ
        সকলোটিলৈ লক্ষ্য ৰাখি সৃষ্টি কৰা এই বিজ্ঞাপন সমূহে দৰ্শকক আকৰ্ষিত কৰিছে
        আৰু এতিয়া ভাৰতীয় উৎসৱ পাৰ্বণ সমূহতো ডেইৰী মিল্কে আকৰ্ষণীয় স্থান
        পাবলৈ লৈছে
        Ìíúòṣì
        영국의동굴목록이며영국에서가장크고깊은동굴에대한정보를포함하고있습니다
        های ایرانی عادت کنیم.   سرعت فیلم پایین بود و چندین
        بار وسط فیلم ساعتم رو نگاه کردم.  بازی های بهتری از بازیگرهای فیلم
        دیده بودیم و بازی ها هم در سطح متوسطی بود.  در کل ۲.۵ از
        نظر شما در مورد داستان، فیلمنامه، دیالوگ ها و موضوع
        فیلم  هت‌تریک چیست؟
    """
    filter_chars = filter_chars.replace("\n", "").replace(" ", "")
    filter_chars = sorted(list(set(filter_chars)))
    most_common_chars = get_n_most_common_chars(df=df.sample(config.n_total), n=10000)
    filter_chars = [x for x in filter_chars if x not in most_common_chars]
    return filter_chars


def get_n_most_common_chars(df: pl.DataFrame, n: int = 10000) -> list[str]:
    def flatten(xss):
        return [x for xs in xss for x in xs]

    print("Starting to flatten and combine characters from columns.")
    all_chrs = flatten(
        df["question"].str.to_lowercase().to_list()
        + df["response"].str.to_lowercase().to_list()
        + df["system_prompt"].str.to_lowercase().to_list()
    )
    print("Completed flattening and combining characters.")

    print("Starting to count character occurrences.")
    char_count = pl.Series(all_chrs).value_counts()
    print("Completed counting character occurrences.")

    print("Converting to pandas DataFrame and sorting.")
    char_frq = char_count.to_pandas().sort_values(by="count")
    print("Completed conversion and sorting.")

    print(f"Selecting characters with occurrences more than {n}.")
    most_common_chars = char_frq[char_frq["count"] > n][""].tolist()
    print("Completed selecting most common characters.")

    return most_common_chars


# ---------------------------------------------------------------------------- #
#                                  Path Stuff                                  #
# ---------------------------------------------------------------------------- #


def create_directory_if_not_exists(directory_path):
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory_path}")
    else:
        pass


def load_parquet_file_with_polars(file_path):
    """
    Loads a Parquet file using Polars.

    Args:
    file_path (str): Path to the Parquet file to be loaded.

    Returns:
    polars.DataFrame: Contents of the Parquet file.

    Raises:
    FileNotFoundError: If the Parquet file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    return pl.read_parquet(file_path)


def count_total_characters(df):
    """
    Counts the total number of characters in the columns 'question', 'response' of a Polars DataFrame.

    Parameters:
    df (polars.DataFrame): The dataframe to process.

    Returns:
    int: The total number of characters in the specified columns.
    """
    # Check if the specified columns exist in the DataFrame
    for column in ["question", "response"]:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

    # Calculate the sum of character lengths for each specified column
    total_characters = sum(
        df.select(
            [
                pl.col(column).str.len_chars().sum()
                for column in ["question", "response"]
            ]
        )
    )

    return total_characters.item()


def estimate_translated_dataset_size_given_budget(
    df: pl.DataFrame,
    max_budget_in_eur: int,
):
    """
    Estimates the number of dataset entries that can be translated by DeepL within a given budget.

    This function calculates the total number of characters across specific columns ('question',
    'response', and 'system_prompt') in a Polars DataFrame, and then estimates how many dataset
    entries can be translated based on a specified budget and a fixed translation cost.

    Parameters:
    df (polars.DataFrame): The dataframe containing the dataset to be translated.
                           It must have the columns 'question', 'response', and 'system_prompt'.
    max_budget_in_eur (int): The maximum budget available for translation, in euros.

    Returns:
    int: The estimated number of dataset entries that can be translated within the given budget.

    Note:
    - The function assumes uniform distribution of characters across dataset entries.
    """
    price_pr_million_chars = 20
    total_char_count = count_total_characters(df)
    chars_pr_example = total_char_count / len(df)

    budgeted_chars = (max_budget_in_eur // price_pr_million_chars) * 1000000

    return budgeted_chars // chars_pr_example


def merge_new_examples_to_master_dataset(df_translated: pl.DataFrame):
    ds = datasets.load_dataset("kobprof/skolegpt-instruct")
    df_master = ds["train"].to_pandas()
    df_translated = df_translated.to_pandas()
    df_master = pd.concat([df_master, df_translated])
    ds = datasets.Dataset.from_pandas(df_master, preserve_index=False)
    ds = ds.shuffle()
    return ds
