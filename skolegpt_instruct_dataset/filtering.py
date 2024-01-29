import re

import datasets
import polars as pl

from .utils import return_filter_char_list


def filter_data(
    df: pl.DataFrame,
    common_prefixes: list[str],
    common_postfixes: list[str],
) -> pl.DataFrame:
    """Data filtering pipeline."""

    original_dataset_size = len(df)
    print(
        "Starting filter_data function. Original dataset size:", original_dataset_size
    )
    df = remove_already_translated_instructions(df)
    df = remove_translation_instructions(df)
    df = remove_common_pre_postfixes(df, common_prefixes, common_postfixes)
    df = remove_questions_ending_with_colon(df)
    df = remove_multiple_choice_questions(df)
    df = basic_cleaning(df)
    df = remove_exotic_chars(df)
    df = remove_duplicate_questions_and_responses(df)

    percent_removed = round(100 * (1 - len(df) / original_dataset_size), 4)
    print(f"{percent_removed} % of dataset removed after preprocessing.")

    return df


# ---------------------------------------------------------------------------- #
#                              Preprocessing Steps                             #
# ---------------------------------------------------------------------------- #


def remove_already_translated_instructions(df: pl.DataFrame) -> pl.DataFrame:
    def get_already_translated_ids():
        try:
            ds = datasets.load_dataset("kobprof/skolegpt-instruct")
            ids = ds["train"]["id"]
        except:
            ids = []
        return ids

    already_translated_ids = get_already_translated_ids()

    df = df.filter(~df["id"].is_in(already_translated_ids))

    return df


def remove_translation_instructions(df):
    # Hard filter on "translate"
    df = df.filter(~df["question"].str.to_lowercase().str.contains("translate"))
    return df


def remove_common_pre_postfixes(df, common_prefixes, common_postfixes):
    prefix_pattern = r"^(?:" + "|".join(re.escape(p) for p in common_prefixes) + ")"
    postfix_pattern = r"(?:" + "|".join(re.escape(p) for p in common_postfixes) + ")$"
    df = df.with_columns(
        df["question"].str.replace_all(prefix_pattern, "").alias("question")
    )
    return df.with_columns(
        df["question"].str.replace_all(postfix_pattern, "").alias("question")
    )


def remove_questions_ending_with_colon(df):
    return df.filter(~df["question"].map_elements(lambda x: x.strip().endswith(":")))


def remove_multiple_choice_questions(df):
    option_patterns = [
        r"(?i)\b[A-D]\)",  # Matches A), B), C), D) in a case-insensitive manner
        r"(?i)\b[1-4]\)",  # Matches 1), 2), 3), 4) in a case-insensitive manner
        r"(?i)\b\([A-D]\)",  # Matches (A), (B), (C), (D) in a case-insensitive manner
        r"(?i)\b[A-D]\.",  # Matches A., B., C., D. in a case-insensitive manner
        r"(?i)\b[A-D]:",  # Matches A:, B:, C:, D: in a case-insensitive manner
        r"\(i+\)",  # Matches (i), (ii), (iii), etc.
        r"\[[A-Z]\]",  # Matches [A], [B], [C], etc.
        r"\b[i]+\.",  # Matches ii., iii., iv., etc.
    ]
    combined_option_pattern = "|".join(option_patterns)
    condition = (
        df["question"].str.contains("Options:")
        | df["question"].str.contains("OPT:")
        | df["question"].str.contains("OPTIONS:")
    ) | df["question"].str.contains(combined_option_pattern)
    return df.filter(~condition)


def basic_cleaning(df):
    df = df.with_columns(
        [
            df["system_prompt"].str.strip_chars(),
            df["question"].str.strip_chars(),
            df["response"].str.strip_chars(),
        ]
    )
    df = df.filter(df["question"] != "")
    df = df.filter(df["response"] != "")
    return df


def remove_exotic_chars(df: pl.DataFrame) -> pl.DataFrame:
    # Filter questions and responses with exotic chars
    def contains_characters(text, characters):
        for char in characters:
            if char in text:
                return True
        return False

    filter_chars = return_filter_char_list(df=df)

    df = df.filter(
        ~df["question"].map_elements(
            lambda x: contains_characters(text=x, characters=filter_chars)
        )
    )
    df = df.filter(
        ~df["response"].map_elements(
            lambda x: contains_characters(text=x, characters=filter_chars)
        )
    )
    return df


def remove_duplicate_questions_and_responses(df: pl.DataFrame) -> pl.DataFrame:
    # remove duplicated reponse and questions
    df = df.unique(subset=["response"], keep="first")
    df = df.unique(subset=["question"], keep="first")
    return df


# ---------------------------------------------------------------------------- #
#                                Stratification                                #
# ---------------------------------------------------------------------------- #
def stratify_dataframe(
    df: pl.DataFrame,
    n_total: int,
    instruction_sources: list[str],
    seed: int,
) -> pl.DataFrame:
    # Calculate the ideal number of samples from each source for a balanced dataset
    samples_per_source = n_total // len(instruction_sources)

    # Calculate counts of each source in the dataframe
    source_counts = df["source"].value_counts()

    # Identify sources with fewer samples than the ideal number
    underrepresented_sources = source_counts.filter(
        source_counts["count"] < samples_per_source
    ).rows()

    # Identify sources with enough or more samples than the ideal number
    sufficiently_represented_sources = source_counts.filter(
        source_counts["count"] >= samples_per_source
    ).rows()

    additional_samples_needed = 0
    for source_count in underrepresented_sources:
        additional_samples_needed += samples_per_source - source_count[1]

    samples_per_source += additional_samples_needed // len(
        sufficiently_represented_sources
    )

    combined_samples = []

    # Include all samples from underrepresented sources
    for source in underrepresented_sources:
        combined_samples.append(df.filter(df["source"] == source[0]))

    # Randomly sample from sufficiently represented sources
    for source in sufficiently_represented_sources:
        combined_samples.append(
            df.filter(df["source"] == source[0]).sample(samples_per_source, seed=seed)
        )

    stratified_df = pl.concat(combined_samples)

    return stratified_df
