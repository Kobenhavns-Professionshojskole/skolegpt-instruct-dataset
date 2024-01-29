import polars as pl
from datasets import load_dataset


def get_data(n_max: int, seed: int = 42) -> pl.DataFrame:
    """Data extraction pipeline."""

    ds = load_dataset(
        "Open-Orca/OpenOrca",
        streaming=True,
        split="train",
        data_files="1M-GPT4-Augmented.parquet",
    )
    ds = ds.shuffle(seed=seed)

    examples = []
    for example in ds:
        examples.append(example)
        if len(examples) > n_max:
            break

    df = pl.DataFrame(examples)

    # add source column
    df = df.with_columns(pl.col("id").apply(lambda x: x.split(".")[0]).alias("source"))

    return df
