import pathlib

import typer

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.filtering import stratify_dataframe
from skolegpt_instruct_dataset.utils import load_parquet_file_with_polars


def main(
    n_total: int = config.n_total,
    instruction_sources: list[str] = config.instruction_sources,
    seed: int = config.seed,
):
    df = load_parquet_file_with_polars(
        config.data_dir / config.filtered_dataset_file_name
    )

    df = stratify_dataframe(
        df=df,
        n_total=n_total,
        instruction_sources=instruction_sources,
        seed=seed,
    )

    print("Completed stratifying dataframe.")

    df.write_parquet(config.data_dir / config.stratified_dataset_file_name)


if __name__ == "__main__":
    typer.run(main)
