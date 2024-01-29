import typer

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.translate import translate_dataset
from skolegpt_instruct_dataset.utils import load_parquet_file_with_polars


def main(save_freq: int = 100):
    df = load_parquet_file_with_polars(
        config.data_dir / config.stratified_dataset_file_name
    )

    df = df.sample(len(df), shuffle=True)

    df_translated = translate_dataset(df, save_freq)

    print("Completed translating dataset.")

    df_translated.write_parquet(config.data_dir / config.translated_dataset_file_name)

    breakpoint()


if __name__ == "__main__":
    typer.run(main)
