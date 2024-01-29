import typer

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.data import get_data
from skolegpt_instruct_dataset.utils import create_directory_if_not_exists


def main(
    n_max: int = config.n_max,
    seed: int = config.seed,
):
    create_directory_if_not_exists(config.data_dir)

    df = get_data(
        n_max=n_max,
        seed=seed,
    )

    df.write_parquet(config.data_dir / config.sampled_dataset_file_name)


if __name__ == "__main__":
    typer.run(main)
