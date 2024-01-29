import pathlib

data_dir = pathlib.Path(__file__).parents[1] / "data"


class Config:
    seed: int = 42
    n_max: int = 3500000  # number of raw samples from OpenOrca dataset
    n_total: int = 90000  # number of examples in final Danish translated dataset
    data_dir: pathlib.Path = data_dir
    sampled_dataset_file_name: str = "sampled_dataset.parquet"
    filtered_dataset_file_name: str = "filtered_dataset.parquet"
    stratified_dataset_file_name: str = "stratified_dataset.parquet"
    translated_dataset_file_name: str = "translated_dataset.parquet"
    instruction_sources: list[str] = [
        "flan",
        "niv",
        "t0",
        "cot",
    ]  # instuction example sources
    common_prefixes: list[str] = [
        "Question:",
        "Definition:",
        "Detailed Instructions:",
        "Instructions:",
        "Q:",
        "Teacher:",
        "Student:",
        "Write a sentence not in English.",
        "Denny asked:",
        "Choose your answer:",
        "Answer the following question:",
        "Given the question:",
        "Please answer the following question:",
    ]  # common instruction prefixes
    common_postfixes: list[str] = [
        "Answer:",
        "Solution:",
        "A:",
        "Output:",
        "Teacher:",
        "Student:",
        "Stream of thoughts:",
        "Step-by-step reasoning:",
        "Chain-of-thought:",
        "Let's think first:",
        "The thinking starts now:",
        "Stream of consciousness:",
        "Which language is this?",
        "Please think gradually:",
        "The answer is:",
        "Me:",
        "Some thinking first:",
        "Some random thoughts:",
        "Let's solve step-by-step:",
        "Numbered answers:",
        "Let's answer step by step:",
        "The answer to this question is:",
        "Explanation:",
        "Teacher: Let's think:",
        "Let's think:",
        "Chain of thought:",
        "Your thoughts:",
        "Summary:",
    ]  # common instruction postfixes


config = Config()
