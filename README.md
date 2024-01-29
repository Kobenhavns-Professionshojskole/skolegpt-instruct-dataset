<h1 align="center">
<img src="logo.png" width="250">
</h1>

## SkoleGPT Instruct Dataset
"skolegpt-instruct" is an open source dataset for Danish instruction fine-tuning of LLM's. The dataset is translation of a quality filtered subset of the [OpenOrca instruction dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca). The project is a part of the [SkoleGPT project](https://skolegpt.dk/).

## Usage
The project consist of 4 steps: sampling, filtering, stratification and translation. This project utilizes Poetry for dependency management, so you may need to install Poetry using pip install poetry if it's not already set up on your system.

1. Sample OpenOrca dataset:
```bash
poetry run python sample_dataset.py 
```

2. Filter sampled dataset:
```bash
poetry run python filter_dataset.py
```

3. Stratification of sampled dataset by the OpenOrca sources: niv, flan, cot, t0:
```bash
poetry run python stratify_dataset.py
```

4. Translate filterd dataset:
```bash
poetry run python translate_dataset.py
```

## Dataset
### Data Sampling
The data extraction process involves loading and shuffling the [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca), specifically the "1M-GPT4-Augmented.parquet" file. A specified number of entries are then selected to form a subset, which is organized into a DataFrame with an added "source" column for origin tracking. This results in a manageable and tailored subset of the dataset for analysis or further processing.

### Filtering
The filter_data function is designed to preprocess and filter the raw OpenOrca dataset. This process involves several steps, each targeting specific types of data or formatting issues within the dataset. 

Below is an outline of these steps:

1. **Remove Already Translated Instructions:** If translated examples already exists in the [kobprof/skolegpt-instruct](https://huggingface.co/datasets/kobprof/skolegpt-instruct) dataset on the Hugging Face hub, remove them from the dataset.

2. **Remove Translation Instructions:** Filters out entries containing the word "translate" in the "question" field, targeting instances that are likely to be translation instructions.

3. **Remove Common Prefixes and Postfixes:** Strips common prefixes and postfixes from the "question" field. This is achieved through regular expressions constructed from provided lists of common prefixes and postfixes.

4. **Remove Questions Ending with a Colon:** Filters out entries where the "question" field ends with a colon, as these often indicate incomplete or improperly formatted questions.

5. **Remove Multiple Choice Questions:** Identifies and removes multiple-choice questions. This is done using regular expressions to detect common multiple-choice question formats, such as options labeled with letters or numbers.

6. **Basic Cleaning:** Performs basic cleaning of the dataset by stripping characters from the "system_prompt", "question", and "response" fields and removing entries where "question" or "response" fields are empty.

7. **Remove Exotic Characters:** Filters out entries containing exotic characters in the "question" and "response" fields. The list of characters to filter is dynamically generated based on the dataset content.

8. **Remove Duplicate Questions and Responses:** Eliminates duplicates in the dataset, ensuring uniqueness in both "question" and "response" fields.

### Translation
The dataset translation is carried out via the DeepL service. This process necessitates having a DeepL account with a linked credit card. DeepL provides a free tier, allowing access to their API for translating up to 500,000 characters, which can be found [here](https://support.deepl.com/hc/en-us/articles/360021200939-DeepL-API-Free). There are approximately 16 unique system prompts consistently used throughout all instructions. By translating only these unique system prompts instead of translating them for each row, we can significantly conserve character usage.
# skolegpt-instruct-dataset
