from typing import Dict, List

import argilla as rg
import typer
from datasets import Dataset, load_dataset
from openai import OpenAI
from pydantic import BaseModel
from retry import retry
from tqdm import tqdm


class Text2SQLSample(BaseModel):
    prompt: str
    schema: str
    query: str

client = rg.Argilla(api_url="http://0.0.0.0:6900", api_key="argilla.apikey")


def create_text2sql_dataset(dataset_name: str, data_samples: Dataset | List[Dict]):
    guidelines = """
    Please examine the given DuckDB SQL question and context. 
    Write the correct DuckDB SQL query that accurately answers the question based on the context provided. 
    Ensure the query follows DuckDB SQL syntax and logic correctly.
    """

    settings = rg.Settings(
        guidelines=guidelines,
        fields=[
            rg.TextField(
                name="prompt",
                title="Prompt",
                use_markdown=False,
            ),
            rg.TextField(
                name="schema",
                title="Schema",
                use_markdown=True,
            ),
            rg.TextField(
                name="query",
                title="Query",
                use_markdown=True,
            ),            
        ],
        questions=[
            rg.TextQuestion(
                name="sql",
                title="Please write SQL for this query",
                description="Please write SQL for this query",
                required=True,
                use_markdown=True,
            )
        ],
    )

    if 'admin' not in [x.name for x in client.workspaces.list()]:
        workspace = rg.Workspace(name="admin")
        workspace.create()

    dataset = rg.Dataset(
        name=dataset_name,
        workspace="admin",
        settings=settings,
        client=client,
    )
    dataset.create()
    records = []
    for idx in range(len(data_samples)):
        x = rg.Record(
            fields={
                "prompt": data_samples[idx]["prompt"],
                "schema": data_samples[idx]["schema"],
                "query": data_samples[idx]["query"],
            },
        )
        records.append(x)
    dataset.records.log(records, batch_size=1000)

def upload_duckdb_text2sql():
    dataset_name = "motherduckdb/duckdb-text2sql-25k"
    raw_dataset = load_dataset(dataset_name, split="train")
    raw_datasets = raw_dataset.train_test_split(test_size=0.05, seed=42)
    
    # raw_datasets['train'].to_json(path_or_buf='./data/train.json')
    # raw_datasets['test'].to_json(path_or_buf='./data/test.json')

    create_text2sql_dataset(dataset_name='duckdb-text2sql-train', data_samples=raw_datasets['train'].to_list())
    create_text2sql_dataset(dataset_name='duckdb-text2sql-test', data_samples=raw_datasets['test'].to_list())


@retry(tries=3, delay=1)
def generate_synthetic_example() -> Dict[str, str]:
    client = OpenAI()

    prompt = """
    Generate a example for text2sql task for DuckDB database: 
    The example should include 
    - schema: a valid database schema
    - prompt: a typical user question related to this table prompt
    - query: the corresponding SQL query to answer user prompt.
    Return only JSON.
    """

    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": "You are DuckDB and SQL expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="gpt-4o",
        response_format=Text2SQLSample,
        temperature=1,
    )
    sample = chat_completion.choices[0].message.parsed
    return sample.model_dump()


def create_text2sql_dataset_synthetic(num_samples: int = 10):
    

    samples = []
    for _ in tqdm(range(num_samples)):
        sample = generate_synthetic_example()
        samples.append(sample)

    dataset_name = "duckdb-text2sql-synthetic"
    create_text2sql_dataset(dataset_name=dataset_name, data_samples=samples)

if __name__ == "__main__":
    app = typer.Typer()
    app.command()(upload_duckdb_text2sql)
    app.command()(create_text2sql_dataset_synthetic)
    app()
