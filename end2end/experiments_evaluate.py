from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm
import typer
import json

import evaluate
from tqdm import tqdm
import typer
from typing import List

from pydantic import BaseModel

class Text2SQLSample(BaseModel):
    query: str

def get_test_data() -> Dataset:
    test_json = "./data/test.json"
    dataset = Dataset.from_json(test_json)
    return dataset

def get_metrics(generated_sql: List[str], gt_sql: List[str]) -> Dataset:

    rouge = evaluate.load("rouge")
    metrics = rouge.compute(predictions=generated_sql, references=gt_sql)
    return metrics    

def evaluate_openai(model: str = "gpt-4o"):
    dataset = get_test_data()
    client = OpenAI()

    prompt_template = """
    Write the corresponding SQL query based on user prompt and database schema:

    - user prompt: {user_prompt}
    - database schema: {schema}
    Return only JSON.
    """

    generated_sql = []
    gt_sql = []
    for x in tqdm(dataset):
        prompt = prompt_template.format(user_prompt=x['prompt'], schema=x['schema'])
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
        model=model,
        response_format=Text2SQLSample,
        temperature=1,
        )
        query = chat_completion.choices[0].message.parsed.query

        generated_sql.append(query)
        gt_sql.append(x['query'])

    metrics = get_metrics(generated_sql=generated_sql, gt_sql=gt_sql)
    print(f"metrics = {metrics}")

def evaluate_basic_rag():
    pass 

def evaluate_llama(model="meta-llama/Llama-3.2-1B-Instruct", url: str = "http://localhost:8000/v1"):
    
    dataset = get_test_data()

    client = OpenAI(base_url=url, api_key="any-api-key")

    prompt_template = """
    Write the corresponding SQL query based on user prompt and database schema:

    - user prompt: {user_prompt}
    - database schema: {schema}
    Return only JSON.
    """

    generated_sql = []
    gt_sql = []
    for x in tqdm(dataset):
        prompt = prompt_template.format(user_prompt=x['prompt'], schema=x['schema'])
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
        model=model,
        response_format=Text2SQLSample,
        )
        query = chat_completion.choices[0].message.parsed.query

        generated_sql.append(query)
        gt_sql.append(x['query'])

    metrics = get_metrics(generated_sql=generated_sql, gt_sql=gt_sql)
    print(f"metrics = {metrics}")


def evaluate_finetuning():
    pass 

if __name__ == "__main__":
    app = typer.Typer()
    app.command()(evaluate_openai)
    app.command()(evaluate_basic_rag)
    app.command()(evaluate_llama)
    app.command()(evaluate_finetuning)
    app()

