import streamlit as st
import duckdb
import pandas as pd
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
from pydantic import BaseModel

class Text2SQLSample(BaseModel):
    query: str


def text2sql(user_query: str, schema: str) -> str:
    model: str = "gpt-4o"
    client = OpenAI()

    prompt_template = """
    Write the corresponding SQL query based on user prompt and database schema:

    - user prompt: {user_prompt}
    - database schema: {schema}
    Return only JSON.
    """

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
    return query

# Initialize DuckDB connection
conn = duckdb.connect()

# Load the httpfs extension for hf:// paths
conn.execute("INSTALL httpfs;")
conn.execute("LOAD httpfs;")

# Title
st.title("Hugging Face Dataset Query with DuckDB")

# Input for Hugging Face dataset link
hf_link = st.text_input(
    "Enter the Hugging Face dataset link (hf://...):",
    "hf://datasets/datasets-examples/doc-formats-csv-1/data.csv",
)

# Input for SQL query
default_query = f"SELECT * FROM '{hf_link}' LIMIT 10;"
query = st.text_area("Enter your SQL query:", default_query, height=100)


# Execute query when button is clicked
if st.button("Run Query"):

    try:
        # Use DESCRIBE to get schema
        schema_query = f"DESCRIBE SELECT * FROM '{hf_link}'"
        schema_df = conn.execute(schema_query).df()
        st.write("### Dataset Schema:")
        st.dataframe(schema_df)
    except Exception as e:
        st.error(f"An error occurred while fetching schema: {e}")

    try:
        # Execute the query
        df = conn.execute(query).df()
        # Display the results
        st.write("### Query Results:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"An error occurred: {e}")