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
from langfuse.decorators import observe


class Text2SQLSample(BaseModel):
    query: str

@observe
def text2sql(user_prompt: str, schema: str, table_name: str) -> str:
    model: str = "gpt-4o"
    client = OpenAI()

    prompt = f"""
    Write the corresponding SQL query based on user prompt and database schema:

    - user prompt: {user_prompt}
    - database schema: {schema}
    Return only JSON.

    Table name is {table_name}
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
    "hf://datasets/UCSC-VLAA/Recap-DataComp-1B/data/train_data/train-00000-of-02719.parquet",
)

# Input for NLP query
nlp_query = st.text_area("Enter your query in natural language:", "Show me the first 10 rows.")

# Execute query when button is clicked
if st.button("Run Query"):

    try:
        # Use DESCRIBE to get schema
        schema_query = f"DESCRIBE SELECT * FROM '{hf_link}'"
        schema_df = conn.execute(schema_query).df()
        st.write("### Dataset Schema:")
        st.dataframe(schema_df)

        # Convert schema_df to string format suitable for text2sql
        schema_str = "\n".join([f"{row['column_name']} ({row['column_type']})" for index, row in schema_df.iterrows()])

    except Exception as e:
        st.error(f"An error occurred while fetching schema: {e}")
        schema_str = ""  # Ensure schema_str is defined

    try:
        # Convert NLP query to SQL using text2sql
        sql_query = text2sql(nlp_query, schema_str, table_name=hf_link)
        st.write("### Generated SQL Query:")
        st.code(sql_query, language='sql')

        # sql_query = sql_query.replace("table_name", hf_link)
        # Execute the SQL query
        df = conn.execute(sql_query).df()
        # Display the results
        st.write("### Query Results:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"An error occurred: {e}")