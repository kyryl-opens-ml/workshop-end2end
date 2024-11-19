from dagster import (
    AssetExecutionContext,
    AssetOut,
    Config,
    MetadataValue,
    asset,
    multi_asset,
)
from datasets import concatenate_datasets


@multi_asset(
    outs={
        "df_train": AssetOut(),
        "df_val": AssetOut(),
    },
    group_name="data",
)
def origin_text2sql_dataset(context: AssetExecutionContext):
    return "df_train", "df_val"


@asset(group_name="data")
def labeling_data(
    context: AssetExecutionContext, df_train, df_val
) -> str:
    pass


@asset(group_name="data")
def test_dataset(context: AssetExecutionContext, labeling_data: str):
    pass


@asset(group_name="data")
def train_dataset(context: AssetExecutionContext, labeling_data: str):
    pass


@asset(group_name="data")
def feedback_dataset():
    pass


@asset(group_name="model")
def training_args():
    pass

@asset(group_name="model")
def script_args():
    pass


@asset(group_name="model")
def pre_trained_llm(script_args):
    pass


@asset(group_name="model")
def tokenizer(script_args):
    pass


@asset(group_name="model")
def trained_model(
    context: AssetExecutionContext,
    train_dataset,
    test_dataset,
    tokenizer,
    pre_trained_llm,
    training_args,
    script_args,
):
    pass


@asset(group_name="model")
def trained_model_with_feedback(
    context: AssetExecutionContext,
    train_dataset,
    feedback_dataset,
    test_dataset,
    tokenizer,
    pre_trained_llm,
    training_args,
    script_args,
):
    pass
