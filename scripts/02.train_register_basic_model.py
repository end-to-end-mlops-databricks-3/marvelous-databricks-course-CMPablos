"""Train - register basic model."""

import argparse
import os

import mlflow
from dotenv import load_dotenv
from loguru import logger
from marvelous.common import is_databricks
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.basic_model import BasicModel

# Configure tracking uri
mlflow.get_tracking_uri()

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Load data and prepare features
basic_model.load_data()
basic_model.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()
logger.info("Model training completed.")

basic_model.register_model()
logger.info("Registered model")
