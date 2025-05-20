"""Databricks notebook source."""


# COMMAND ------------------

import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
print(config)
setup_logging(log_file="logs/notebook_data_processing.log")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the hotel_reservations dataset from volume
spark = SparkSession.builder.getOrCreate()

hotel_reservation_data = r"/Volumes/mlops_dev/cpablosr/hotel_reservations/hotel_reservations.csv"
sdf = spark.read.csv(hotel_reservation_data, header=True)
df = sdf.toPandas()
# COMMAND ----------
with Timer() as preprocess_timer:
    # Initialize the DataProcessor
    data_processor = DataProcessor(df, config, spark)

    # Preprocessing the data
    data_processor.preprocess()

logger.info(f"Data preprocessing took: {preprocess_timer}")

# COMMAND ----------
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")
# COMMAND ----------
print(X_train.columns)
# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ----------
