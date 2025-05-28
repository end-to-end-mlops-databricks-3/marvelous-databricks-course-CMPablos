# Databricks notebook source
# MAGIC %pip install hotel_reservations_price-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Configure tracking uri
import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os
from marvelous.common import is_databricks
from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week3"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
# config = ProjectConfig.from_yaml(config_path="project_config.yml")



# COMMAND ----------

# Initialize model
for i in config: print(i)
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define hotel_reservations age feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.hotel_reservations_test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("lead_time", "no_of_special_requests", "arrival_year", config.target_feature)


# COMMAND ----------


from pyspark.sql.functions import col

X_test = X_test.withColumn("no_of_adults", col("no_of_adults").cast("int")) \
       .withColumn("no_of_children", col("no_of_children").cast("int")) \
       .withColumn("no_of_weekend_nights", col("no_of_weekend_nights").cast("int")) \
       .withColumn("no_of_week_nights", col("no_of_week_nights").cast("int")) \
       .withColumn("arrival_month", col("arrival_month").cast("int")) \
       .withColumn("arrival_date", col("arrival_date").cast("int")) \
       .withColumn("no_of_previous_cancellations", col("no_of_previous_cancellations").cast("int")) \
       .withColumn("no_of_previous_bookings_not_canceled", col("no_of_previous_bookings_not_canceled").cast("int"))


# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------

