"""Dataset processing script."""

import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.data_processor import DataProcessor

config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file=f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# Load the hotel_reservations dataset from volume
spark = SparkSession.builder.getOrCreate()

hotel_reservation_data = (
    f"/Volumes/{config.catalog_name}/{config.schema_name}//hotel_reservations/hotel_reservations.csv"
)
sdf = spark.read.csv(hotel_reservation_data, header=True, inferSchema=True)
df = sdf.toPandas()

# Preprocess the data
with Timer() as preprocess_timer:
    data_processor = DataProcessor(df, config, spark)
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
