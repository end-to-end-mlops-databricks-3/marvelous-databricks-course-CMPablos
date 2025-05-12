import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from mlops_course.config import ProjectConfig

class DataProcessor:
    def __init__(self, df: pd.DataFrame, 
                 config: ProjectConfig, 
                 spark: SparkSession) -> None:
        self.df = df
        self.config = config
        self.spark = spark
    def preprocess(self) -> None:
        """Prepocess the DataFrame stored in the instanced class."""

        # Replacing booking_status values to numeric
        self.df['booking_status'].replace('Not_Canceled', 0, inplace=True)
        self.df['booking_status'].replace('Canceled', 1, inplace=True)

        # Converting features to numeric
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors = "coerce")

        # Converting categorical features
        cat_features = self.config.cat_features
        for col in self.config.cat_features:
            self.df[col] = self.df[col].astype("category")

        target_feature = self.config.target
        relevant_cols = cat_features + num_features + target_feature
        self.df = self.df[relevant_cols]

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splitsthe dataframe into a train and a test set

        :param test_size: Proportion of the dataset to include in the test plit
        :param random_state: Sets the seed for the pseudo-random shuffling applied to the data
        :return: A tuple containing the train and test DataFrames
        """
        
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set
    
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Loads the train and test sets into Databricks tables.
        
        :param train_set: The training DataFrame to be loaded.
        :param test_set: The testing DataFrame to be loaded.
        """

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name
        dataset_name = self.config.dataset_name

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{catalog_name}.{schema_name}.{dataset_name}_train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{catalog_name}.{schema_name}.{dataset_name}_test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name
        dataset_name = self.config.dataset_name

        self.spark.sql(
            f"ALTER TABLE {catalog_name}.{schema_name}.{dataset_name}_train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {catalog_name}.{schema_name}.{dataset_name}_test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )