from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlops_course.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_total_guests"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the hotel_reservations_features table and populate it.

        This table stores features related to hotel_reservations.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, lead_time INT, no_of_special_requests INT, arrival_year INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hr_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, lead_time, no_of_special_requests, arrival_year FROM {self.catalog_name}.{self.schema_name}.hotel_reservations_train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, lead_time, no_of_special_requests, arrival_year FROM {self.catalog_name}.{self.schema_name}.hotel_reservations_test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the total number of guests

        This function sums the number of adults with the number of children.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(no_of_adults INT, no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return no_of_adults + no_of_children
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'YearBuilt' to integer type.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.hotel_reservations_train_set").drop(
            "lead_time", "no_of_special_requests", "arrival_year"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.hotel_reservations_test_set").toPandas()

        self.train_set = self.train_set.withColumn("no_of_adults", self.train_set["no_of_adults"].cast("int"))
        self.train_set = self.train_set.withColumn("no_of_children", self.train_set["no_of_children"].cast("int"))
        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["lead_time", "no_of_special_requests", "arrival_year"],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="total_guests",
                    input_bindings={"no_of_adults": "no_of_adults", "no_of_children": "no_of_children"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        current_year = datetime.now().year
        self.test_set["total_guests"] = self.test_set["no_of_children"] + self.test_set["no_of_adults"]

        self.X_train = self.training_df[self.num_features + self.cat_features + ["total_guests"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["total_guests"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Precision Score: {precision}")
            logger.info(f"📊 Recall Score: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version
##############################################
    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
