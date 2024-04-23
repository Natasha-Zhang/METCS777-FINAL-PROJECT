from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, ChiSqSelector
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType, NumericType, BooleanType

# Start Spark session
spark = SparkSession.builder.appName("FeatureSelectionMethods").getOrCreate()

# Load the cleaned data
file_path = 's3://natz-demo-bucket/cleaned_2020.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check and convert 'Cancelled' if it's not numeric
if isinstance(df.schema["Cancelled"].dataType, BooleanType):
    df = df.withColumn("Cancelled", when(col("Cancelled") == True, 1).otherwise(0))

# Separate the columns by type for feature transformation
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType) and field.name != 'Cancelled']

# Index the categorical columns so they can be used by the models
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").setHandleInvalid("skip") for column in categorical_columns]

# Assemble the features into a vector
assembler = VectorAssembler(inputCols=[c + "_index" for c in categorical_columns] + numerical_columns, outputCol="features")

# Build the pipeline
pipeline = Pipeline(stages=indexers + [assembler])

# Fit the pipeline to the data
df_transformed = pipeline.fit(df).transform(df)

# Feature selection using Chi-Squared test
chi_sq_selector = ChiSqSelector(featuresCol="features", outputCol="selectedFeatures", labelCol="Cancelled", numTopFeatures=30)
chi_sq_model = chi_sq_selector.fit(df_transformed)
df_selected = chi_sq_model.transform(df_transformed)

# Get the list of selected feature names after Chi-Squared test
selected_features = chi_sq_model.selectedFeatures
selected_feature_names = [assembler.getInputCols()[index] for index in selected_features]

# Convert list of selected features to RDD and save directly to S3
output_path = 's3://natz-demo-bucket/selected_chi.txt'
spark.sparkContext.parallelize(selected_feature_names).saveAsTextFile(output_path)

# Stop the Spark session
spark.stop()
