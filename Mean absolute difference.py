from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col, abs as abs_spark, when
from pyspark.sql.types import IntegerType, DoubleType, StringType, BooleanType


# Start Spark session
spark = SparkSession.builder.getOrCreate()

# Load the cleaned data
file_path = 's3://jerryfan-demo-bucket/cleaned_2020.csv/'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Ensure 'Cancelled' column is treated correctly
if isinstance(df.schema["Cancelled"].dataType, BooleanType):
    df = df.withColumn("Cancelled", when(col("Cancelled") == True, 1).otherwise(0))

# Define categorical and numeric columns
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType)) and field.name not in categorical_columns + ['Cancelled']]

# Create indexers for categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
assembler_inputs = [c + "_index" for c in categorical_columns] + numeric_cols

# Assemble features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Build and run the pipeline
pipeline = Pipeline(stages=indexers + [assembler])
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Compute Mean Absolute Difference for numeric columns
mad_values = []
for column in numeric_cols:
    mad = df_transformed.select(abs_spark(col(column) - col("Cancelled")).alias("diff")).groupBy().avg("diff").collect()[0][0]
    mad_values.append((column, mad))

# Create DataFrame for MAD values
mad_df = spark.createDataFrame(mad_values, ["Feature", "Mean_Absolute_Difference"])
mad_df = mad_df.orderBy("Mean_Absolute_Difference", ascending=False)

# Show the results
mad_df.show()

# Save results
mad_df.write.option("header", True).csv('s3://jerryfan-demo-bucket/mad_values.csv')

# Stop the Spark session
spark.stop()
