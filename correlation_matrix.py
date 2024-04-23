from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, abs as abs_spark, when
from pyspark.sql.types import IntegerType, DoubleType, StringType, BooleanType

# Start Spark session
spark = SparkSession.builder.getOrCreate()

# Load the cleaned data
file_path = 's3://natz-demo-bucket/cleaned_2020.csv/'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Ensure 'Cancelled' column is treated correctly
if isinstance(df.schema["Cancelled"].dataType, BooleanType):
    df = df.withColumn("Cancelled", when(col("Cancelled") == True, 1).otherwise(0))

# Define categorical and numeric columns
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType)) and field.name not in categorical_columns + ['Cancelled']]

# Create indexers for categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
assembler_inputs = [c + "_index" for c in categorical_columns] + numeric_cols + ['Cancelled']

# Assemble features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Check unique values for 'DivAirportLandings' and 'Cancelled'
divAirportLandings_count = df.select('DivAirportLandings').distinct().count()
cancelled_count = df.select('Cancelled').distinct().count()

print(f"Distinct count of 'DivAirportLandings': {divAirportLandings_count}")
print(f"Distinct count of 'Cancelled': {cancelled_count}")

# Build and run the pipeline
pipeline = Pipeline(stages=indexers + [assembler])
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(df_transformed, "features").head()[0].toArray()

# Ensure proper naming in the correlation matrix DataFrame
feature_names = assembler_inputs  # This should include all feature names properly
correlation_df = spark.createDataFrame(correlation_matrix.tolist(), schema=feature_names)

# Extract and sort the correlations with 'Cancelled'
cancelled_correlations = correlation_df.select("Cancelled").rdd.flatMap(lambda x: x).collect()
cancelled_corr_df = spark.createDataFrame(zip(feature_names, cancelled_correlations), ["Feature", "Correlation"])
top_correlations = cancelled_corr_df.filter(cancelled_corr_df.Feature != "Cancelled") \
                                    .withColumn("AbsCorrelation", abs_spark("Correlation")) \
                                    .orderBy("AbsCorrelation", ascending=False).limit(30)

# Show top 30 variables most correlated with 'Cancelled'
top_correlations.show()

# Save results
correlation_df.write.option("header", True).csv('s3://natz-demo-bucket/correlation_matrix_spark.csv')
top_correlations.write.option("header", True).csv('s3://natz-demo-bucket/top_correlations.csv')

# Stop the Spark session
spark.stop()
