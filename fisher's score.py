from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col, when, avg, variance
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
numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType)) and field.name != 'Cancelled']

# Compute Fisher Scores for numeric columns
fisher_scores = []
for column in numeric_cols:
    group1 = df.filter(col("Cancelled") == 1).select(column)
    group2 = df.filter(col("Cancelled") == 0).select(column)
    
    mean1 = group1.agg(avg(column)).first()[0]
    mean2 = group2.agg(avg(column)).first()[0]
    var1 = group1.agg(variance(column)).first()[0]
    var2 = group2.agg(variance(column)).first()[0]
    
    # Compute Fisher Score
    if var1 + var2 > 0:  # Avoid division by zero
        fisher_score = (mean1 - mean2) ** 2 / (var1 + var2)
        fisher_scores.append((column, fisher_score))

# Create DataFrame for Fisher Scores
fisher_score_df = spark.createDataFrame(fisher_scores, ["Feature", "Fisher_Score"])
fisher_score_df = fisher_score_df.orderBy("Fisher_Score", ascending=False)

# Show the results
fisher_score_df.show()

# Save results
fisher_score_df.write.option("header", True).csv('s3://jerryfan-demo-bucket/fisher_scores.csv')

# Stop the Spark session
spark.stop()
