from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, DoubleType

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load the data
file_path = 's3://natz-demo-bucket/Combined_Flights_2020.csv' 
df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Check for any columns that look like they were incorrectly added
extra_columns = [c for c in df.columns if c.startswith("_c")]
if extra_columns:
    # Drop the extra columns if they exist
    df = df.drop(*extra_columns)

# Replace NA in numerical columns with 0 and in character columns with "unknown"
for col_name in df.columns:
    if isinstance(df.schema[col_name].dataType, StringType):
        df = df.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(col(col_name)))
    elif isinstance(df.schema[col_name].dataType, (IntegerType, DoubleType)):
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))

# Output path for the cleaned data
output_path = 's3://natz-demo-bucket/cleaned_2020.csv'
df.write.csv(output_path, header=True)

# Stop the Spark session
spark.stop()
