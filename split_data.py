from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, NumericType, BooleanType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
# Start Spark session
spark = SparkSession.builder.getOrCreate()

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

# Adjust these names based on actual column names
selected_columns_chi = [
    "Airline_index", "Origin_index", "Dest_index", 
    "Marketing_Airline_Network_index", "Operated_or_Branded_Code_Share_Partners_index",
    "IATA_Code_Marketing_Airline_index", "Operating_Airline_index", 
    "IATA_Code_Operating_Airline_index", "Tail_Number_index", 
    "OriginCityName_index", "OriginState_index", "OriginStateName_index", 
    "DestCityName_index", "DestState_index", "DestStateName_index", 
    "DepTimeBlk_index", "ArrTimeBlk_index", "CRSDepTime", "DepTime", 
    "DepDelayMinutes", "DepDelay", "ArrTime", "ArrDelayMinutes", 
    "AirTime", "CRSElapsedTime", "ActualElapsedTime", "Distance", 
    "Quarter", "Month", "DayofMonth",
    "Cancelled"  # Include the 'Cancelled' column in the selection
]

df_selected_chi = df_transformed.select(selected_columns_chi)

# Rebalancing the dataset if necessary
major_df_chi = df_selected_chi.filter(col("Cancelled") == 0)
minor_df_chi = df_selected_chi.filter(col("Cancelled") == 1)
minor_count_chi = minor_df_chi.count()

if minor_count_chi > 0:
    ratio_chi = major_df_chi.count() / minor_count_chi
    balanced_df_chi = minor_df_chi.sample(withReplacement=True, fraction=ratio_chi)
    combined_df_chi = major_df_chi.unionAll(balanced_df_chi)
else:
    print("No cancelled flights to rebalance.")
    combined_df_chi = df_selected_chi

# Splitting the data into training and testing sets
train_df_chi, test_df_chi = combined_df_chi.randomSplit([0.7, 0.3])

# Save the processed dataframes as CSV
train_df_chi.write.csv('s3://natz-demo-bucket/train_data_chi.csv', header=True)
test_df_chi.write.csv('s3://natz-demo-bucket/test_data_chi.csv', header=True)

#correlation matrix
# Adjust these names based on actual column names
selected_columns_cm = [
    "ArrTime", "WheelsOn", "WheelsOff", "DepTime", "ActualElapsedTime",
    "TaxiOut", "AirTime", "TaxiIn", "Tail_Number_index", "Month",
    "Quarter", "ArrivalDelayGroups", "ArrDel15", "DepDel15", "DepartureDelayGroups",
    "DayofMonth", "ArrDelayMinutes", "DepDelayMinutes", "DivAirportLandings",
    "DOT_ID_Operating_Airline", "ArrDelay", "DOT_ID_Marketing_Airline",
    "Operated_or_Branded_Code_Share_Partners_index", "ArrTimeBlk_index", 
    "DepTimeBlk_index", "DepDelay", "CRSArrTime", "Distance", "DistanceGroup", "Cancelled"  # Include the 'Cancelled' column in the selection
]

df_selected_cm = df_transformed.select(selected_columns_cm)

# Rebalancing the dataset if necessary
major_df_cm = df_selected_cm.filter(col("Cancelled") == 0)
minor_df_cm = df_selected_cm.filter(col("Cancelled") == 1)
minor_count_cm = minor_df_cm.count()

if minor_count_cm > 0:
    ratio_cm = major_df_cm.count() / minor_count_cm
    balanced_df_cm = minor_df_cm.sample(withReplacement=True, fraction=ratio_cm)
    combined_df_cm = major_df_cm.unionAll(balanced_df_cm)
else:
    print("No cancelled flights to rebalance.")
    combined_df_cm = df_selected_cm

# Splitting the data into training and testing sets
train_df_cm, test_df_cm = combined_df_cm.randomSplit([0.7, 0.3])

# Save the processed dataframes as CSV
train_df_cm.write.csv('s3://natz-demo-bucket/train_data_cm.csv', header=True)
test_df_cm.write.csv('s3://natz-demo-bucket/test_data_cm.csv', header=True)
# Stop Spark session
spark.stop()
