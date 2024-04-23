from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, abs as abs_spark, when
from pyspark.sql.types import IntegerType, DoubleType, StringType, BooleanType

spark = SparkSession.builder.getOrCreate()

#load data
file_path = 's3://natz-demo-bucket/cleaned_2020.csv/'
df = spark.read.csv(file_path, header=True, inferSchema=True)

#variable transformation like chi-square test
if isinstance(df.schema["Cancelled"].dataType, BooleanType):
    df = df.withColumn("Cancelled", when(col("Cancelled") == True, 1).otherwise(0))
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType)) and field.name not in categorical_columns + ['Cancelled']]
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
assembler_inputs = [c + "_index" for c in categorical_columns] + numeric_cols + ['Cancelled']
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
pipeline = Pipeline(stages=indexers + [assembler])
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

#compute correlation matrix
correlation_matrix = Correlation.corr(df_transformed, "features").head()[0].toArray()

#add naming in the correlation matrix
feature_names = assembler_inputs
correlation_df = spark.createDataFrame(correlation_matrix.tolist(), schema=feature_names)

#extract and sort the correlations with 'Cancelled'
cancelled_correlations = correlation_df.select("Cancelled").rdd.flatMap(lambda x: x).collect()
cancelled_corr_df = spark.createDataFrame(zip(feature_names, cancelled_correlations), ["Feature", "Correlation"])
top_correlations = cancelled_corr_df.filter(cancelled_corr_df.Feature != "Cancelled") \
                                    .withColumn("AbsCorrelation", abs_spark("Correlation")) \
                                    .orderBy("AbsCorrelation", ascending=False).limit(30)

#save output
correlation_df.write.option("header", True).csv('s3://natz-demo-bucket/correlation_matrix_spark.csv')
top_correlations.write.option("header", True).csv('s3://natz-demo-bucket/top_correlations.csv')

spark.stop()
