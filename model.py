from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                       RandomForestClassifier, GBTClassifier,
                                       NaiveBayes, LinearSVC)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

# Start Spark session
spark = SparkSession.builder.appName("Classification Models Comparison") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Load the training and test datasets
train_data = spark.read.csv('s3://natz-demo-bucket/train_data_cm.csv/', header=True, inferSchema=True)
test_data = spark.read.csv('s3://natz-demo-bucket/test_data_cm.csv/', header=True, inferSchema=True)

# Function to replace negative, NaN, and infinity values
def replace_negative(df):
    for col_name in df.columns:
        df = df.withColumn(col_name, when(col(col_name) < 0, 0).otherwise(col(col_name)))  # Replace negative values with 0
        df = df.na.fill({col_name: 0})  # Replace NaNs with 0
    return df

train_data = replace_negative(train_data)
test_data = replace_negative(test_data)

# Assemble features
feature_columns = train_data.columns[:-1]  # Assuming 'Cancelled' is the last column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Balancing classes by adjusting weights
balancing_ratio = train_data.filter(col('Cancelled') == 1).count() / train_data.count()
calculate_weights = udf(lambda x: 1 * balancing_ratio if x == 0 else (1 - balancing_ratio), DoubleType())
train_data = train_data.withColumn("classWeightCol", calculate_weights("Cancelled"))

# Evaluators
evaluator_auc = BinaryClassificationEvaluator(labelCol='Cancelled', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='Cancelled', predictionCol='prediction', metricName='accuracy')

# List of classifiers
classifiers = [
    LogisticRegression(featuresCol='features', labelCol='Cancelled', weightCol='classWeightCol'),
    DecisionTreeClassifier(featuresCol='features', labelCol='Cancelled'),
    RandomForestClassifier(featuresCol='features', labelCol='Cancelled'),
    GBTClassifier(featuresCol='features', labelCol='Cancelled'),
    NaiveBayes(featuresCol='features', labelCol='Cancelled'),
    LinearSVC(featuresCol='features', labelCol='Cancelled')
]

# Function to train and evaluate a model with cross-validation
def train_evaluate_model(classifier):
    paramGrid = ParamGridBuilder().build()  # Add specific hyperparameters if needed
    pipeline = Pipeline(stages=[assembler, classifier])
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator_auc,
                              numFolds=5)
    cvModel = crossval.fit(train_data)
    predictions = cvModel.transform(test_data)

    # Compute additional metrics using MulticlassMetrics
    rdd = predictions.select(['prediction', 'Cancelled']).rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = MulticlassMetrics(rdd)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    accuracy = evaluator_accuracy.evaluate(predictions)  # Calculate accuracy

    # Confusion matrix
    confusion_matrix = metrics.confusionMatrix().toArray()
    confusion_df = spark.createDataFrame([Row(TP=float(confusion_matrix[1,1]), FP=float(confusion_matrix[0,1]), 
                                              FN=float(confusion_matrix[1,0]), TN=float(confusion_matrix[0,0]))])

    # Save confusion matrix
    confusion_matrix_path = f's3://natz-demo-bucket/cm_confusion_matrix/cm_{classifier.__class__.__name__}.csv'
    confusion_df.write.csv(confusion_matrix_path, mode='overwrite', header=True)

    return (classifier.__class__.__name__, evaluator_auc.evaluate(predictions), accuracy, precision, recall, f1Score)

# Evaluate all classifiers
results = [train_evaluate_model(classifier) for classifier in classifiers]
results_df = spark.createDataFrame(results, ["Classifier", "AUC", "Accuracy", "Precision", "Recall", "F1 Score"])

# Show results
results_df.show()

# Save results to local storage
results_df.write.csv('s3://natz-demo-bucket/evaluation_cm_results.csv', mode='overwrite', header=True)

# Stop Spark session
spark.stop()