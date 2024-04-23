# METCS777-FINAL-PROJECT

## Environment setup
### Prerequisites
1. Python 3.x
2. AWS Account
### Step 1: Create an EMR Cluster
1. Choose "Create Cluster"
2. Amazon EMR release: emr-6.15.0
3. Application bundle: Spark 3.5.0 and Hadoop 3.3.6
4. Security and Access: Set up EC2 key pairs, and security groups.
5. Amazon EMR service role: EMR_DefaultRole
6. EC2 instance profile for Amazon EMR: EMR_EC2_DefaultRole
7. Review settings and launch the cluster.
### Step 2: Upload file to S3 bucket
1. Choose "Upload"
2. Select files from local
3. Copy S3 URI link of selected file
### Step 3: Launching Spark Jobs
1. Choose "Steps" in created EMR Cluster
2. Choose "Spark Application"
3. Paste S3 URI link on "application location"
4. Choose "Add Step", run the code

## Dataset Description
Our project leverages the Airline Flights Dataset, which contains extensive flight information, including airline cancellations and delays starting in January 2018. Specifically, we focus on 2020 data to predict flight cancellations. 

For this project, the dataset includes attributes such as flight number, airline code, planned and actual departure times, and delay duration, as well as more detailed identifiers such as airline and airport DOT and IATA codes. The target variable is "Cancelled", which is binary coded as "True" for canceled flights (6% of the data) and "False" for non-cancelled flights (94% of the data) . We use the Commercial_Flights_2020.csv file, which contains the highest distribution of TRUE cancellations, for a clearer analysis.

Dataset source:https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?resource=download&select=Combined_Flights_2020.csv
## Data Cleaning
The code checks for and removes any extraneous columns that may have been added by mistake during data import. Since the row for "Cancelled" shows "TRUE", it all contains missing values. It handles missing values by replacing null values in numeric columns with zeros and by replacing null values in string columns with the string "unknown". After cleaning, the dataset is written back to another S3 bucket location in CSV format.
```{cleaning}
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, DoubleType

#create Spark session
spark = SparkSession.builder.getOrCreate()

#load the data
file_path = 's3://natz-demo-bucket/Combined_Flights_2020.csv' 
df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

#check for any cextra columns that incorrectly added
extra_columns = [c for c in df.columns if c.startswith("_c")]
if extra_columns:
    #drop the extra columns if they exist
    df = df.drop(*extra_columns)

#replace NA in numerical columns with 0 and in character columns with "unknown"
for col_name in df.columns:
    if isinstance(df.schema[col_name].dataType, StringType):
        df = df.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(col(col_name)))
    elif isinstance(df.schema[col_name].dataType, (IntegerType, DoubleType)):
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))

#output path for the cleaned data
output_path = 's3://natz-demo-bucket/cleaned_2020.csv'
df.write.csv(output_path, header=True)

#stop the Spark session
spark.stop()
```
## Data Transformation
The data transformation part includes checking and conversion of the "Cancelled" column to ensure it is numeric, which is traget preidict variable in this project. It then classifies the columns into categorical and numeric types, which facilitates subsequent feature selection.

For categorical data, the code uses a StringIndexer to convert the categorical column into a numeric index that the machine learning model can easily process. These indexed categorical columns and raw numeric columns are then combined into vector columns called "features" using VectorAssembler.

```{transformation}
#load data
file_path = 's3://natz-demo-bucket/cleaned_2020.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

#check and convert 'Cancelled'to numeric
if isinstance(df.schema["Cancelled"].dataType, BooleanType):
    df = df.withColumn("Cancelled", when(col("Cancelled") == True, 1).otherwise(0))

#separate the columns by type
categorical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
numerical_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType) and field.name != 'Cancelled']

#index the categorical columns so they can be used by the models
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").setHandleInvalid("skip") for column in categorical_columns]

#assemble the features into a vector
assembler = VectorAssembler(inputCols=[c + "_index" for c in categorical_columns] + numerical_columns, outputCol="features")

#build the pipeline
pipeline = Pipeline(stages=indexers + [assembler])

#fit the pipeline to the data
df_transformed = pipeline.fit(df).transform(df)
```
## Feature Selection
In this project, we use four different feature selection methods to refine and optimize the dataset to improve predictive modeling performance. These methods were chosen to address the high dimensionality of the original dataset by identifying the features most relevant to our model. The selection techniques used are:

### Chi-square Test
This method is used for categorical features to evaluate the independence between each feature and the target variable. It helps determine whether the occurrence of a specific characteristic and the occurrence of a specific outcome are independent.

```{chi-square}
#feature selection using Chi-Squared test
chi_sq_selector = ChiSqSelector(featuresCol="features", outputCol="selectedFeatures", labelCol="Cancelled", numTopFeatures=30)
chi_sq_model = chi_sq_selector.fit(df_transformed)
df_selected = chi_sq_model.transform(df_transformed)

#get the list of selected feature names
selected_features = chi_sq_model.selectedFeatures
selected_feature_names = [assembler.getInputCols()[index] for index in selected_features]

#save output
output_path = 's3://natz-demo-bucket/selected_chi.txt'
spark.sparkContext.parallelize(selected_feature_names).saveAsTextFile(output_path)
```
This code uses the ChiSqSelector in PySpark's MLlib to identify the top 30 features with the most statistically significant association with the target variable "Cancelled". The code first fits a ChiSqSelector to the transformed dataset (df_transformed), which contains the feature vectors. After fitting, it applies the model to transform the dataset, extracting the first 30 features into a new column selectedFeatures. It then retrieves the indexes of these selected features and maps them back to their original names using the VectorAssembler input columns.

Output: "Airline_index", "Origin_index", "Dest_index", "Marketing_Airline_Network_index", "Operated_or_Branded_Code_Share_Partners_index", "IATA_Code_Marketing_Airline_index", "Operating_Airline_index", "IATA_Code_Operating_Airline_index", "Tail_Number_index", "OriginCityName_index", "OriginState_index", "OriginStateName_index", "DestCityName_index", "DestState_index", "DestStateName_index", "DepTimeBlk_index", "ArrTimeBlk_index", "CRSDepTime", "DepTime", "DepDelayMinutes", "DepDelay", "ArrTime", "ArrDelayMinutes", "AirTime", "CRSElapsedTime", "ActualElapsedTime", "Distance", "Quarter", "Month", "DayofMonth", "Cancelled" 

### Correlation Matrix
This method evaluates the linear relationship between each pair of features and between the features and the target variable. Features that are highly correlated with the target are selected, while redundant features with high cross-correlation are usually removed to reduce multicollinearity.

```{cm}
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
```
This code shows the calculation of the correlation matrix for the features in the transformed DataFrame, paying special attention to the correlation with the "cancelled" target variable. This process is performed using PySpark's MLlib Correlation.corr() method, which computes pairwise correlations between all features in the "features" column of the df_transformed DataFrame. The correlation matrix is first converted into a properly named DataFrame (correlation_df) using the feature names from the VectorAssembler input column (assembler_inputs) for easier interpretation. From this dataframe, correlations specifically related to the "cancel" feature are extracted, converted into a flat list, and then paired with their corresponding feature names. This is encapsulated into another DataFrame (cancelled_corr_df) which is further processed to sort the features based on their absolute value related to "Cancelled". The top 30 features with the highest absolute correlation were selected to gain insight into which variables have the strongest linear relationship with flight cancellations.

Output:  "ArrTime", "WheelsOn", "WheelsOff", "DepTime", "ActualElapsedTime", "TaxiOut", "AirTime", "TaxiIn", "Tail_Number", "Month", "Quarter", "ArrivalDelayGroups", "ArrDel15", "DepDel15", "DepartureDelayGroups", "DayofMonth", "ArrDelayMinutes", "DepDelayMinutes", "DivAirportLandings", "DOT_ID_Operating_Airline", "ArrDelay", "DOT_ID_Marketing_Airline", "Operated_or_Branded_Code_Share_Partners", "ArrTimeBlk", "DepTimeBlk", "DepDelay", "CRSArrTime", "Distance", "DistanceGroup", "Cancelled"

### Fisher's Score
Fisher score is a measure of the discriminative power of a variable based on a scatter matrix of different categories. It ranks features based on the ratio of between-class variance to within-class variance and selects the highest-scoring features for class separation.

```{fisher}
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
```
The code shows that the data set is initially divided into two groups based on the "cancelled" status of the flights. For each numerical feature, the mean and variance are calculated separately within these groups. Using these statistical measures, calculate the Fisher score for each feature, carefully approaching the scenario to avoid division by zero by ensuring that the combined variance is greater than zero. Once the Fisher scores are calculated, they are sorted in descending order and encapsulated into a DataFrame for clarity and ease of analysis. Features with higher Fisher scores are highlighted because they have greater discriminatory power in differentiating between canceled and non-cancelled flights.

Output: "ArrTime", "WheelsOn", "WheelsOff", "DepTime", "ActualElapsedTime", "TaxiOut", "AirTime", "TaxiIn", "Month", "Quarter", "ArrivalDelayGroups", "ArrDel15", "DepDel15", "DepartureDelayGroups", "ArrDelayMinutes", "DepDelayMinutes", "DayofMonth", "ArrDelay", "DOT_ID_Operating_Airline", "DOT_ID_Marketing_Airline", "DepDelay", "DivAirportLandings", "Cancelled"
    
### Mean Absolute Difference
This method examines the absolute difference between the means of features of different categories. Features with higher average differences are more likely to have an impact on the prediction results and are therefore preferred.

```{MAD}
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
```
The code process of evaluating feature correlation by calculating the mean absolute difference (MAD) between each numerical feature and the binary "cancelled" state in the transformed dataset. The method involves iterating through each numerical column to calculate the absolute difference from the "cancelled" state, averaging these differences to determine the MAD for each feature. The results are then aggregated into a list and structured into a data frame for enhanced organization and clarity. The data frame is sorted in descending order of MAD values to highlight features with the greatest variability and potential predictive power regarding flight cancellations.

Output:  "DestAirportSeqID", "OriginAirportSeqID", "DestCityMarketID", "OriginCityMarketID", "DOT_ID_Operating_Airline", "DOT_ID_Marketing_Airline", "DestAirportID", "OriginAirportID", "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline", "Year", "CRSArrTime", "ArrTime", "WheelsOn", "CRSDepTime", "WheelsOff", "DepTime", "Distance", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "DestWac", "OriginWac", "DestStateFips", "OriginStateFips", "ArrDelay", "DayofMonth", "TaxiOut", "DepDelay", "ArrDelayMinutes", "Cancelled"

## Split Data
The code shows splitting the dataset into training and test sets after performing feature selection using chi-square test. To address potential imbalances in the dataset, especially given the uneven distribution of the "cancelled" target variable, the script rebalances the dataset by oversampling the minority class (canceled flights) to match the majority class, ensuring both classes are equally representative. After the data set is balanced, it is randomly divided into a training set and a test set in a ratio of 7:3.

**The other 3 feature selection outputs also use this code to split the data.**
```{split}
#selected feature after chi-square test
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
    "Quarter", "Month", "DayofMonth", "Cancelled" 
]

df_selected_chi = df_transformed.select(selected_columns_chi)

#rebalancing the dataset
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

#splitting the data into training and testing sets
train_df_chi, test_df_chi = combined_df_chi.randomSplit([0.7, 0.3])

#save output
train_df_chi.write.csv('s3://natz-demo-bucket/train_data_chi.csv', header=True)
test_df_chi.write.csv('s3://natz-demo-bucket/test_data_chi.csv', header=True)
```

## Bulid Models with Machine Learning Algorithm and Evaluation
The code is an example of using PySpark to build and evaluate multiple machine learning models for binary classification tasks with dataset after Chi-Square Test selection. Data cleaning functions are used to handle negative, NaN, and infinity values to ensure that the data set is suitable for modeling. Assemble features into vectors using VectorAssembler, assuming "Cancelled" is the target variable. The evaluation process uses a cross-validation approach within a pipeline framework, allowing robust testing across different data folds. Models evaluated include:

Logistic regression: a linear method for binary classification.

Decision tree classifier: a nonlinear model that splits data according to certain conditions.

Random Forest Classifier: An ensemble of decision trees that provides higher accuracy and robustness.

Gradient boosted tree (GBT) classifier: Another ensemble method that builds trees in a sequential manner, focusing on correcting errors of previous trees.

Naive Bayes: A probabilistic classifier based on applying Bayes' theorem.

Linear Support Vector Machine (Linear SVC): A classifier that attempts to find the best margin that separates classes.

Performance metrics such as ROC area (AUC), accuracy, precision, recall, and F1 score are calculated for each model. The script also captures a confusion matrix to further understand the performance of each model.

```{model}
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

spark = SparkSession.builder.getOrCreate()

#load data
train_data = spark.read.csv('s3://natz-demo-bucket/train_data_cm.csv/', header=True, inferSchema=True)
test_data = spark.read.csv('s3://natz-demo-bucket/test_data_cm.csv/', header=True, inferSchema=True)

#replace negative, NaN, and infinity values
def replace_negative(df):
    for col_name in df.columns:
        df = df.withColumn(col_name, when(col(col_name) < 0, 0).otherwise(col(col_name)))
        df = df.na.fill({col_name: 0})
    return df

train_data = replace_negative(train_data)
test_data = replace_negative(test_data)

#assemble features
feature_columns = train_data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

#balancing classes by adjusting weights
balancing_ratio = train_data.filter(col('Cancelled') == 1).count() / train_data.count()
calculate_weights = udf(lambda x: 1 * balancing_ratio if x == 0 else (1 - balancing_ratio), DoubleType())
train_data = train_data.withColumn("classWeightCol", calculate_weights("Cancelled"))

#evaluators
evaluator_auc = BinaryClassificationEvaluator(labelCol='Cancelled', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='Cancelled', predictionCol='prediction', metricName='accuracy')

#list of classifiers
classifiers = [
    LogisticRegression(featuresCol='features', labelCol='Cancelled', weightCol='classWeightCol'),
    DecisionTreeClassifier(featuresCol='features', labelCol='Cancelled'),
    RandomForestClassifier(featuresCol='features', labelCol='Cancelled'),
    GBTClassifier(featuresCol='features', labelCol='Cancelled'),
    NaiveBayes(featuresCol='features', labelCol='Cancelled'),
    LinearSVC(featuresCol='features', labelCol='Cancelled')
]

#train and evaluate a model with cross-validation
def train_evaluate_model(classifier):
    paramGrid = ParamGridBuilder().build()
    pipeline = Pipeline(stages=[assembler, classifier])
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator_auc,
                              numFolds=5)
    cvModel = crossval.fit(train_data)
    predictions = cvModel.transform(test_data)

    #compute additional metrics using MulticlassMetrics
    rdd = predictions.select(['prediction', 'Cancelled']).rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = MulticlassMetrics(rdd)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    accuracy = evaluator_accuracy.evaluate(predictions)
    #confusion matrix
    confusion_matrix = metrics.confusionMatrix().toArray()
    confusion_df = spark.createDataFrame([Row(TP=float(confusion_matrix[1,1]), FP=float(confusion_matrix[0,1]), 
                                              FN=float(confusion_matrix[1,0]), TN=float(confusion_matrix[0,0]))])

    #save output
    confusion_matrix_path = f's3://natz-demo-bucket/cm_confusion_matrix/cm_{classifier.__class__.__name__}.csv'
    confusion_df.write.csv(confusion_matrix_path, mode='overwrite', header=True)

    return (classifier.__class__.__name__, evaluator_auc.evaluate(predictions), accuracy, precision, recall, f1Score)

#evaluate all classifiers
results = [train_evaluate_model(classifier) for classifier in classifiers]
results_df = spark.createDataFrame(results, ["Classifier", "AUC", "Accuracy", "Precision", "Recall", "F1 Score"])

#save output
results_df.write.csv('s3://natz-demo-bucket/evaluation_cm_results.csv', mode='overwrite', header=True)

spark.stop()
```
**The other 3 feature selection outputs also use this code to bulid models.**

## Sample Output
The output show the performance metrics of multiple machine learning classifiers applied to the dataset, with a specific focus on predicting flight cancellations. These metrics include area under the receiver operating characteristic curve (AUC), accuracy, precision, recall, and F1 score.

All evaluation results indicate that all models perform well in terms of predictions. However, this may be due to overfitting or the effect of high-level imbalance in the original dataset. Even if we rebalance the dataset and perform feature selection before building the model, we still cannot eliminate the possibility of overfitting. Furthermore, there is a severe class imbalance as canceled flights only account for 6% of the dataset. Even without effective prediction capabilities, making "uncanceled" predictions for all instances yields seemingly high accuracy.

**Chi-Square**
| Classifier | AUC | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.999998465291188 | 0.9998873926744520 | 0.9997861604149760 | 0.999988705853121 | 0.9998874228767300 |
| DecisionTreeClassifier | 0.9999995245112250 | 0.9998997477101710 | 0.9997995690735080 | 1.0 | 0.9998997744926080 |
| RandomForestClassifier | 0.9999995413913560 | 0.9998997477101710 | 0.9997995690735080 | 1.0 | 0.9998997744926080|
| GBTClassifier | 0.9999997958444130 | 0.9999043367234380 | 0.9998115639066540 | 0.9999971764632800 | 0.9999043615711380 |
| NaiveBayes | 0.9999873316655590 | 0.9976645452482000 | 0.9999992908047880 | 0.9953305761497270 | 0.997659471499042 |
| LinearSVC | 0.9999980081930630 | 0.9998496215652560 | 0.9996993837366600 | 1.0 | 0.9998496692723990 |

AUC: The AUC scores for all classifiers are very close to 1, indicating an almost perfect ability to differentiate between positive classes (cancelled) and negative classes (not canceled).

Accuracy: From 99.77% to 99.99%, higher accuracy indicates that the model correctly identifies the majority of true positives and true negatives.

Precision and Recall: These are also very high, with several classifiers achieving perfect recall (1.0), indicating no false negatives. Precision measures the accuracy of positive predictions, with values close to 1 indicating few false positives.

F1 score: the harmonic mean of precision and recall. All models are also close to 1, indicating a balanced performance between precision and recall, which is ideal in most scenarios.

**Chi-Square & logistic Regression**

| TP | FP | FN | TN |
| --- | --- | --- | --- |
| 1416647.0 | 303.0 | 16.0 | 1415887.0 |

True Positives (TP): The highest number is 1,416,647, indicating that almost all canceled flights were correctly identified.

False Positives (FP): Very low at 303, indicating that few non-cancelled flights were incorrectly classified as canceled.

False Negatives (FN): Extremely low at 16, meaning the model captured almost all cancellations.

True Negatives (TN): Also very high at 1,415,887, indicating strong performance in identifying non-canceled flights.
