"""
Andrew Marin
CS 777
Final Term Project
This code is meant to be run on the Census_Data.csv file and performs data cleanup steps, feature selection,
and then runs Logistic Regression and Decision Tree Classifiers. The performance of the classifiers is then analyzed.
To run this code, all you need to do is call the script name and then provide the .csv dataset as the first parameter.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, col, when
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

if __name__ == "__main__":

    spark = SparkSession.builder \
    .appName("Andrew-Marin-Term_Project") \
    .getOrCreate()

    ### Load the Data ###
    data = spark.read.csv(sys.argv[1], header=True, inferSchema=True)

    # Rename columns with dots in their names
    data = data.withColumnRenamed("education.num", "education_num") \
            .withColumnRenamed("marital.status", "marital_status") \
            .withColumnRenamed("capital.gain", "capital_gain") \
            .withColumnRenamed("capital.loss", "capital_loss") \
            .withColumnRenamed("hours.per.week", "hours_per_week") \
            .withColumnRenamed("native.country", "native_country")

    # Print Schema
    #data.printSchema()

    # Get the string columns
    string_columns = [column for column, type in data.dtypes if type == 'string']
    # Get the numeric columns
    numeric_columns = [column for column, type in data.dtypes if type in ['int', 'double']]

    ############### Data Cleanup and Preparation ###############


    # Trim all string columns
    for col_name in string_columns:
        data = data.withColumn(col_name, trim(col(col_name)))

    # Handle '?' in all columns
    for col_name in data.columns:
        data = data.withColumn(col_name, when(col(col_name) == '?', None).otherwise(col(col_name)))

    # Check for missing/null values
    #data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns]).show()

    # categorical columns
    categorical_columns = ['workclass', 'occupation', 'native_country', 'marital_status', 'education', 'relationship', 'race', 'sex']

    # Limit categorical columns to just 10 unique values (need to do so for decision trees)
    def limit_categories(data, categorical_columns, top_n):
        for col_name in categorical_columns:

            # Get the count of each category
            category_counts = data.groupBy(col_name).count().orderBy(F.desc("count")).limit(top_n).collect()
            
            # top N categories
            top_categories = [row[col_name] for row in category_counts]

            # Replace categories not in the top N with 'Other'
            data = data.withColumn(col_name, when(col(col_name).isin(top_categories), col(col_name)).otherwise('Other'))
            
        return data

    # Apply to all categorical columns
    data = limit_categories(data, categorical_columns, top_n=10)

    # Handle Missing Values
    for col_name in categorical_columns:
        data = data.fillna({col_name: 'Unknown'})

    # Missing Values
    #data.select([count(when(col(column) == 'Unknown', column)).alias(column) for column in categorical_columns]).show()

    # Create indexers for all categorical columns so we can encode them
    indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid='skip') for col_name in categorical_columns]

    # Create a Pipeline
    pipeline = Pipeline(stages=indexers)

    # Fit and transform the data
    data = pipeline.fit(data).transform(data)

    # Drop the categorical columns
    data = data.drop(*categorical_columns)

    data.cache()

    # Binary Label for class
    data = data.withColumn("income_binary", when(col("income") == ">50K", 1).otherwise(0))
    data = data.drop("income")

    ############## End Data Cleanup and Preparation ###############

    ############### Feature Selection & Numeric Value Scaling ###############

    ### Feature Selection ###

    #  Features into a Vector (for Random Forest)
    assembler_rf = VectorAssembler(inputCols=numeric_columns + [f"{col}_index" for col in categorical_columns], outputCol="features_rf")
    data_rf = assembler_rf.transform(data)

    data_rf.cache()

    # Train a Random Forest model for feature selection
    rf = RandomForestClassifier(featuresCol='features_rf', labelCol='income_binary', numTrees=100)
    rf_model = rf.fit(data_rf)

    # Get feature importances
    importances = rf_model.featureImportances

    # Create a list of (index, importance) pairs
    feature_importances = [(i, importance) for i, importance in enumerate(importances)]

    # Sort the features by importance
    sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Select the top N features
    N = 10 
    top_n_features = sorted_features[:N]

    # Get the original feature names using the feature mapping for the RF
    feature_mapping_rf = {i: col for i, col in enumerate(numeric_columns + [f"{col}_index" for col in categorical_columns])}
    top_feature_names_rf = [(feature_mapping_rf[i], importance) for i, importance in top_n_features]

    # Display the top features
    print("Top Features from Random Forest:")
    for name, importance in top_feature_names_rf:
        print(f"Feature: {name}, Importance: {importance}")

    # Extract only the top feature names
    top_feature_cols = [name for name, _ in top_feature_names_rf]

    # subset_data to include top features
    subset_data = data.select(*top_feature_cols, "income_binary")

    ### Data Scaling for Numeric Features ###

    # Scaling numeric features
    top_numeric_features = [col for col in top_feature_cols if col in numeric_columns]

    # Prepare the data for scaling - only scale numeric features
    assembler = VectorAssembler(inputCols=top_numeric_features, outputCol="features")
    data_scaled = assembler.transform(subset_data)

    data_scaled.cache()

    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scalerModel = scaler.fit(data_scaled)
    data_scaled = scalerModel.transform(data_scaled)

    # Define categorical features
    categorical_features = ['marital_status_index', 'relationship_index', 'occupation_index', 'education_index', 'sex_index']

    # Filter categorical features to include only those that are in data_scaled
    existing_categorical_features = [col for col in categorical_features if col in data_scaled.columns]

    # Assemble both the scaled numeric features and existing categorical features
    final_assembler = VectorAssembler(inputCols=['scaled_features'] + existing_categorical_features, outputCol="final_features")
    data_scaled = final_assembler.transform(data_scaled)

    # drop the intermediate columns
    data_scaled = data_scaled.drop(*top_numeric_features, 'features', 'scaled_features')
    # Drop the original categorical columns
    data_scaled = data_scaled.drop(*categorical_features)

    data_scaled.cache()


    ############### End of Feature Selection & Numeric Value Scaling ###############

    ############## Training & Running Models ###############

    # Train-Test Split (80-20)
    train_data, test_data = data_scaled.randomSplit([0.8, 0.2], seed=42)

    # Define the Logistic Regression model with the feature column
    lr = LogisticRegression(featuresCol='final_features', labelCol='income_binary')

    # Define the Decision Tree model with the feature column
    dt = DecisionTreeClassifier(featuresCol='final_features', labelCol='income_binary')

    # Create parameter grids for both models - Original
    # paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()
    # paramGrid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 20]).build()

    # Create parameter grids for both models - More hyperparameters to check
    paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100, 200]) \
    .build()
    
    paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 20]) \
    .addGrid(dt.maxBins, [16, 32, 64]) \
    .build()

    # Define Evaluators (accuracy, precision, recall, f1)
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="income_binary", predictionCol="prediction", metricName="accuracy")

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="income_binary", predictionCol="prediction", metricName="weightedPrecision")

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="income_binary", predictionCol="prediction", metricName="weightedRecall")

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="income_binary", predictionCol="prediction", metricName="f1")
    
    # Cross-Validation for Logistic Regression using parameter grids
    cv_lr = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid_lr,
                        evaluator=f1_evaluator,  # Use F1 score as a primary metric
                        numFolds=5)  # k = 5 cross-validation

    # Cross-Validation for Decision Tree
    cv_dt = CrossValidator(estimator=dt,
                        estimatorParamMaps=paramGrid_dt,
                        evaluator=f1_evaluator,  # Use F1 score as a primary metric
                        numFolds=5)  # k = 5 cross-validation

    # Fit the models
    cv_model_lr = cv_lr.fit(train_data)
    cv_model_dt = cv_dt.fit(train_data)

    # Predictions on the test set for both models
    predictions_lr = cv_model_lr.transform(test_data)
    predictions_dt = cv_model_dt.transform(test_data)

    ############### End of Training and Running Models ###############

    ############### Evaluation of Models ###############

    # Evaluate Models
    accuracy_lr = accuracy_evaluator.evaluate(predictions_lr)
    precision_lr = precision_evaluator.evaluate(predictions_lr)
    recall_lr = recall_evaluator.evaluate(predictions_lr)
    f1_lr = f1_evaluator.evaluate(predictions_lr)

    accuracy_dt = accuracy_evaluator.evaluate(predictions_dt)
    precision_dt = precision_evaluator.evaluate(predictions_dt)
    recall_dt = recall_evaluator.evaluate(predictions_dt)
    f1_dt = f1_evaluator.evaluate(predictions_dt)

    ### Print Metrics ###

    # Print metrics for Logistic Regression
    print("Logistic Regression Metrics:")
    print(f"Accuracy: {accuracy_lr}")
    print(f"Precision: {precision_lr}")
    print(f"Recall: {recall_lr}")
    print(f"F1 Score: {f1_lr}")

    print('\n')

    # Print metrics for Decision Tree
    print("Decision Tree Metrics:")
    print(f"Accuracy: {accuracy_dt}")
    print(f"Precision: {precision_dt}")
    print(f"Recall: {recall_dt}")
    print(f"F1 Score: {f1_dt}")

    print('\n')

    # Convert predictions to RDD and cast types explicitly
    prediction_and_labels_lr = predictions_lr.select(
        col("prediction").cast("double"),  
        col("income_binary").cast("double")
    ).rdd

    prediction_and_labels_dt = predictions_dt.select(
        col("prediction").cast("double"), 
        col("income_binary").cast("double")
    ).rdd

    # MulticlassMetrics for Logistic Regression
    metrics_lr = MulticlassMetrics(prediction_and_labels_lr)

    # MulticlassMetrics for Decision Tree
    metrics_dt = MulticlassMetrics(prediction_and_labels_dt)

    # Confusion matrices
    print("Confusion Matrix for Logistic Regression:")
    print(metrics_lr.confusionMatrix().toArray())

    print ('\n')

    print("Confusion Matrix for Decision Tree:")
    print(metrics_dt.confusionMatrix().toArray())

    ############### End of Model Evaluation ###############

    spark.stop()