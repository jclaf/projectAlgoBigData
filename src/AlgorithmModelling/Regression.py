from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

def tmp_indexing(train,test):
    str_indx_2 = StringIndexer(inputCol = "region_id", outputCol = "label",stringOrderType="frequencyDesc").fit(train)
    df_train = str_indx_2.transform(train)

    str_indx_3 = StringIndexer(inputCol = "region_id", outputCol = "label",stringOrderType="frequencyDesc").fit(test)
    df_test = str_indx_3.transform(test)
    return df_train,df_test

def regression_linear_train(data,str,vec,train) :
    lr = LinearRegression(labelCol="str_indx_label",featuresCol='features', maxIter=40, regParam=0.3, elasticNetParam=0.8)
    lr_pipeline = Pipeline(stages=[str,vec,lr])
    lrModel = lr_pipeline.fit(train)
    return lrModel.stages[2]

def regression_linear_test(model,test) :
    return model.transform(test)

def gradient_boost(vec,train,test):
    gbt = GBTRegressor(featuresCol = 'features', labelCol = 'label', maxIter=10)

    gbt_pipeline = Pipeline(stages=[vec,gbt])
    gbtModel = gbt_pipeline.fit(train)

    gbt_model = gbtModel.stages[1]
    return gbt_model.transform(test)

def evaluation_rg_train(model):
    print("Coefficients: %s" % str(model.coefficients))
    print("Intercept: %s" % str(model.intercept))
    trainingSummary = model.summary
    print("nombre Iterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("*** Evaluation sur les données d'entrainement ***")
    print("MSE: %.2f" % trainingSummary.meanSquaredError)
    print("RMSE: %.2f" % trainingSummary.rootMeanSquaredError)
    print("r2: %.2f" % trainingSummary.r2)
    print("MAE = %.2f" % trainingSummary.meanAbsoluteError)
    print("Explained variance = %.2f" % trainingSummary.explainedVariance)

def evaluation(data) :
    evaluator_rmse = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="rmse")
    evaluator_mse = RegressionEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="mse")
    evaluator_mae = RegressionEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="mae")
    evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="r2")
    evaluator_var = RegressionEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="var")

    print("*** Evaluation sur les données test ***")
    print("MSE = %.3g" % evaluator_mse.evaluate(data))
    print("RMSE = %.3g" % evaluator_rmse.evaluate(data))
    print("MAE = %.3g" % evaluator_mae.evaluate(data))
    print("r2 = %.3g" % evaluator_r2.evaluate(data))
    print("Explained variance = %.3g" % evaluator_var.evaluate(data))

def run_reg(data,str_indx,vec_indx,df_train,df_test) :
    train,test = tmp_indexing(df_train,df_test)
    print("******* Linear Regression *********") 
    lr_model = regression_linear_train(data,str_indx,vec_indx,train)
    evaluation_rg_train(lr_model)    
    lr_pred = regression_linear_test(lr_model,test)
    evaluation(lr_pred)
    print("******* Gradient-boosted tree regression *********") 
    gbt_pred = gradient_boost(vec_indx,train,test)
    evaluation(gbt_pred)
    
    