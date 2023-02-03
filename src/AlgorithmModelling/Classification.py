from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.feature import IndexToString
from pyspark.ml import Pipeline

from Utils.Func import print_df, print_ps



def indexing(data) :
    str_indx = StringIndexer(inputCol = "region_id", outputCol = "str_indx_label",stringOrderType="frequencyDesc").fit(data)
    vec_indx = VectorIndexer(inputCol = "pcaFeatures", outputCol = "vec_indx_features", maxCategories = 4).fit(data)
    return str_indx,vec_indx

def indexing_label(str) :
    return IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=str.labels)
 
def decision_tree(data,str,vec,train,test) :
    dt_class = DecisionTreeClassifier(labelCol = "str_indx_label", featuresCol = "vec_indx_features",maxDepth=5,impurity= "gini")
    pipeline = Pipeline(stages=[str, vec, dt_class])
    model_class = pipeline.fit(train)
    return model_class.transform(test)

def random_forest(data,str,vec,label,train,test) :
    rf_waste = RandomForestClassifier(labelCol = "str_indx_label", featuresCol = "vec_indx_features",impurity= "gini",numTrees=3)
    pipeline = Pipeline(stages=[str, vec, rf_waste, label])
    rf_model = pipeline.fit(train)
    return rf_model.transform(test)



def evaluation(data) :
    print("*********** Evaluation ************")
    acc = MulticlassClassificationEvaluator(labelCol="str_indx_label",predictionCol="prediction",metricName="accuracy")
    f1 = MulticlassClassificationEvaluator(labelCol="str_indx_label", predictionCol="prediction", metricName="f1")
    precision = MulticlassClassificationEvaluator(labelCol="str_indx_label", predictionCol="prediction", metricName="precisionByLabel")
    recall = MulticlassClassificationEvaluator(labelCol="str_indx_label", predictionCol="prediction", metricName="recallByLabel")
    auc = BinaryClassificationEvaluator(labelCol="str_indx_label", rawPredictionCol="prediction")

    acc_val = acc.evaluate(data)
    f1_val = f1.evaluate(data)
    precision_val = precision.evaluate(data)
    recall_val = recall.evaluate(data)
    auc_val = auc.evaluate(data)

    print("Accuracy= %.2g " % acc_val)
    print("F1 = %.2g" % f1_val)
    print("Precision = %.2g" % precision_val)
    print("Recall = %.2g" % recall_val)
    print("AUC = %.2g" % auc_val)

def run_cl(data,df_train,df_test):
    str_indx,vec_indx = indexing(data)
     
    print("******* Decision Tree *********") 
    ar_pred = decision_tree(data,str_indx,vec_indx,df_train,df_test)
    #print_df(ar_pred)
    #print_ps(ar_pred)
    evaluation(ar_pred)
    label = indexing_label(str_indx)
    print("******* Random Forest *********") 
    rf_pred = random_forest(data,str_indx,vec_indx,label,df_train,df_test)
    #print_df(rf_pred)
    #print_ps(rf_pred)
    evaluation(rf_pred)
    
    return str_indx,vec_indx