from pyspark.ml.clustering import KMeans,GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator


from Utils.Func import plot_cluster, plot_score, print_df
import pandas as pd
import numpy as np

def kmeans_v1(data):
    kmeans = KMeans(featuresCol="pcaFeatures").setK(5).setSeed(10)
    k_model = kmeans.fit(data.select("region_id","pcaFeatures"))
    k_pred = k_model.transform(data)

    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='pcaFeatures',metricName='silhouette', distanceMeasure='squaredEuclidean')
    score = evaluator.evaluate(k_pred)
    getScore(k_model,score)
    return k_pred
    
def kmeans_v2(data):
    silhouette_score=[]
    evaluator_v2 = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaledFeatures', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
    for i in range(2,10):
        KMeans_algo=KMeans(featuresCol='scaledFeatures', k=i).setSeed(10)
        KMeans_fit=KMeans_algo.fit(data)
        output=KMeans_fit.transform(data)     
        score=evaluator_v2.evaluate(output)
        silhouette_score.append(score)
        getScore(KMeans_fit,score)
    return silhouette_score 

def GMM(data):
    gmm = GaussianMixture(featuresCol="pcaFeatures" ,tol=0.0001).setK(5).setSeed(10)
    gmm_model = gmm.fit(data.select('pcaFeatures'))
    gmm_pred = gmm_model.transform(data)
    print("Gaussians shown as a DataFrame: ")
    print_df(gmm_model.gaussiansDF)
    return gmm_pred    

def getScore(model,score):
    print("Within Set Sum of Squared Errors = %g" % model.summary.trainingCost)
    print("Silhouette with squared euclidean distance = %.2g" % score)
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)    

def print_cluster(data,pred):
    x_pca = np.array(data.rdd.map(lambda row: row.pcaFeatures).collect())
    cluster_kmeans = np.array(pred.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)
    pca_d = np.hstack((x_pca,cluster_kmeans))
    return pd.DataFrame(data=pca_d, columns=("1st_principal", "2nd_principal","cluster_assignment"))
    

def run_cluster(data) : 
    print("******* Kmeans version 1 *********")
    kmeans_pred = kmeans_v1(data)
    pca_df = print_cluster(data,kmeans_pred)
    plot_cluster(pca_df,"Cluster Kmeans v1")
    print("******* Kmeans version 2 *********")
    score = kmeans_v2(data)
    plot_score(score,"Best k with Kmeans_v2")
    print("******* Gaussian Mixture Model *********")
    g_pred = GMM(data)
    print_cluster(data,g_pred)
    plot_cluster(pca_df,"Cluster Gaussian Mixture Model")
    
    