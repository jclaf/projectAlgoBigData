import findspark
findspark.init()
findspark.find()
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext

from DataExploration.Data_Exploration import run_data
from DataExploration.Data_Transformation import run_dt
from ACP.ACP import run_acp
from AlgorithmModelling.Classification import run_cl
from AlgorithmModelling.Regression import run_reg
from AlgorithmModelling.Clustering import run_cluster

from Utils.Func import * 
import sys

def read_data(spark) :
    if len(sys.argv) > 1:
        df = sys.argv[1]
    else : 
        df = "../data/country_level_data_0.csv"
    return spark.read.csv(df,inferSchema=True,header =True)

def execute(data=None) :
    sc=SparkContext()
    spark = SparkSession(sc)
    data = read_data(spark)
    run(data)
    spark.stop()

def run(data) :
    print("****** Data Exploration *********")
    df = run_data(data)
    print("\n")
    print("****** Data Transformation ******")
    df_asm = run_dt(df)
    print_df(df_asm)
    print("\n")
    print("************* ACP  **************")
    X = run_acp(df_asm)
    print("\n")
    (df_train, df_test) = X.randomSplit([0.70,0.30])  
    print("****** Algorithm Modelling ******\n")
    print("***** Supervised Learning *******")
    str_indx,vec_indx = run_cl(X,df_train,df_test)
    run_reg(X,str_indx,vec_indx,df_train,df_test)
    print("***** Unsupervised Learning*******")
    run_cluster(X)
def main():
    execute()
    
if __name__ == "__main__" :
    main()