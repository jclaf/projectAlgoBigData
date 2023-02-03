from pyspark.ml.feature import VectorAssembler
from Utils.Func import print_describe
from copy import copy

def select_numeric_data(data) :
    return [t[0] for t in data.dtypes if t[1] == 'double']

def create_vector(data,numCols) :
    assembler = VectorAssembler(inputCols = numCols, outputCol = 'features')
    return assembler.transform(data)

def run_dt(data) :
    numeric_features = copy(select_numeric_data(data))
    print_describe(data.select(numeric_features))
    #ajout de la colonne population
    numeric_features.append('population_population_number_of_people')
    
    return create_vector(data,numeric_features)
    
    