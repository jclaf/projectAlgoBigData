from pyspark.ml.feature import StandardScaler, PCA
from Utils.Func import print_df,plot_matrix,verif_corr
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation

def scaler_acp(data) :
    scaler = StandardScaler(
        inputCol = 'features', 
        outputCol = 'scaledFeatures',
        withMean = True,
        withStd = True).fit(data)
    return scaler.transform(data)

def acp(data) :
    n_components = 2
    pca = PCA(
        k = n_components, 
        inputCol = 'scaledFeatures', 
        outputCol = 'pcaFeatures'
    ).fit(data)
    print('Explained Variance Ratio', pca.explainedVariance.toArray())
    return pca.transform(data)

def data_without_other_information(data) : 
    X = data
    str_col = ['other_information_information_system_for_solid_waste_management','other_information_national_agency_to_enforce_solid_waste_laws_and_regulations',
           'other_information_national_law_governing_solid_waste_management_in_the_country','other_information_ppp_rules_and_regulations']
    for x in str_col :
        X = X.drop(x)
    return X  

def Corr_Matrix(data) : 
    correlation = Correlation.corr(data, 'features', 'pearson').collect()[0][0]
    return correlation.toArray().tolist()
    #print(str(correlation).replace('nan', 'NaN'))


def run_acp(data) :
    df_scaled = scaler_acp(data)
    print_df(df_scaled)
    df_pca = acp(df_scaled)
    print_df(df_pca)
    X = data_without_other_information(df_pca)
    corrmatrix = Corr_Matrix(X)
    plot_matrix(corrmatrix,"correlation_matrix")
    #verif_corr(X,'population_population_number_of_people')
    print_df(X)
    return X