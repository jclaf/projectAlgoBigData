from pyspark.sql.functions import avg,col, count, isnan,when,lit,round,udf
from pyspark.sql.types import IntegerType,BooleanType,DateType,StringType,DoubleType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from Utils.Func import print_df, print_ps


def clear_data(data) :
    data.dropDuplicates()
    dataDrop = data.drop(
            'composition_rubber_leather_percent',
            'composition_wood_percent',
            'composition_yard_garden_green_waste_percent',
            'other_information_summary_of_key_solid_waste_information_made_available_to_the_public',
            'special_waste_agricultural_waste_tons_year',
            'special_waste_construction_and_demolition_waste_tons_year',
            'special_waste_industrial_waste_tons_year',
            'special_waste_medical_waste_tons_year',
            'waste_collection_coverage_rural_percent_of_geographic_area',
            'waste_collection_coverage_rural_percent_of_households',
            'waste_collection_coverage_rural_percent_of_population',
            'waste_collection_coverage_rural_percent_of_waste',
            'waste_collection_coverage_total_percent_of_geographic_area',
            'waste_collection_coverage_total_percent_of_households',
            'waste_collection_coverage_total_percent_of_population',
            'waste_collection_coverage_total_percent_of_waste',            
            'waste_collection_coverage_urban_percent_of_geographic_area',
            'waste_collection_coverage_urban_percent_of_households',
            'waste_collection_coverage_urban_percent_of_population',
            'waste_collection_coverage_urban_percent_of_waste',
            'waste_treatment_anaerobic_digestion_percent',
            'waste_treatment_compost_percent',
            'waste_treatment_other_percent',
            'waste_treatment_controlled_landfill_percent',
            'waste_treatment_incineration_percent',
            'waste_treatment_landfill_unspecified_percent',
            'waste_treatment_open_dump_percent',
            'waste_treatment_sanitary_landfill_landfill_gas_system_percent',
            'waste_treatment_waterways_marine_percent',
            'where_where_is_this_data_measured',
            'waste_treatment_unaccounted_for_percent',
            )
    return dataDrop


def verif_null_value(data) :
    #verification de valeur None 
    for col in data.columns:
        print(col,"with None values: ", data.filter(data[col].isNull()).count())
    #verification de valeur NA 
    for col in data.columns:
        print(col,"with NA values: ", data.filter(data[col]=="NA").count())

def replace_value(data,str_in,str_out) :
    return data.replace(str_in,str_out)

def convert_str_to_double(data) :
    col_str = ['gdp','composition_food_organic_waste_percent','composition_glass_percent','composition_metal_percent','composition_other_percent','composition_paper_cardboard_percent','composition_plastic_percent','special_waste_e_waste_tons_year','special_waste_hazardous_waste_tons_year','total_msw_total_msw_generated_tons_year','waste_treatment_recycling_percent']
    for col in col_str :
        data = data.withColumn(col,data[col].cast(DoubleType()))
    return data
        
#Find the avg of all numeric columns

def mean_of_pyspark_columns(df, numeric_cols, verbose=False):
    col_with_mean=[]
    for col in numeric_cols:
        mean_value = df.select(avg(df[col]))
        avg_col = mean_value.columns[0]
        res = mean_value.rdd.map(lambda row : row[avg_col]).collect()
        
        if (verbose==True): print(mean_value.columns[0], "\t", res[0])
        col_with_mean.append([col, res[0]])    
    return col_with_mean

def fill_missing_with_mean(df, numeric_cols):
    col_with_mean = mean_of_pyspark_columns(df, numeric_cols) 
    
    for col, mean in col_with_mean:
        df = df.withColumn(col, when(df[col].isNull()==True, 
        round(lit(mean),2)).otherwise(df[col]))
        
    return df

def categorizer(group):
    if group == "Yes" or group =="yes" :
        return 1
    else :
        return 0

def bucket_udf(data) :
    str_col = ['other_information_information_system_for_solid_waste_management','other_information_national_agency_to_enforce_solid_waste_laws_and_regulations',
           'other_information_national_law_governing_solid_waste_management_in_the_country','other_information_ppp_rules_and_regulations']
    bucket = udf(categorizer, IntegerType() )
    for col in str_col :
        data= data.withColumn(col,bucket(col))
    
    return data


def run_data(data):
    print_ps(data)
    #suppression des doublons
    df1 = clear_data(data)
    verif_null_value(df1)
    df2 = replace_value(df1,"NA",None)
    verif_null_value(df2)
    df2 = convert_str_to_double(df2)
    
    print(data.printSchema())
    #remplace les valeurs null par la moyenne de chaque colonne
    numeric_col = ['gdp','composition_food_organic_waste_percent','composition_glass_percent','composition_metal_percent','composition_other_percent','composition_paper_cardboard_percent','composition_plastic_percent','special_waste_e_waste_tons_year','special_waste_hazardous_waste_tons_year','total_msw_total_msw_generated_tons_year','waste_treatment_recycling_percent']
    df3 = fill_missing_with_mean(df2,numeric_col)
    df3 = bucket_udf(df3)
    verif_null_value(df3)

    df3.filter(df3.country_name.isNull()).collect()
    df3=df3.dropna()
    verif_null_value(df3)
    print_df(df3)
    
    return df3
    