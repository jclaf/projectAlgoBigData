import matplotlib.pyplot as plt
import seaborn as sns
import six


def print_df(data) :
    print(data.toPandas())
    
def print_ps(data) :
    print(data.printSchema())
    
def print_describe(data) :
    print(data.describe().toPandas().transpose())
    
def plot_matrix(data,namefile) :
    sns.set()
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(data, annot=True, linewidths=0.7, cmap="RdBu_r",fmt='.2g')
    plt.savefig("../data/image/" + namefile)

def verif_corr(data,str) :
    for i in data.columns:
        if not( isinstance(data.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to", str,"for", i, data.stat.corr(str,i))

def plot_cluster(data,namefile):
    sns.set()
    ax = sns.FacetGrid(data,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()
    plt.savefig("../data/image/" + namefile)
    
def plot_score(score,namefile):
    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(2,10),score)
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.savefig("../data/image/" + namefile)