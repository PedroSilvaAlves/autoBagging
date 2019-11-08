import numpy as np
import pandas as pd
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator


class autoBaggingClassifier:

    def __init__(self,state):
        if state == 'create' or state  == 'predict':
            self.type = state
        else:
            self.type = 'error'
    
    def fit(self,
            file_name_datasets,
            target_names,
            meta_functions_categorical,
            meta_functions_numerical,
            post_processing_steps):
        # Por cada file abrir o csv e tirar para um array de DataFrames
        
        if self.type == 'create':
            meta_features = []
            for file_name, target in zip(file_name_datasets, target_names):
                print(file_name)
                dataset = pd.read_csv(file_name)
                if self._validateDataset(dataset,target):
                    meta_features.append(self._metafeatures(dataset,[target],meta_functions_categorical,meta_functions_numerical,post_processing_steps))
                
            self.meta_data = pd.DataFrame(meta_features)
            self.meta_data.to_csv('meta_data.csv')
            print(self.meta_data)


    def _metafeatures(self, dataset,target,meta_functions_categorical,meta_functions_numerical,post_processing_steps):
            
        #Instantiate metafunctions and post-processing functions
        entropy = Entropy()
        correlation = PearsonCorrelation()
        mutual_information =  MutualInformation()
        scorrelation = SpearmanCorrelation()
        _mean = Mean()
        _sd = StandardDeviation()
        _nagg = NonAggregated()

        #Run experiments
        metafeatures_values, metafeatures_names = metafeature_generator(
        dataset, # Pandas Dataframe
        target, # Name of the target variable
        [mutual_information, entropy, correlation, scorrelation], # Metafunctions
        [_mean, _sd, _nagg] # Post-processing functions
        )
        
        #print(metafeatures_names)
        
        metafeatures_values = np.array(metafeatures_values)
        #print(metafeatures_values)
        metafeatures_names = np.array(metafeatures_names)
        #meta_features = pd.DataFrame(columns = metafeatures_names)
        meta_featureExample = dict(zip(metafeatures_names,metafeatures_values))
        return meta_featureExample
        
        

    def predict(self, dataset, targetname):
        if self.type == 'predict':
            if self._validateDataset(dataset,targetname):
                print("Erro, error não é um problema de Classifier")

    def _validateDataset(self,dataset,targetname):
        if dataset[targetname].dtype != 'object':
            if sorted(dataset[targetname].unique()) != [0,1]:
                print("False, não é válido")
                return False
        #print("True, é valido")
        return True

TargetNames = []
FileNameDataset = []

FileNameDataset.append('./datasets/heart.csv')
TargetNames.append('target')
FileNameDataset.append('./datasets/categoricalfeatureencoding.csv')
TargetNames.append('target')
FileNameDataset.append('./datasets/sanfranciscocrime_split.csv')
TargetNames.append('Category')

# Metafeatures
post_processing_steps = {'mean': Mean(),
                         'std': StandardDeviation(),
                         'skew': Skew(),
                         'kurtosis': Kurtosis()}
meta_functions_categorical = {'entropy': Entropy()}
meta_functions_numerical = {'mean': basic_meta_functions.Mean(),
                            'std': basic_meta_functions.StandardDeviation(),
                            'skew': basic_meta_functions.Skew(),
                            'kurtosis': basic_meta_functions.Kurtosis()}
#



model = autoBaggingClassifier('create')
model.fit(FileNameDataset,TargetNames, meta_functions_categorical, meta_functions_numerical, post_processing_steps)