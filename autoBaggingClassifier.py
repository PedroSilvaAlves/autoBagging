import numpy as np
import pandas as pd
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.post_processing_functions.basic import Mean, \
    StandardDeviation, Skew, Kurtosis

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
            self.datasets = []
            self.meta_data = {}
            for file_name, target in zip(file_name_datasets, target_names):
                dataset = pd.read_csv(file_name)
                #self._validateDataset(dataset,target)
                print(target)
                print([target])
                meta_features = self._metafeatures(dataset,[target],meta_functions_categorical,meta_functions_numerical,post_processing_steps)
                #print(dataset.head())
                #print(dataset.dtypes)
                #print(target.head())
                self.datasets.append(dataset)
                self.meta_data[file_name.split('.csv',1)[0]] = meta_features

    def _metafeatures(self, dataset,target,meta_functions_categorical,meta_functions_numerical,post_processing_steps):
        data, attributes = analyze_pd_dataframe(dataset,target)
        print(attributes)
        meta_features = {}
        for name, mfc in meta_functions_categorical.items():
            values = []
            for index, meta_information in attributes.items():
                column_type = meta_information['type']
                if column_type == 'categorical':
                    raw_value = mfc(data[:, index])[0]
                    values.append(raw_value)

            for pps_name, pps in post_processing_steps.items():
                metafeature_name = '%s:%s' % (name, pps_name)
                value = pps(values)[0]
                meta_features[metafeature_name] = value

        for name, mfc in meta_functions_numerical.items():
            values = []
            for index, meta_information in attributes.items():
                column_type = meta_information['type']
                if column_type == 'numerical':
                    raw_value = mfc(data[:, index])[0]
                    values.append(raw_value)

            for pps_name, pps in post_processing_steps.items():
                metafeature_name = '%s:%s' % (name, pps_name)
                value = pps(values)[0]
                meta_features[metafeature_name] = value

        print(len(meta_features))
        print(meta_features)
        return meta_features

    def predict(self, dataset, targetname):
        if self.type == 'predict':
            self._validateDataset(dataset,targetname)
            print("Erro, error não é um problema de Classifier")
    
    def printDatasets(self):
        print(self.datasets)

    def _validateDataset(self,dataset,targetname):
        if dataset[targetname].dtype != pd.object:
            if sorted(dataset[targetname].unique()) != [0,1]:
                return False
        return True

TargetNames = []
FileNameDataset = []

FileNameDataset.append('./datasets/heart.csv')
TargetNames.append('target')
#FileNameDataset.append('./datasets/sanfranciscocrime.csv')
#TargetNames.append('Category')

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
model.printDatasets()