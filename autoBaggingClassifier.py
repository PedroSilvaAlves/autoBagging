import numpy as np
import pandas as pd
import xgboost as xgb
import math as m
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_squared_error
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import Mean, StandardDeviation, Skew, Kurtosis
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator
import warnings

class autoBaggingClassifier:
        
    def __init__(self,state):
        if state == 'create' or state  == 'predict':
            self.type = state
        else:
            self.type = 'error'
        self.base_estimators = {    'Decision Tree (max_depth=1)': DecisionTreeClassifier(max_depth=1,random_state=0),
                           'Decision Tree (max_depth=2)': DecisionTreeClassifier(max_depth=2,random_state=0),
                           'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3,random_state=0),
                           'Decision Tree (max_depth=4)': DecisionTreeClassifier(max_depth=4,random_state=0),
                           'Naive Bayes': GaussianNB(),
                           'Majority Class': DummyClassifier(random_state=0)}
        self.grid = ParameterGrid({"n_estimators" : [50,100,200],
                           "bootstrap" : [True, False]
                         })
        self.pruning = ParameterGrid({'pruning_method' :["none"],
                        'pruning_cp': [0.10,0.20,0.30,0.50]})

    def fit(self,
            file_name_datasets,     # Nome de todos os ficheiros .CSV
            target_names,           # Nome dos targets de todas os datasets
            meta_functions,         # Meta - Functions
            post_processing_steps): 
        # Por cada file abrir o csv e tirar para um array de DataFrames
        self.datasets = []          # Vai conter todos os Datasets
        self.bagging_workflows = [] # Vai conter todos os Bagging workflows
        if self.type == 'create':
            x_meta = []      # Vai conter todas as Metafeatures, uma linha um exemplo de um algoritmo com um certo tipo de parametros
            y_meta = []
            for file_name, target in zip(file_name_datasets, target_names):
                print("Creating Meta-features for: ",file_name)
                try:
                    dataset = pd.read_csv(file_name)
                except FileNotFoundError:
                    print("Path do dataset está errado, deve conter uma pasta 'dataset' no path do ficheiro autoBagging")
                    quit()

                self.datasets.append(dataset)
                if self._validateDataset(dataset,target):
                    # MetaFeatures
                    meta_features_estematic = self._metafeatures(dataset,target,meta_functions,post_processing_steps)
                    
                    # É necessário dividir o dataset em exemplos e os targets
                    X = SimpleImputer().fit_transform(dataset.drop(target,axis=1))
                    y = dataset[target]
                    #scoring = 'accuracy'
                    # Criar base-models
                    for pruning in self.pruning:
                        for params in self.grid: # Combinações de Parametros
                            meta_features = meta_features_estematic.copy()
                            Rank = {}
                            for base_estimator in self.base_estimators: # Combinação dos algoritmos base
                                # Criar modelo
                                bagging_workflow = BaggingClassifier(base_estimator=self.base_estimators[base_estimator],
                                                random_state=0,
                                                **params)

                                # Avaliar Algoritmos

                                # IGNORE
                                #kfold = KFold(n_splits=4, random_state=0)
                                #cv_results = cross_val_score(bagging_workflow, X, y, cv=kfold, scoring=scoring)
                                #print(base_estimator," --> Score: %0.2f (+/-) %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
                                #Rank[base_estimator] = cv_results.mean()
                                #print(params)
                                #Rank_Params[base_estimator] = params
                                #print(base_estimator, " --> Score: %0.3f)" % Rank[base_estimator])
                                
                                # Treinar o modelo
                                bagging_workflow.fit(X,y)
                                predictions = bagging_workflow.predict(X) # Main Prediction
                                #print("Start pruning " + pruning['pruning_method'])
                                if pruning['pruning_method'] == 'bb':
                                    bb_index= self._bb(y, predictions, X, pruning['pruning_cp'])
                                    print(bb_index)
                                    bagging_workflow = bagging_workflow[bb_index]
                                    predictions = bagging_workflow.predict(X) # Prunned Prediction
                                else :
                                    if pruning['pruning_method'] == 'mdsq':
                                        mdsq_index = self._mdsq(y, predictions, X, pruning['pruning_cp'])
                                        print(mdsq_index)
                                        bagging_workflow = bagging_workflow[mdsq_index]
                                        predictions = bagging_workflow.predict(X) # Prunned Prediction
                                
                                # Adicionar ao array de metafeatures, landmark do algoritmo atual
                                predictions = bagging_workflow.predict(X)
                                Rank[base_estimator] = cohen_kappa_score(y,predictions)
                                # Adicionar ao array de metafeatures, as caracteriticas dos baggings workflows
                                meta_features['bootstrap'] = np.multiply(params['bootstrap'],1)
                                meta_features['n_estimators'] = params['n_estimators']
                                meta_features['pruning_method'] = pruning['pruning_method']
                                meta_features['pruning_cp'] = pruning['pruning_cp']
                                # Adicionar a lista de Workflows
                                self.bagging_workflows.append(bagging_workflow)

                            #print(sorted(Rank, key=Rank.__getitem__ ,reverse=True))
                            i = 1
                            for base_estimator in sorted(Rank, key=Rank.__getitem__ ,reverse=True):
                                meta_features['Algorithm:' + base_estimator] = i
                                Rank[base_estimator] = i
                                i=i+1
                                array_rank = []
                            for value in Rank.values():
                                array_rank.append(value)
                            #print(array_rank)
                            array_rank.append(params['n_estimators'])
                            array_rank.append(np.multiply(params['bootstrap'],1))
                            y_meta.append(array_rank)      # Este array contem o target
                            x_meta.append(meta_features)   # Este array a adiconar contem as metafeatures do dataset e o scores do algoritmo base a testar

            # Meta Data é a junção de todas as metafeatures com os scores dos respeticos algoritmos base
            self.meta_data = pd.DataFrame(x_meta)
            self.meta_data.to_csv('meta_data.csv') # Guardar Meta Data num ficheiro .CSV
            
            print("Meta-Data Created & Saved!")
            print("Loading Test Dataset")
            # Criar o target para o Meta Data
            y_meta = np.array(y_meta)
            # Get o test Dataset
            dataset = pd.read_csv('./datasets/test/weatherAUS.csv')
            dataset = dataset.drop('RISK_MM', axis=1)
            target_test = 'RainTomorrow'
            print("Creating Meta-features")
            meta_features = self._metafeatures(dataset,target_test,meta_functions,post_processing_steps)
            X_test = pd.DataFrame(meta_features,index=[0])
            print("Prepare data for Meta-Model")
            # Tratar dos dados para entrar no XGBOOST
            for f in self.meta_data.columns: 
                if self.meta_data[f].dtype=='object': 
                    lbl = LabelEncoder() 
                    lbl.fit(list(self.meta_data[f].values)) 
                    self.meta_data[f] = lbl.transform(list(self.meta_data[f].values))

            for f in X_test.columns: 
                if X_test[f].dtype=='object': 
                    lbl = LabelEncoder() 
                    lbl.fit(list(X_test[f].values)) 
                    X_test[f] = lbl.transform(list(X_test[f].values))

            self.meta_data.fillna((-999), inplace=True) 
            X_test.fillna((-999), inplace=True)

            self.meta_data=np.array(self.meta_data) 
            X_test=np.array(X_test) 
            self.meta_data = self.meta_data.astype(float) 
            X_test = X_test.astype(float)
            print("Constructing Meta-Model:")
            # Criar o Meta Model XGBOOST
            meta_model = xgb.XGBRegressor(  objective="reg:squarederror",
                                            colsample_bytree = 0.3,
                                            learning_rate = 0.1,
                                            max_depth = 5,
                                            alpha = 10,
                                            n_estimators = 100)
            self.meta_model = MultiOutputRegressor(meta_model)
            
            # Aplicar Learning algorithm
            self.meta_model.fit(self.meta_data,y_meta)

            # Prever o melhor algoritmo
            preds = self.meta_model.predict(X_test)
            #print(preds)
            print("Recommended Bagging workflow: ")
            print("Number of models: %1.0f" % preds[0][6])
            bootstrap = False
            if preds[0][7] > 0.5:
                bootstrap = True
            print("Bootstrap: ", bootstrap)
            algorithm_score = preds[0][0]
            for i in range(0,5):
                if preds[0][i] < algorithm_score:
                    algorithm_score = preds[0][i]
                    algorithm_index = i
            switcher = {
                0: "Decision Tree (max_depth=4)",
                1: "Decision Tree (max_depth=3)",
                2: "Naive Bayes",
                3: "Decision Tree (max_depth=2)",
                4: "Decision Tree (max_depth=1)",
                5: "Majority Class"}
            print("Algorithm : ", switcher.get(algorithm_index))

        return self
    def predict(self, dataset, targetname):
        if self.type == 'predict':
            if self._validateDataset(dataset,targetname):
                print("Erro, error não é um problema de Classifier")

    def _metafeatures(self, dataset,target,meta_functions,post_processing_steps):

        metafeatures_values, metafeatures_names = metafeature_generator(
            dataset, # Pandas Dataframe
            [target], # Name of the target variable
            meta_functions, # Metafunctions
            post_processing_steps # Post-processing functions
        )
        metafeatures_values = np.array(metafeatures_values)
        metafeatures_names = np.array(metafeatures_names)
        meta_features = dict(zip(metafeatures_names,metafeatures_values))

        meta_features['Number of Examples'] = dataset.shape[0]
        meta_features['Number of Features'] = dataset.shape[1]
        meta_features['Number of Classes'] = dataset[target].unique().shape[0]
        meta_features_allnames = ['Features.SpearmanCorrelation.Mean',
                                'Features.SpearmanCorrelation.StandardDeviation',
                                'Features.SpearmanCorrelation.Skew',
                                'Features.SpearmanCorrelation.Kurtosis',
                                'FeaturesLabels.SpearmanCorrelation.Mean',
                                'FeaturesLabels.SpearmanCorrelation.StandardDeviation',
                                'FeaturesLabels.SpearmanCorrelation.Skew',
                                'FeaturesLabels.SpearmanCorrelation.Kurtosis',
                                'Features.Mean.Mean',
                                'Features.Mean.StandardDeviation',
                                'Features.Mean.Skew',
                                'Features.Mean.Kurtosis',
                                'Features.StandardDeviation.Mean',
                                'Features.StandardDeviation.StandardDeviation',
                                'Features.StandardDeviation.Skew',
                                'Features.StandardDeviation.Kurtosis',
                                'Features.Skew.Mean',
                                'Features.Skew.StandardDeviation',
                                'Features.Skew.Skew',
                                'Features.Skew.Kurtosis',
                                'Features.Kurtosis.Mean',
                                'Features.Kurtosis.StandardDeviation',
                                'Features.Kurtosis.Skew',
                                'Features.Kurtosis.Kurtosis',
                                'Features.Entropy.Mean',
                                'Features.Entropy.StandardDeviation',
                                'Features.Entropy.Skew',
                                'Features.Entropy.Kurtosis',
                                'Features.MutualInformation.Mean',
                                'Features.MutualInformation.StandardDeviation',
                                'Features.MutualInformation.Skew',
                                'Features.MutualInformation.Kurtosis',
                                'FeaturesLabels.MutualInformation.Mean',
                                'FeaturesLabels.MutualInformation.StandardDeviation',
                                'FeaturesLabels.MutualInformation.Skew',
                                'FeaturesLabels.MutualInformation.Kurtosis',
                                'bootstrap',
                                'n_estimators',
                                'pruning_method',
                                'pruning_cp',
                                'Algorithm:Decision Tree (max_depth=4)',
                                'Algorithm:Decision Tree (max_depth=3)',
                                'Algorithm:Naive Bayes',
                                'Algorithm:Decision Tree (max_depth=2)',
                                'Algorithm:Decision Tree (max_depth=1)',
                                'Algorithm:Majority Class']
        for feature_name in meta_features_allnames:
            if not (feature_name) in meta_features:
                meta_features[feature_name] = np.nan
        return meta_features
    

    #
    # Prunning: Boosting-based pruning of models
    #
    def _bb(self,target, # Target names
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total n umber of models to cut off
        
        #targets = data[target]
        #print(preds.shape[0])
        prunedN = m.ceil((preds.shape[0] - (preds.shape[0] * cutPoint)))
        preds = pd.DataFrame(preds)
        #print(preds)
        #print(prunedN)
        weights = []
        for i in range(data.shape[0]):
            weights.append(1/data.shape[0])
        #print(weights)
        
        ordem = {}
        for i in range(prunedN):
            k = 1
            errors = []
            for x in range(len(weights)):
                pred = np.asscalar(np.array(preds.iloc[x]))
                print('target', target[x])
                erro = sum(
                        (
                          (
                            not
                            (pred == target[x])
                          )
                            * 1
                        )
                            * weights)
                print('erro =', erro)
                errors.append(erro)
            #for x , y in zip(np.transpose(preds), target):
            #    errors.append(sum((not (x == targets.any())* 1) * weights))
            #    erro = sum(((not (x == target[y])) * 1) * weights)
            #    print('erro =', erro)
            #    errors.append(erro)
            
            print('check 1 ')
            k = k+1
            for w in ordem.keys():
                print('w', w)
                errors[w] = max(errors) * 2
            #print(errors)
            print('check 2')
            k = k+1
            ordem[i] = np.argmin(errors)
            print(ordem)
            print('check 3')
            k = k+1
            errorU = min(errors)
            print('check 4')
            k = k+1
            predU = []
            for x in range(len(ordem)):
                predU.append(preds[ordem[x]] == target[x])
            print('predU = ',predU)
            print('check 5')
            k = k+1
            if errorU > 0.5:
                weights = []
                for i in range(data.shape[0]):
                    weights.append(1/data.shape[0])
            else:
                for w in range(len(weights)):
                    if predU[w].bool:
                        # Error % 0
                        
                        weights[w] = weights[w] / (2*errorU)
                    else:
                        # Error % 0
                        weights[w] = weights[w] / (2*(1 - errorU))

            for w in range(len(weights)):
                if weights[w] > 10.000e+300:
                    weights[w] = 10.000e+300
                    
        return ordem
    
    #
    # Prunning: Margin Distance Minimization
    #   
    def _mdsq(self,target, # Target names
                preds, # Predicts na training data
                data, # training data
                cutPoint): # ratio of the total n umber of models to cut off
        targets = data[target]
        
        prunedN = (preds.shape[1] - (preds.shape[1] * cutPoint))
        pred = [] # 1 ou -1 se acertar ou não
        ens = []
        o = []
        for i in range(data[targets].length):
            ens.append(0)
            o.append(0.075)

        for x, tg in zip(preds,data[targets]):
            if x == tg:
                pred.append(1)
            else:
                pred.append(-1)

        pred = pd.Dataframe(pred)
        
        ordem = []
        for i in range(1,prunedN):
            dist = []
            for x, y, z in zip(pred, ens, o):
                dist.append(m.sqrt(sum((((x + y) / i) - z)^2)))
            ens = ens + pred[min(dist)]
            pred = pd.Dataframe(pred[list(set(pred) - set(min(dist)))]) # Buscar o name em vez do valor, está mal.
            ordem[i] = min(dist) # Como inteiro, pelo nome
        return ordem   
    
    def _validateDataset(self,dataset,targetname):
        #if dataset[targetname].dtype != 'object':
            #if sorted(dataset[targetname].unique()) != [0,1]:
                #print("Não é válido o Dataset")
               # return False
        #print("True, é valido")
        return True



# MAIN FUNCTION 

warnings.simplefilter(action='ignore', category=FutureWarning)
TargetNames = []
FileNameDataset = []


FileNameDataset.append('./datasets/titanic.csv')
TargetNames.append('Survived')
FileNameDataset.append('./datasets/heart.csv')
TargetNames.append('target')
#FileNameDataset.append('./datasets/walmart.csv')
#TargetNames.append('TripType')

#FileNameDataset.append('./datasets/categoricalfeatureencoding.csv')
#TargetNames.append('target')
#FileNameDataset.append('./datasets/sanfranciscocrime_split.csv')
#TargetNames.append('Category')

post_processing_steps = [Mean(),
                         StandardDeviation(),
                         Skew(),
                         Kurtosis()]


meta_functions = [Entropy(),
                  #PearsonCorrelation(),
                  MutualInformation(),
                  SpearmanCorrelation(),
                  basic_meta_functions.Mean(),
                  basic_meta_functions.StandardDeviation(),
                  basic_meta_functions.Skew(),
                  basic_meta_functions.Kurtosis()]


model = autoBaggingClassifier('create')
model.fit(FileNameDataset,TargetNames, meta_functions, post_processing_steps)
