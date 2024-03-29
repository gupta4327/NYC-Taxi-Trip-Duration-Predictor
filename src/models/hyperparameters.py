#importing all necessary libraries
from hyperopt import hp
from hyperopt.pyll.base import scope
import pickle
from pathlib import Path

#nested dictionary for different hyperparameters of different models
hyperparameters = {
    'LinearRegression':
            {
            },
    'DecisionTree':
            {
               'max_depth': hp.choice('max_depth', [6,8,10 ]),
               'criterion': hp.choice('criterion', ['ginni', 'entropy']),
               'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None])
            },
    'RandomForest': 
           {
                'n_estimators' : hp.choice('n_estimators', [5,8,10]), 
                'max_depth': hp.choice('max_depth', [6,8,10 ]), 
                'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
                'max_samples': hp.uniform('max_sample',0.4,0.8)
           },
    'GradientBoost':
            {
                'n_estimators' : hp.choice('n_estimators', [15,20,30]), 
                'max_depth': hp.choice('max_depth', [4,6,8]), 
                'subsample': hp.uniform('max_sample',0.6,0.8),
                'learning_rate': hp.uniform('learning_rate',0.01,0.1),
                'verbose': hp.choice('verbose', [1])
            },
    'XtremeGradientBoost':
            {
                'n_estimators' : hp.choice('n_estimators', [15,20,30]), 
                'max_depth': hp.choice('max_depth', [4,8,10,12]), 
                'subsample': hp.uniform('max_sample',0.6,0.8),
                'learning_rate': hp.uniform('learning_rate',0.03,0.3),      
                'gamma':hp.uniform('verbose',0.09,0.4),
                'verbosity': hp.choice('verbosity', [1])
            }

}


if __name__ == '__main__':
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    hyperparameters_path = Path(home_dir.as_posix()+'/models/hyperparameters.pkl')
    pickle.dump(hyperparameters, open(hyperparameters_path, 'wb'))
