import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn.model_selection import cross_val_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# load the content of the housing file 
df = pd.read_csv("housing.csv")

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=200):
    ''' This function will be use to save our plots '''
    path = os.path.join(fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    ''' This class is a creat new features by combining 
    prexixtince features in the dataframe'''
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    
#simple way of splitibng data (most commonly use )
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis =1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

#test df
housing_test = test_set.drop("median_house_value", axis=1)#test
housing_lebel_test = test_set["median_house_value"].copy()#test
housing_prepared = full_pipeline.fit_transform(housing)
tes_prepared = full_pipeline.transform(housing_test) #test

#:::::::::::::::::: Fine tune our model 
#A::::::::::> Linear Support Vector Machine  Regressor (SVR)
#1:______________grid serach _________________________
#finding the best params using gride search
np.random.seed(42)

param_grid = [ # seting params for liner karmel
    {
        'kernel':['linear'],
        'C':[300., 1000., 3000., 10000., 30000.0],
        'epsilon':[ 0.03, 0.1, 0.3, 1.0, 3.0]
    }
  ]

svr_reg = SVR()

# Using Grid search to fing best params for prediction
grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_)) 
'''
#{'C': 30000.0, 'epsilon': 3.0, 'kernel': 'linear'}
#score for Grid search with SVR = 68999.58100756572
'''

# training the model with best estimator and compute the rmse for prediction
svr_regresor = SVR(kernel='linear',C = 30000.0,epsilon = 1.0)
svr_regresor_grid = svr_regresor.fit(housing_prepared, housing_labels)     
svr_prediction = svr_regresor.predict(tes_prepared)
print(svr_prediction)
rmse_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test,svr_prediction))
print(rmse_grid_serch) # rmse =75130.5547642227

#[ 58979.4325694  101257.87737045 243073.30850321 ... 441357.04461995
 #107241.78869219 179715.30570606]
#rmse 75130.5547642227


#2:_________________randomise search________________
# Using Ramomized search to fing best params for SVR
param_distribs = {
        'C': randint(low = 1, high=200),
        'epsilon': uniform(loc=0, scale=3),
    }

rnd_search = RandomizedSearchCV(svr_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print(rnd_search.best_params_) 
#Svar
#97499.64328496905 {'C': 103, 'epsilon': np.float64(2.3896289605806986)}
#114561.89194551535 {'C': 15, 'epsilon': np.float64(2.1959818254342154)}
#89192.06738910932 {'C': 189, 'epsilon': np.float64(1.790550473839461)}
#95180.27350580518 {'C': 122, 'epsilon': np.float64(0.46798356100860794)}
#101600.49412036216 {'C': 75, 'epsilon': np.float64(1.3777466758976016)}
#95758.29112908387 {'C': 117, 'epsilon': np.float64(1.8033450352296265)}
#92239.95512560439 {'C': 152, 'epsilon': np.float64(1.9526654188465586)}
#105617.55526884251 {'C': 53, 'epsilon': np.float64(2.909729556485983)}
#91727.45794408295 {'C': 158, 'epsilon': np.float64(0.6370173320348285)}
#88985.0455359325 {'C': 192, 'epsilon': np.float64(2.9766346778736525)}
# {'C': 192, 'epsilon': np.float64(2.9766346778736525)}


# training the model with best estimator(R_search) and comput the rmse for prediction
svr_regresor = SVR(kernel='linear', C = 192, epsilon = 2.90)
svr_regresor_random = svr_regresor.fit(housing_prepared, housing_labels)     
svr_prediction_random = svr_regresor_random.predict(tes_prepared)
print(svr_prediction_random)
rmse_random_serch = np.sqrt(mean_squared_error(housing_lebel_test,svr_prediction_random))
print(rmse_random_serch) # rmse = 74634.0863699671

#[ 61130.34130654  93917.25315715 224127.84362815 ... 433576.083937
#106529.23954161 180991.3500863 ]
#74634.0863699671
#B:::::::::::::::::::::::::::::::::::::::::> Random forest regressor 
#1:______________grid serach _________________________
#finding the best params using gride search

param_grid = [ # seting grid params for Grid search
    { 
        'n_estimators':[10, 20, 40],
        'max_features': [2, 4, 6, 8]  
    }
  ]


#1:______________grid serach _________________________
# Using Gride search to fing best params for RF
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_))
#result:
#{'max_features': 6, 'n_estimators': 40} <-------- best params 
#49321.015504575225  score

forest_reg =  RandomForestRegressor(max_features= 6, n_estimators= 40)
forest_reg_grid = forest_reg.fit(housing_prepared, housing_labels)     
forest_prediction = forest_reg_grid.predict(tes_prepared)
print(forest_prediction)
rmse_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test, forest_prediction))
print(rmse_grid_serch) # rmse = 49138.525653492834

#[ 48387.5 107405.  408360.3 ... 477813.2  75092.5 161472.5]
#49601.08796322671

#2:_________________randomise search________________
# Using Ramomized search to fing best params for RF
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }


forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

print(rnd_search.best_params_)
#result : {'max_features': 7, 'n_estimators': 180}

forest_reg =  RandomForestRegressor(max_features= 7, n_estimators= 180)
forest_reg_random = forest_reg.fit(housing_prepared, housing_labels)     
forest_prediction_r = forest_reg_random.predict(tes_prepared)
print(forest_prediction_r)
rmse_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test, forest_prediction_r))
print(rmse_grid_serch)
#result : rmse = 48706.551213856306

#C:::::::::::::::::::::::::::::::::::::::::> Decision tree
#1:______________grid serach _________________________
#finding the best params using gride search

param_grid = [ # seting grid params for Rna
    {   
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] 
    }
  ]


#1:______________grid serach _________________________
dtree_reg = DecisionTreeRegressor(random_state=42) # Initialize a decision tree regressor
grid_search = GridSearchCV(dtree_reg , param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_))


#result:
#{'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}
#60152.26205889712

dtree_reg = DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 4, min_samples_split =10)
dtree_reg_grid = dtree_reg.fit(housing_prepared, housing_labels)     
dtree_reg_prediction = dtree_reg_grid.predict(tes_prepared)
print(dtree_reg_prediction)
rmse_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test, dtree_reg_prediction))
print(rmse_grid_serch) # rmse = 49138.525653492834

# Result
#[ 57807.45423729 118262.5        487771.41463415 ... 498455.51515152
#76655.03875969 179638.96103896]
#rmse=60753.706052118716

#2:_________________randomise search________________
# Using Ramomized search to fing best params for RF
param_distribs = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

dtree_reg = DecisionTreeRegressor(random_state=42) # Initialize a decision tree regressor
dtree_reg_rnd_search = RandomizedSearchCV(dtree_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
dtree_reg_rnd_search.fit(housing_prepared, housing_labels)

print(dtree_reg_rnd_search.best_params_)
#result : {'max_depth': 15, 'min_samples_leaf': 19, 'min_samples_split': 13}


dtree_reg = DecisionTreeRegressor(max_depth = 15, min_samples_leaf = 19, min_samples_split= 13)
dtree_reg_reg_random = dtree_reg.fit(housing_prepared, housing_labels)     
dtree_reg_prediction_r = dtree_reg_reg_random.predict(tes_prepared)
print(dtree_reg_prediction_r)
rmse_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test, dtree_reg_prediction_r))
print(rmse_grid_serch)
#result : 
#[ 48579.48717949 148157.73076923 489450.75       ... 498527.26315789
#76510.52631579 189812.12121212]

# rmse = 58105.77174624186


#D:::::::::::::::::::::::::::::::::::::::::> Linear regression
#https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning-in-linear-regression/#2-fine-tuning-linear-regression-model-using-gridsearchcv

#1:______________grid serach _________________________
#finding the best params using gride search

param_dist = {'alpha': [0.1, 1.0, 10.0, 100.0]} # seting grid params for Rna
ridge = Ridge()
random_search = RandomizedSearchCV(ridge, param_dist, n_iter=10, cv=5)
# Using Grid search to fing best params for LR

grid_search = GridSearchCV(ridge, param_dist, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_)) 
#result
#{'alpha': 1.0}
#67865.92361380697 # RMSE: score
# Using best estimator(G_search) and compute the rmse for RIDGE  regression
ridge = Ridge(alpha= 1.0)
ridge_grid_best = ridge.fit(housing_prepared, housing_labels)     
ridge_prediction = ridge_grid_best.predict(tes_prepared)
print(ridge_prediction)
rmse_lin_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test,ridge_prediction))
print("test RMSE:", rmse_lin_grid_serch)

#result
#[ 61911.4886301  121776.20007617 267571.08037959 ... 447788.82686337
# 117272.96555885 185616.56219042]
#test RMSE: 72711.13753322628
#2:_________________randomise search________________
# Using Ramomized search to fing best params for Ridge regression



param_distribs = {
        'alpha': randint(low=1, high=200),
    }
ridge = Ridge()

# Using Grid search to fing best params for LR

rnd_search = RandomizedSearchCV(ridge, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

print("best parameters:", rnd_search.best_params_)
# best parameters: {'alpha': 15}
# Using best estimator(Randomize_search) and compute the rmse for RIDGE  regression
ridge = Ridge(alpha= 15)
ridge_grid_best = ridge.fit(housing_prepared, housing_labels)     
ridge_prediction = ridge_grid_best.predict(tes_prepared)
print(ridge_prediction)
rmse_lin_grid_serch = np.sqrt(mean_squared_error(housing_lebel_test,ridge_prediction))
print("test RMSE:", rmse_lin_grid_serch)

#[ 62261.40675337 120626.45453312 265574.67023995 ... 447223.03362221
# 117244.98789597 185816.02881636]
#test RMSE: 72741.47929505719
# Data for ploting Grid search 

rmse_GRID = np.array([75130.55, 49601.08, 60753.70, 172711.13])
models_all = ["SVR", "RF", "DT","LR"]

#font style
font1 = {'family':'serif','color':'blue','size':11}
font2 = {'family':'serif','color':'darkred','size':11}

# Bar settings
x = np.arange(len(models_all))  # the label locations
width = 0.1  # width of the bars

# Plot
plt.figure(figsize=(6, 6))
plt.bar(x - width, rmse_GRID, width=width, label='rmse_GRID', color='hotpink')
# Labels and title
plt.xticks(x, models_all)
plt.title(" Grid search RMSE comparison",fontdict = font1)
plt.ylabel(" Root Mean Square Error (RMSE) Values",fontdict = font2)
plt.xlabel("Machine Learning Models", fontdict = font2)
#plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_fig("Grid_Serche_rmse")
plt.show()

# Data for ploting Randomized saerch 
rmse_Rand = np.array([74634.08, 48706.55, 58105.77, 72741.47 ])
models_all = ["SVR",  "RF",  "DT", "LR",]

#font style
font1 = {'family':'serif','color':'blue','size':11}
font2 = {'family':'serif','color':'darkred','size':11}

# Bar settings
x = np.arange(len(models_all))  # the label locations
width = 0.1  # width of the bars

# Plot
plt.figure(figsize=(6, 6))
plt.bar(x, rmse_Rand, width=width, label='rmse_Rand ', color='red')

# Labels and title
plt.xticks(x, models_all)
plt.title("Randomize search RMSE comparison ",fontdict = font1)
plt.ylabel(" Root Mean Square Error (RMSE) Values",fontdict = font2)
plt.xlabel("Machine Learning Models", fontdict = font2)
#plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_fig("Randomize_searche_rmse")
plt.show()


#links:
#LR fine tunning : https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning-in-linear-regression/