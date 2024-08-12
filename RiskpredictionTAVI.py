# -*- coding: utf-8 -*-
"""
@author: Laura Schepers 
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import Lasso,LassoCV,LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score,f1_score,precision_score,roc_auc_score,recall_score, roc_curve,auc
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler 
import xgboost as xgb
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


#%% Load data
print('Load data')
path= r'Z:\CENTER\CENTER2_clean.xlsx'
data_old=pd.read_excel(path)
data_old=data_old[data_old['Access']==2] #only data that has variable Access=2 is kept. This means that the model is made for transfemoral access
        
#%% Preprocessing - filling in NaNs
print('Preprocessing data')
data_irrelevant= ['CENTERID','Access','DateOfProcedure', 'ReasonSurgeryOpmerkingen','STUDYID', 'STUDY', 'Bleeding_date_anytype', 'YearOfProcedure','Valvetype_Comment','Sheathcomment','VascularcomplicationDate',
                  'YearOfProcedureQuartiles', 'Discharge_date', 'Death_date','Bleedingnotes','VascularcomplicationNotes', 'Bleeding_location_notes','Neworworseningconductiondisturbancesdate',
                 'MajorBleeding_date','MI_date','LastFollowUpdate', 'DOB', 'Stroke_date', 'NewAFdate', 'PermanentPacemaker_date', 'Coment_Arritmia_SP',
                 'BL_ECG','FU_ECG','OTHERECG', 'AOI', 'MVR', 'StudyPopulation']
data_relevant=data_old.drop(columns=data_irrelevant) # Remove irrelevant columns and or columns containing text 

list_features=pd.read_excel(r'Z:\CENTER\Output_features.xlsx')
output_features=list_features.iloc[:,0].tolist() #Set the features in excel to a list
data_relevant=data_relevant.drop(columns=output_features) 

deleted_features=[]
missing_percentage_features=(data_relevant.isnull().mean()*100).round(2)

#Delete features that consist for more than 50% of NaNs
for feature, percentage in missing_percentage_features.items(): 
    if percentage>502:
        deleted_features.append(feature) 
        data_relevant.drop(columns=feature, inplace=True) 
#122 features are deleted 

#Delete patients that have more than 50% missing features
missing_percentages_patient=(data_relevant.isnull().mean(axis=1)*100).round(2)
for patient, percentage in missing_percentages_patient.items(): 
    if percentage>50:
        deleted_features.append(patient) 
        data_relevant.drop(index=patient, inplace=True) 
#378 patients are deleted

#%% Split data into training and validation set, keep validation set seperate. 
X=data_relevant
y=data_relevant['Death_30Days'] #For other adverse outcomes use PermanentPacemaker_InHospital, Stroke, and MI
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.reset_index(drop=True)
y_train.reset_index(drop=True)
data_model=pd.concat([X_train, y_train],axis=1)
X_val.reset_index(drop=True)
y_val.reset_index(drop=True)
data_val=pd.concat([X_val, y_val],axis=1)

#%% Divide data in continuous data and categorical data. Categorical data should be set to a integer.
binary_features= ['EmergencySurgery','EarlyvsNew', 'Gender','DM', 'eGFR_lowerthan30', 'eGFRcategorial', 'HistoryMI', 'HistoryCABG','HistoryPCI', 
                  'HistoryPVD', 'HistoryCVAorTIA', 'Dyslipidemia','Hypertension', 'SignificantCAD','PreviousSurgery','HistoryAorticValveIntervention',
                  'BL_PM', 'BL_AF','ESvsMCV','viv','Postdilatation', 'Predilatation', 'DeviceSuccess', 'Death', 'Stroke', 'PermanentPacemaker_InHospital', 
                  'MI', 'LVEFlowerthan50','NYHA34', 'LowGrad','LFLG','LVEFbelow40', 'NYHA_34','Age_olderthan85', 'Chads_female','chads_HF',
                  'Chads_vascular','Chadsvasc_5orhigher','Underweight', 'Obesity','AccessNonTF','AccessTATAO','AccessDummy', 
                  'age_olderthan80', 'filter_$', 'BL_DAPT', 'Discharge_med_available','Death_30Days', 'Death','Stroke','MI','PermanentPacemaker_InHospital']
binary_data=data_model[binary_features] 

data_nans=data_model.drop(columns=binary_data)
continuous_features=['lengthM', 'AorticValveArea','LogisticEuroscore','BSA','AVA_index','eGFR', 'STSprom', 'Creatine'] 
data_categorical=data_nans.drop(columns=continuous_features)
columns_categorical=['FollowUptime','ValveType','ValveSize','SheathSize',  'age', 'lengthCM', 'weight','BMI','eGFR',
                     'NYHA','LVEF', 'MeanAorticValveGrad','PeakAorticValveGrad','STSprom','Creatine','ProcedureTime', 
                     'STSrisk', 'CHADSVASC','SheathCategories','Chads_age','CHADSVASCtertiles',]
data_continuous=data_nans[continuous_features]
columns_continuous=['lengthM','AorticValveArea','LogisticEuroscore','BSA', 'AVA_index', 'eGFR', 'STSprom', 'Creatine']

#%%Evaluate the imputers for NaNs
#Imputate categorical and continous data using K-nearest neigbors
print("KNN")
#To be able to evaluate the imputer, training and test sets need to be created for testing the imputation. 
#A mask of NaNs is lied over known values to calculate the R2 and MSE. 

train_data,test_data=train_test_split(data_nans,test_size=0.2, random_state=42)
test_data_original=test_data.copy()
imputer_KNN=KNNImputer(n_neighbors=5) #Evaluated which number of neighbors works the best
fitted_data=imputer_KNN.fit_transform(train_data) #.fit_transform -> learn the parameters and apply the transformation to new data
train_data_imputed=pd.DataFrame(fitted_data,columns=train_data.columns)
train_data_imputed[columns_categorical]=train_data_imputed[columns_categorical].astype('int64')

test_data_imputed=pd.DataFrame(imputer_KNN.transform(test_data),columns=test_data.columns) #.transform -> apply the learned transformation to new data 
test_data_imputed[columns_categorical]=test_data_imputed[columns_categorical].astype('int64')

#Evaluate the KNN using a mask of NaNs
test_evaluate=test_data_original.dropna()
mask=np.random.rand(*test_evaluate.shape)<0.2 #mask is 20% of the values
test_masked=test_evaluate.mask(mask)

test_data_imputed=pd.DataFrame(imputer_KNN.transform(test_masked),columns=test_masked.columns) #.transform -> apply the leraned transformation to new data 
test_data_imputed[columns_categorical]=test_data_imputed[columns_categorical].astype('int64')

original_values=test_evaluate[mask]
imputed_values=test_data_imputed[mask]

r2=r2_score(original_values,imputed_values)
mse=mean_squared_error(original_values,imputed_values)

print (r2)
print(mse)

data=imputer_KNN.transform(data_nans) #.transform -> apply the leraned transformation to new data 
data=pd.DataFrame(data,columns=data_nans.columns)
data[columns_categorical]=data[columns_categorical].astype('int64')

#Imputate the binary data using IterativeImputer
print("IterativeImputer")
#To be able to evaluate the imputer, training and test sets need to be created for testing the imputation. 
#A mask of NaNs is lied over known values to calculate the R2 and MSE. 
train_data,test_data=train_test_split(binary_data,test_size=0.2, random_state=42)
test_data_original=test_data.copy()

imputer=IterativeImputer(max_iter=50,random_state=42,min_value=0, max_value=1)
train_imputed=pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
train_imputed=train_imputed.round().astype(int)

test_imputed=pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
test_imputed=test_imputed.round().astype(int)

#Evaluate the IterativeImputer using a mask of NaNs
test_evaluate=test_data_original.dropna()
mask=np.random.rand(*test_evaluate.shape)<0.2 #mask is 20% of the values
test_masked=test_evaluate.mask(mask)

test_data_imputed=pd.DataFrame(imputer.transform(test_masked),columns=test_masked.columns) #.transform -> apply the leraned transformation to new data 
test_data_imputed=test_data_imputed.astype('int64')

original_values=test_evaluate[mask]
imputed_values=test_data_imputed[mask]

r2=r2_score(original_values,imputed_values)
mse=mean_squared_error(original_values,imputed_values)

print (r2)
print(mse)

#%%Fill in the NaNs using the imputers
# Now that the imputers are evaluated the NaNs need to be filled in

data_imputed=pd.DataFrame(imputer.transform(binary_data), columns=binary_data.columns)
binary_data=data_imputed.round().astype(int)

fitted_data=imputer_KNN.transform(data_nans)
KNN_imputed=pd.DataFrame(fitted_data,columns=data_nans.columns)
KNN_imputed[columns_categorical]=KNN_imputed[columns_categorical].astype('int64')

#Combine data 
binary_data.reset_index(drop=True,inplace=True)
KNN_imputed.reset_index(drop=True,inplace=True)
data_tot=pd.concat([KNN_imputed, binary_data],axis=1)

# Outputs that are not used for this risk prediction model, but can be used for training another model.
remove_outputs=['Death_30Days', 'Death','Stroke', 'MI','PermanentPacemaker_InHospital']
y_old=data_tot['Death_30Days'] #For other adverse outcomes use PermanentPacemaker_InHospital, Stroke, and MI

y_old=y_old.iloc[:,1]
data_totaal=data_tot.drop(columns=remove_outputs)

#%% Resample data using RandomOverSampler
print('Oversampling')
oversampler=RandomOverSampler(sampling_strategy='minority',random_state=42)
X,y=oversampler.fit_resample(data_totaal,y_old) 

#%% Remove double featurs and split data in continuous, categorical and binary 
#Remove double variables, for example NYHA, NYHA_34 and NYHA34, reduce that to only NYHA in the DataFrame.
remove_double=['SheathSize','STSrisk','AVA_index','lengthM', 'EmergencySurgery', 'eGFR_lowerthan30','eGFRcategorial','DeviceSuccess',
               'LVEFbelow40','NYHA34','Age_olderthan85','Chads_age','Chads_female','chads_HF','Chads_vascular','CHADSVASCtertiles',
               'Chadsvasc_5orhigher','AccessNonTF', 'AccessTATAO','AccessDummy','age_olderthan80','BL_DAPT','Discharge_med_available',
               'NYHA_34','LVEFlowerthan50','LowGrad','Underweight', 'Obesity', 'FollowUptime','BSA', 'Predilatation', 
               'ProcedureTime','Postdilatation','BMI','ValveType','HistoryCABG', 'Dyslipidemia','SignificantCAD','LFLG','filter_$','MeanAorticValveGrad',
               'LogisticEuroscore', 'STSprom']
data=X.drop(columns=remove_double)


#Create continuous DataFrame
continuous_variables=['age','lengthCM','weight','eGFR', 'LVEF','AorticValveArea','PeakAorticValveGrad']
data_continuous=data[continuous_variables]

#Create categorical and binary DataFrame
datarel_cat=data.drop(columns=continuous_variables)

#Create categorical DataFrame
categorical=['ValveSize','HistoryAorticValveIntervention','CHADSVASC', 'SheathCategories', 'NYHA']
data_categorical=datarel_cat[categorical]
categorical=['ValveSize','CHADSVASC', 'NYHA']

#Create binary DataFrame
data_binary=datarel_cat.drop(columns=categorical)
data_binary=data_binary.iloc[:,1:]
# Change two categorical features to binary features  
data_binary['SheathCategories']=data_categorical['SheathCategories'].apply(lambda x: 1 if x >=3 else 0)
data_binary['HistoryAorticValveIntervention']=data_categorical['HistoryAorticValveIntervention'].apply(lambda x: 1 if x >=1 else 0)

data_categorical=data_categorical.drop(columns=['SheathCategories','HistoryAorticValveIntervention'])

X=pd.concat([data_continuous, data_binary, data_categorical],axis=1)
#%% Feature selection 

#%% Lasso Regression with KFold cross validation
print('Start lasso regression')
cv=KFold(n_splits=10, shuffle=True, random_state=42) #Create 10 splits 
cv_precision=[]
cv_recall=[]
cv_f1=[]
cv_accuracy=[]
cv_auc=[]
selected_features_all=[]
feature_coef_all=[]
# Loop over 10 folds of the data cv.split(X) using training and testing indices. 
for train_i, test_i in cv.split(X): 
    # Select rows of data for training and test sets  
    X_train,X_test=X.iloc[train_i], X.iloc[test_i] 
    y_train,y_test=y.iloc[train_i], y.iloc[test_i]
    
    # Scale the continuous variables 
    scaler=StandardScaler()
    X_train_scaled=X_train.copy()
    X_test_scaled=X_test.copy()
    X_train_scaled[continuous_variables]=scaler.fit_transform(X_train[continuous_variables])
    X_test_scaled[continuous_variables]=scaler.transform(X_test[continuous_variables])
    
    categorical_transformer=OneHotEncoder(handle_unknown='ignore')
    preprocessor=ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)
        ],remainder='passthrough')
    
    X_train_transformed=pd.DataFrame(preprocessor.fit_transform(X_train_scaled[categorical]).toarray())
    X_test_transformed=pd.DataFrame(preprocessor.transform(X_test_scaled[categorical]).toarray())
    
    X_train1=pd.concat([X_train_scaled,X_train_transformed],axis=1)
    X_test1=pd.concat([X_test_scaled,X_test_transformed],axis=1)
    
    lasso_cv=LassoCV(alphas=np.logspace(-4,4,100),cv=5)
    lasso_cv.fit(X_train_scaled,y_train)
    best_alpha=lasso_cv.alpha_
    
    # Fit lasso model on training data with best alpha
    lasso_model=Lasso(alpha=best_alpha) # Create a vector based on the weights (based on 0 or 1), multiply script Lasso Y with the vector of weights. 
    #%Lasso
    lasso_model.fit(X_train_scaled,y_train)
    # Predict mortality using lasso model
    y_pred=lasso_model.predict(X_test_scaled)
    y_pred_bin=(y_pred>=0.5).astype(int) #Set y to 0 or 1
    
    # Evaluate model 
    cv_accuracy.append(accuracy_score(y_test,y_pred_bin))
    cv_precision.append(precision_score(y_test,y_pred_bin))
    cv_recall.append(recall_score(y_test,y_pred_bin))
    cv_f1.append(f1_score(y_test,y_pred_bin))
    cv_auc.append(roc_auc_score(y_test,y_pred_bin))
    
    # Select features that do not have coefficent 0 
    selected_features=X_train_scaled.columns[lasso_model.coef_ !=0]
    selected_features_all.extend(selected_features)
    
    feature_coef_list=list(zip(X.columns,lasso_model.coef_))
    feature_coef_all.extend(feature_coef_list)
    
mean_accuracy=np.mean(cv_accuracy)
mean_precision=np.mean(cv_precision)
mean_recall=np.mean(cv_recall)
mean_f1=np.mean(cv_f1)
mean_auc=np.mean(cv_auc)

print('Gemiddelde Accuracy:',mean_accuracy)
print('precision:', mean_precision)
print('Recall:', mean_recall)
print('F1:',mean_f1)
print('AUC:',mean_auc)
 
# Selected features and their frequency in the folds
feature_counter=Counter(selected_features_all)
selected_features_summary=pd.DataFrame(feature_counter.items(),columns=['Feature','Frequency'])
selected_features_summary=selected_features_summary.sort_values(by='Frequency',ascending=False)
print('Selected features:', selected_features_summary)

feature_coef_list=list(zip(X.columns,lasso_model.coef_))
results_Lasso=pd.DataFrame({
    'Feature': X.columns,
    'Coefficient':lasso_model.coef_})

#%% Wrapper
## Select which Model should be used for the wrapper
# Define the models
models = {
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree':DecisionTreeClassifier(random_state=42)
}

def evaluate_model(model, X, y):
    cv = KFold(n_splits=10, shuffle=True, random_state=42) 
    scores = {
        'accuracy': [], 
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    selected_features_all = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection using Sequential Feature Selector
        sfs = SFS(estimator=model, n_features_to_select=10, direction='backward', cv=3)
        sfs.fit(X_train, y_train)

        # Transform the datasets
        X_train_selected = sfs.transform(X_train)
        X_test_selected = sfs.transform(X_test)

        # Fit the model on the selected features
        model.fit(X_train_selected, y_train)

        # Predict the labels
        y_pred = model.predict(X_test_selected)
        
        # Collect scores for this fold
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred))
        scores['roc_auc'].append(roc_auc_score(y_test, y_pred))
        
        # Collect selected features
        selected_features_all.append(X.columns[sfs.get_support()])
    
    # Calculate average scores and unique selected features
    avg_scores = {metric: np.mean(scores[metric]) for metric in scores}
    unique_selected_features = np.unique(np.concatenate(selected_features_all))
    
    return avg_scores, unique_selected_features

results = {}

# Iterate over each model and evaluate
for name, model in models.items():
    print(f'Evaluating {name}')
    avg_scores, selected_features = evaluate_model(model, X, y)
    results[name] = {'scores': avg_scores, 'features': selected_features}

# Print the results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Scores: {result['scores']}")
    print(f"Selected Features: {result['features']}\n")

#%% Optimal number of features for Wrapper
#After selecting the model for wrapper, the optimal numer of features is calculated. 
#Perform cross validation and pick the maximum number of features.
print('Calculate optimal number of features')
cv=KFold(n_splits=10, shuffle=True, random_state=42) #Create 10 splits 
performance_scores=[]
num_features=[]
X_num=X.values
max_features=X_num.shape[1]

optimal_num_feature=None
best_performance=-np.inf

for n_feature in range(1,max_features+1):
    selected_features=X_num[:, :n_feature]
    fold_accuracies=[]
    
    for train_i, test_i in cv.split(selected_features):
        X_train,X_test=selected_features[train_i],selected_features[test_i]
        y_train,y_test=y.iloc[train_i],y.iloc[test_i]
        model=DecisionTreeClassifier(random_state=42)
        model.fit(X_train,y_train)
        accuracy=model.score(X_test,y_test)
        fold_accuracies.append(accuracy)
    
    average_accuracy=np.mean(fold_accuracies)
    performance_scores.append(average_accuracy)
    num_features.append(n_feature)
    
# Plot the performance of the model for each number of features
plt.figure(figsize=(20,18))    
plt.plot(num_features,performance_scores)
plt.xlabel('Number of features',fontsize=22)
plt.ylabel('Classifier performance (Accuracy)',fontsize=22)
plt.grid(True)
plt.tick_params(axis='both', which='major',labelsize=22)
plt.show()

# Print number of features that show the best performance
optimal_num=num_features[np.argmax(performance_scores)]
print('Optimal number of features:',optimal_num)

#%% Wrapper - backward elimination with Kfold cross validation
print('Start Wrapper')
cv=KFold(n_splits=10, shuffle=True, random_state=42) #Create 10 splits 
cvw_precision=[]
cvw_recall=[]
cvw_f1=[]
cvw_accuracy=[]
cvw_auc=[]
selected_features_all=[]

# Loop over 10 folds van de data cv.split(X) met behulp van de training en test indexen. 
for train_i, test_i in cv.split(X):
    print('Fold')
    # Select rows of data for training and test sets  
    X_train,X_test=X.iloc[train_i], X.iloc[test_i] 
    y_train,y_test=y.iloc[train_i], y.iloc[test_i] 
    
    sfs=SFS(
                    estimator=DecisionTreeClassifier (random_state=42),
                    n_features_to_select=10,
                    direction='backward',
                    scoring='r2',
                    cv=3
            )
    sfs=sfs.fit(X_train,y_train)
                    
    # Store transformed training and testing datasets with only the selected features
    X_train_selected=sfs.transform(X_train)
    X_test_selected=sfs.transform(X_test)
    
    # Fit wrapper model, DecisionTreeClassifier, on training data 
    model=DecisionTreeClassifier (random_state=42)
    model.fit(X_train_selected,y_train)
    
    # Predict mortality using wrapper model
    y_pred=model.predict(X_test_selected)
    y_pred_bin=(y_pred>=0.5).astype(int)
    
    # Evaluate model
    cvw_accuracy.append(accuracy_score(y_test,y_pred_bin))
    cvw_precision.append(precision_score(y_test,y_pred_bin))
    cvw_recall.append(recall_score(y_test,y_pred_bin))
    cvw_f1.append(f1_score(y_test,y_pred_bin))
    cvw_auc.append(roc_auc_score(y_test,y_pred_bin))
    
    # Retrieve the indices of selected features and crate a list of the selected feature names based on their indices
    selected_features=X.columns[sfs.get_support()]
    selected_features_all.extend(selected_features)

# Calculate mean evaluation metrices based on the 10 folds
mean_accuracy=np.mean(cvw_accuracy)
mean_precision=np.mean(cvw_precision)
mean_recall=np.mean(cvw_recall)
mean_f1=np.mean(cvw_f1)
mean_auc=np.mean(cvw_auc)

print('Gemiddelde Accuracy:',mean_accuracy)
print('precision:', mean_precision)
print('Recall:', mean_recall)
print('F1:',mean_f1)
print('AUC:',mean_auc)

# Selected features and their frequency in the folds
feature_counter=Counter(selected_features_all)
selected_features_summary=pd.DataFrame(feature_counter.items(),columns=['Feature','Frequency'])
selected_features_summary=selected_features_summary.sort_values(by='Frequency',ascending=False)
print('Selected features:', selected_features_summary)

#%% Input features
#The input features chosen by the wrapper are used as an input for the predictin model
wrapper_death=[ 'HistoryAorticValveIntervention', 'eGFR', 'age','lengthCM', 'weight','AorticValveArea','BL_AF','NYHA','LVEF']

wrapper_stroke=['age','lengthCM','weight','eGFR', 'LVEF','AorticValveArea','PeakAorticValveGrad','STSprom',
                'LogisticEuroscore','DM','HistoryMI','HistoryCVAorTIA', 'Hypertension', 'PreviousSurgery', 'ESvsMCV','CHADSVASC', 'NYHA']

wrapper_MI=['age','lengthCM','weight','eGFR', 'LVEF','AorticValveArea','PeakAorticValveGrad','STSprom',
                'LogisticEuroscore','Gender','DM','HistoryPCI','BL_AF','BL_PM', 'ESvsMCV',
                 'ValveSize', 'CHADSVASC', 'SheathCategories','NYHA']

wrapper_PPI=['age','lengthCM','weight','eGFR', 'AorticValveArea','PeakAorticValveGrad','STSprom',
                'LogisticEuroscore','DM','HistoryPCI','HistoryPVD', 'BL_AF','BL_PM', 'viv',
                'HistoryAorticValveIntervention','ValveSize',  'SheathCategories','NYHA']

selected_features=X[wrapper_death]

X_risk=selected_features

X_val1=X_val[wrapper_death]
#%% Risk prediction model with K-fold cross validation 
cv_precision=[]
cv_recall=[]
cv_f1=[]
cv_accuracy=[]
cv_auc=[]
cv_precision_val=[]
cv_recall_val=[]
cv_f1_val=[]
cv_accuracy_val=[]
cv_auc_val=[]
cv=KFold(n_splits=10, shuffle=True, random_state=42) #Create 10 splits 

# Impute the Nans of validation set, otherwise logistic regression does not work. 
#Use SingleImputer with strategy median as this influences the data the least
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')

X_val_clean=imputer.fit_transform(X_val1)
y_val_clean=y_val.fillna(0)


best_model=None
best_mean_auc=0
feature_importances_list=[]
coefficients_list=[]

# Loop over 10 folds van de data cv.split(X) met behulp van de training en test indexen. 
for train_i, test_i in cv.split(X): 
    # Select rows of data for training and test sets 
    X_train,X_test=X_risk.iloc[train_i], X_risk.iloc[test_i] 
    y_train,y_test=y.iloc[train_i], y.iloc[test_i] 
    
    # Fit model on training data
    model=LogisticRegression(max_iter=1000, random_state=42)
    #model=xgb.XGBClassifier(objective='binary:logistic',eta=0.1)
    #model=RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    
    # Predict mortality using model
    y_pred=model.predict(X_test)
    y_pred_bin=(y_pred>=0.5).astype(int)
    y_proba=model.predict_proba(X_test)[:,1]
    
    # Evaluate model
    cv_accuracy.append(accuracy_score(y_test,y_pred_bin))
    cv_precision.append(precision_score(y_test,y_pred_bin))
    cv_recall.append(recall_score(y_test,y_pred_bin))
    from sklearn.metrics import f1_score
    cv_f1.append(f1_score(y_test,y_pred_bin))
    cv_auc.append(roc_auc_score(y_test,y_proba))
    
    y_pred_val=model.predict(X_val_clean)
    y_pred_binary_val=(y_pred_val>=0.5).astype(int)
    yval_prob=model.predict_proba(X_val_clean)[:,1]
    
    cv_accuracy_val.append(accuracy_score(y_val_clean,y_pred_binary_val))
    cv_precision_val.append(precision_score(y_val_clean,y_pred_binary_val))
    cv_recall_val.append(recall_score(y_val_clean,y_pred_binary_val))
    from sklearn.metrics import f1_score
    cv_f1_val.append(f1_score(y_val_clean,y_pred_binary_val))
    cv_auc_val.append(roc_auc_score(y_val_clean,yval_prob))
    
    mean_auc=np.mean(cv_auc_val)
    if mean_auc>best_mean_auc:
        best_mean_auc=mean_auc
        best_model=model

# Calculate mean evaluation metrices based on the 10 folds
mean_accuracy=np.mean(cv_accuracy)
mean_precision=np.mean(cv_precision)
mean_recall=np.mean(cv_recall)
mean_f1=np.mean(cv_f1)
mean_auc=np.mean(cv_auc)

print('Gemiddelde Accuracy:',mean_accuracy)
print('precision:', mean_precision)
print('F1:',mean_f1)
print('Recall:', mean_recall)
print('AUC:',mean_auc)
mean_accuracy_val=np.mean(cv_accuracy_val)
mean_precision_val=np.mean(cv_precision_val)
mean_recall_val=np.mean(cv_recall_val)
mean_f1_val=np.mean(cv_f1_val)
mean_auc_val=np.mean(cv_auc_val)

print('Min accuracy', np.min(cv_accuracy), 'Max accuracy:',np.max(cv_accuracy))
print('Min precision', np.min(cv_precision), 'Max accuracy:',np.max(cv_precision))
print('Min Recall', np.min(cv_recall), 'Max accuracy:',np.max(cv_recall))
print('Min F1', np.min(cv_f1), 'Max accuracy:',np.max(cv_f1))
print('Min AUC', np.min(cv_auc), 'Max accuracy:',np.max(cv_auc))

print( 'Validation')
print('Gemiddelde Accuracy:',mean_accuracy_val)
print('precision:', mean_precision_val)
print('F1:',mean_f1_val)
print('Recall:', mean_recall_val)
print('AUC:',mean_auc_val)

print('Min accuracy', np.min(cv_accuracy_val), 'Max accuracy:',np.max(cv_accuracy_val))
print('Min precision', np.min(cv_precision_val), 'Max accuracy:',np.max(cv_precision_val))
print('Min Recall', np.min(cv_recall_val), 'Max accuracy:',np.max(cv_recall_val))
print('Min F1', np.min(cv_f1_val), 'Max accuracy:',np.max(cv_f1_val))
print('Min AUC', np.min(cv_auc_val), 'Max accuracy:',np.max(cv_auc_val))

logistic_fpr,logistic_tpr,_=roc_curve(y_test,y_proba)
logistic_roc_auc=auc(logistic_fpr,logistic_tpr)

plt.figure(figsize=(20,18))  
plt.plot(logistic_fpr,logistic_tpr,color='b',lw=2,label=f'(AUC={logistic_roc_auc:.2f})')
plt.xlabel('False positive Rate',fontsize=22)
plt.ylabel('True positive Rate',fontsize=22)
#plt.title('Area under the receiver operating curve training and test set', fontsize=24)
plt.tick_params(axis='both', which='major',labelsize=24)
print('AUC test:',logistic_roc_auc)


logistic_fpr_val,logistic_tpr_val,_=roc_curve(y_val_clean,yval_prob)
logistic_roc_auc_val=auc(logistic_fpr_val,logistic_tpr_val)

plt.figure(figsize=(20,18))  
plt.plot(logistic_fpr_val,logistic_tpr_val,color='b',lw=2,label=f'(AUC={logistic_roc_auc_val:.2f})')
plt.xlabel('False positive Rate',fontsize=22)
plt.ylabel('True positive Rate',fontsize=22)
#plt.title('Area under the receiver operating curve validation set')
plt.tick_params(axis='both', which='major',labelsize=24)
print('AUC validation:',logistic_roc_auc_val)
