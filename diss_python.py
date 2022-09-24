from IPython.core.display_functions import display
from lightgbm import LGBMClassifier, log_evaluation
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pylab as plt
import greenplum

import shap
import seaborn as sn

import warnings
warnings.filterwarnings("ignore")

#### setting width of dataframe #####

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',11)

# load data
schema = 'work_psingh'
user = 'psingh'
gp = greenplum.GPConnect(box=2, schema=schema, user=user, dbname='abc')
gp.connect()
df = pd.read_sql_query(f'SELECT * FROM {schema}.parameters_for_analysis_cln', con=gp)
gp.close()


# binarize outcome variable with median of return rate
df['return'] = (df['return_rate'] > df['return_rate'].median()).astype(int)

outcome = 'return'
#df.style_num= df.style_num.astype(int)
#df.tran_type= df.tran_type.astype(int)


# label encoding of variables *******

lab_size = LabelEncoder()
df['sku_size'] = lab_size.fit_transform(df['sku_size'])

lab_color = LabelEncoder()
df['sku_color'] = lab_color.fit_transform(df['sku_color'])


# one hot encoding categorical vars *************************

#import category_encoders as ce
#one_hot_en= ce.OneHotEncoder(cols=['sku_color'])
#df['sku_color']= one_hot_en.fit_transform(df['sku_color'])
# EDA ***************************************

df.head(10)


df.dtypes

df.shape

df.columns

print(df[predictors].info(show_counts='True'))

print((df.describe().T))

#  to check NaN values  ********************

df['receipted_rate'].isna().T
df[df.columns[df.isnull().any()]]
df[predictors].isnull().sum()

corr_mat= df[predictors].corr()
sn.heatmap(corr_mat, annot= True)
plt.show()

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_train = imp.fit(df ['receipted_rate'].values.reshape(-1, 1))

# function to define predictors/independent variables ****************

def dataset(pred):

    global predictors

    if (pred== 'All'):
        predictors = ['receipted_rate', 'sku_color', 'sku_size', 'avg_price_product', 'quantity', 'ecom_rate']

    if (pred== 'Footwear'):
        df['footwear'] = df['product_desc'].str.contains('shoes|cleats|footwear', case=False, regex=True)
        df['footwear'].astype(int)
        predictors= ['receipted_rate', 'sku_color', 'sku_size', 'avg_price_product', 'quantity', 'ecom_rate', 'footwear' ]

    if (pred== 'Brand'):
        df['brand'] = df['product_desc'].str.contains('NIKE| ADIDAS', case=False, regex=True)
        df['brand'].astype(int)
        predictors = ['receipted_rate', 'sku_color', 'sku_size', 'avg_price_product', 'quantity', 'ecom_rate', 'brand']


############################ checking for outliers in data ##########################

#for col in [x for x in predictors if x!= ['footwear']]:

def data_cleaning():
    print("Old Shape of dataset before data processing: ", df.shape)

    for col in predictors:
        #df.boxplot(column= [col], return_type= 'axes')
        #plt.show()
    #df.boxplot(predictors, rot=20, figsize=(9,9))
    #plt.show()

    ##### calculating quartiles and dropping rows containing outlier values ####
    ''' Detection '''
    # IQR
    Q1 = np.percentile(df[col], 25, interpolation='midpoint')

    Q3 = np.percentile(df[col], 75, interpolation='midpoint')

    IQR = Q3 - Q1

    # Upper bound
    upper = np.where(df[col] >= (Q3 + 0.5 * IQR))
    #print(df)
    # Lower bound
    lower = np.where(df[col] <= (Q1 - 0.5 * IQR))
    try:
      ''' Removing the Outliers '''
      df.drop(upper[0], inplace=True)
      df.drop(lower[0], inplace=True)
    except:

      print('Missing row')

    print("New Shape of dataset after data processing: ", df.shape)
    #print(df[predictors].describe().T)

#  Calculating Z-score  *******************************
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_receipt = imp.fit(df['receipted_rate'].values.reshape(-1, 1))
    df['receipted_rate'] = imp_receipt.transform(df['receipted_rate'].values.reshape(-1, 1))
    z_predictors= ['quantity', 'receipted_rate', 'avg_price_product', 'ecom_rate']
    for col_pred in z_predictors:
        z_pred= np.abs(stats.zscore(df[col_pred]))
        print("\nZ-score of %s is:\n %s" % (col_pred, z_pred))

# transformations ********************

df['quantity']=np.log(df['quantity'])

df['receipted_rate']=np.log(df['receipted_rate'])

df['avg_price_product']= np.log (df['avg_price_product'])



#X_train['sku_color'], fitted_lambda=scipy.stats.boxcox(X_train['sku_color'], lmbda= None)

#df['receipted_rate']= (df['receipted_rate']**1/3)

#X_train['sku_color']=(X_train['sku_color']**1/3)

#scaler=preprocessing.StandardScaler()
#df['quantity']= scaler.fit_transform(df['quantity'].values.reshape(-1,1))
#df['receipted_rate']= scaler.fit_transform(df['receipted_rate'].values.reshape(-1,1))

#df.boxplot(column= ['quantity'], return_type= 'axes')
#plt.show()

#  fit model  *****************************************************


def model_eval (ml_model):

    # splitting data *****************

    x_train, x_test, y_train, y_test = train_test_split(df[predictors], df[outcome], test_size=0.3, random_state=10)
    #print(x_train.describe().T)

    # fitting and predicting using models  ******************

    if (ml_model== 'LGBM'):
        mod = LGBMClassifier(learning_rate=0.09, max_depth=3, random_state=42)

        mod.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], categorical_feature=['sku_color', 'sku_size'], callbacks=[log_evaluation(period=10)])

        prediction = mod.predict(x_test)
        print('\n Prediction results of product returns is: \n %s' % prediction)

        # Plotting of Important features based on scores

        feat_importances = pd.Series(mod.feature_importances_, index=x_train.columns)
        print('\nFeature name score of its importance \n',feat_importances)
        feat_importances.plot(kind='barh')  # feat_importances.nlargest(15).plot(kind='barh') can be used to plot n features
        plt.title("Important features")
        plt.show()

    elif (ml_model== 'RF'):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_train = imp.fit(x_train ['receipted_rate'].values.reshape(-1, 1))
        x_train['receipted_rate'] = imp_train.transform(x_train['receipted_rate'].values.reshape(-1,1))
        imp_test = imp.fit(x_test['receipted_rate'].values.reshape(-1, 1))
        x_test['receipted_rate'] = imp_test.transform(x_test['receipted_rate'].values.reshape(-1, 1))
        np.all(np.isnan(x_train))
        x_train.isnull().sum()

        #if (pred=='Footwear'):
            #imp_train = imp.fit(x_train['receipted_rate'].values.reshape(-1, 1))
            #x_train['footwear'] = imp_train.transform(x_train['footwear'].values.reshape(-1, 1))
            #i#mp_test = imp.fit(x_test['receipted_rate'].values.reshape(-1, 1))
            #x_test['receipted_rate'] = imp_test.transform(x_test['receipted_rate'].values.reshape(-1, 1))

        mod = RandomForestClassifier(n_estimators=100, max_depth=3)
        mod.fit(x_train, y_train)
        prediction = mod.predict(x_test)
        print('\n Prediction results of product returns is: \n ' % prediction)

        # Plotting of Important features based on scores

        feat_importances = pd.Series(mod.feature_importances_, index=x_train.columns)
        print('\nFeature name with its importance score: %s \n', feat_importances)
        feat_importances.plot(kind='barh')  # feat_importances.nlargest(15).plot(kind='barh') can be used to plot n features
        plt.title("Important features")
        plt.show()

        #tree plotting
        from sklearn.tree import plot_tree
        #plt.figure(figsize=(15, 10))
        #plot_tree(mod.estimators_[0])
        #plt.show()

    elif (ml_model== 'LR'):

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_train = imp.fit(x_train['receipted_rate'].values.reshape(-1, 1))
        x_train['receipted_rate'] = imp_train.transform(x_train['receipted_rate'].values.reshape(-1, 1))
        imp_test = imp.fit(x_test['receipted_rate'].values.reshape(-1, 1))
        x_test['receipted_rate'] = imp_test.transform(x_test['receipted_rate'].values.reshape(-1, 1))


        mod= LogisticRegression()
        mod.fit(x_train, y_train)
        prediction = mod.predict(x_test)
        print('\n Prediction results of product returns is: \n %s' % prediction)


    # CV score *********************************

    cv1 = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_score_train = cross_val_score(mod, x_train, y_train, cv=cv1, scoring='accuracy')
    print(' CV score of dataset is: \n', cv_score_train)
    print("Summary of CV score for dataset is:\n Average: %.7g | Std: %.7g | Min: %.7g | Max - %.7g" % (np.mean(cv_score_train), np.std(cv_score_train), np.min(cv_score_train), np.max(cv_score_train)))

    cv_score_test = cross_val_score(mod, x_test, y_test, cv=cv1, scoring='accuracy')
    print('\n',cv_score_test)

    cv_score_data = cross_val_score(mod, df[predictors], df[outcome], cv=cv1, scoring='accuracy')
    print('\n', cv_score_data)

    #  accuracy scores ****************************

    print('\nTraining Accuracy: %.7g' % mod.score(x_train, y_train))
    print('\nTesting Accuracy: %.7g' % mod.score(x_test, y_test))
    print('\nModel Accuracy: %.7g' % mod.score(df[predictors], df[outcome]))

    print('\n Roc accuracy score of prediction is: \n %.7g' % roc_auc_score(y_test, prediction))
    print("\nAccuracy:", metrics.accuracy_score(y_test, prediction))

    # ROC curve *****************************

    y_pred_proba = mod.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    pred_prob= pd.DataFrame(y_pred_proba, prediction)
    pred_prob= pred_prob.reset_index()
    pred_prob.columns = ['Prediction', 'Prediciton Porbability']
    pred_prob_head= pred_prob.head(50)
    print (' \n Predicited result where 1 stands for predicted return and 0')
    print ('for predicted non-return along with its probability: \n\n', pred_prob_head)
    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #  confusion matrix  *********************
    metrics.plot_confusion_matrix(mod, x_test, y_test, cmap='Blues_r')
    plt.show()
    print('Metrics Classification report is as follows: \n',metrics.classification_report(y_test, mod.predict(x_test)))



def shap_plot(plot_model):
    x_train, x_test, y_train, y_test = train_test_split(df[predictors], df[outcome], test_size=0.3, random_state=10)
    df['return'].value_counts()


    if plot_model== 'LGBM':
        mod = LGBMClassifier(learning_rate=0.09, max_depth=3, random_state=42)
        mod.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], callbacks=[log_evaluation(period=10)])
        explainer = shap.TreeExplainer(mod)

    elif plot_model == 'RF':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_train = imp.fit(x_train['receipted_rate'].values.reshape(-1, 1))
        x_train['receipted_rate'] = imp_train.transform(x_train['receipted_rate'].values.reshape(-1, 1))
        imp_test = imp.fit(x_test['receipted_rate'].values.reshape(-1, 1))
        x_test['receipted_rate'] = imp_test.transform(x_test['receipted_rate'].values.reshape(-1, 1))

        mod = RandomForestClassifier(n_estimators=100, max_depth=3)
        mod.fit(x_train, y_train)
        explainer = shap.TreeExplainer(mod)

    elif plot_model== 'LR':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_train = imp.fit(x_train['receipted_rate'].values.reshape(-1, 1))
        x_train['receipted_rate'] = imp_train.transform(x_train['receipted_rate'].values.reshape(-1, 1))
        imp_test = imp.fit(x_test['receipted_rate'].values.reshape(-1, 1))
        x_test['receipted_rate'] = imp_test.transform(x_test['receipted_rate'].values.reshape(-1, 1))

        mod = LogisticRegression()
        mod.fit(x_train, y_train)
        explainer = shap.LinearExplainer(mod)

    shap_values = explainer.shap_values(df[predictors])
    expected_value = explainer.expected_value
    shap_array = explainer.shap_values(df[predictors])
    print('Shap values are: %s' % shap_values)

    dep_var=['return', 'non-return']
    shap.summary_plot(shap_values, df[predictors].values, plot_type="bar", class_names=dep_var, feature_names=predictors)

    shap.summary_plot(shap_values[1], df[predictors].values, feature_names = predictors)

    for i in range(len(predictors)):
        shap.dependence_plot(i, shap_values[0], df[predictors].values, feature_names=predictors)

    i=10
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], df[predictors].values[i], feature_names=predictors, matplotlib=True)

    row = 8
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][row],
    base_values=explainer.expected_value[0], data=x_test.iloc[row],
    feature_names=x_test.columns.tolist()))


    df_sub=df[predictors][0:1]
    shap.decision_plot(explainer.expected_value[1], shap_values[1], df_sub, ignore_warnings=True)


dataset(pred='All')
data_cleaning()
model_eval(ml_model='LGBM')
shap_plot(plot_model='LGBM')

df[predictors].isnull().sum()

