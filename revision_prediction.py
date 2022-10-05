from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
import holoviews as hv
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler, FunctionTransformer as FT
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import *
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import TransformerMixin, BaseEstimator
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import confusion_matrix as plot_confusion_matrix
from IPython.display import display
from data_analysis import fix_categorical_information, drop_bathroom_outliers, drop_bedroom_outliers
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(funcName)s : %(message)s")

hv.extension('bokeh')
pd.set_option("display.max_rows",300)




class MeanEncoder(BaseEstimator, TransformerMixin):
    """Encoding target mean per group as feature for discrete variables"""
    def __init__(self):
        pass
    
    def fit(self, X,y):
        X = self._check_and_format(X)
        df = X.assign(target=y)
        self.mapping_dict = dict()
        for column in X.columns:
            self.mapping_dict[column] = df.get([column, 'target']).groupby(column).mean().to_dict()['target']
            
        return self
    
    def transform(self, X,y=None):
        X = self._check_and_format(X)
        for column in X.columns:
            mapper = self.mapping_dict[column]
            X[column] = X[column].map(mapper)
        
        return X
    
    @staticmethod
    def _check_and_format(X):
        if isinstance(X, pd.Series):
            X = X.copy().to_frame()
        elif isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            raise TypeError("Only Pandas DataFrame or Series type allowed")
        
        return X

    

def drop_revision_outliers(df_revision):
    """Remove the 5 properties from Richmond that had several revisions but don't look normal"""
    high_revision = (df_revision
                     .groupby('listing_id')
                     .count()
                     .query("revised_purchase_price > 150") # 5 Richmond properties
                    )
    
    df = (df_revision
          .copy()
          .query("listing_id not in @high_revision.index")
         )
          
    return df
    

def add_features(df_merged):
    df = (df_merged
          .copy()
          .assign(revision = lambda x: np.sign(x['revised_purchase_price'] - x['asking_price']),
                  time_listed = lambda x: (x['listing_dt'].max() - x['listing_dt']).dt.days,
                  day_listed = lambda x: x['listing_dt'].dt.weekday,
                  week_listed = lambda x: x['listing_dt'].dt.isocalendar().week,
                 )
          .fillna({"revision":0}) # Nans where price was not revised
         )
    return df


def sin_transformer(period):
    return FT(lambda x: np.sin(x/period*2*np.pi))

def cos_transformer(period):
    return FT(lambda x: np.cos(x/period*2*np.pi))


def plot_cm_proba(model, X, y_true):
    """Plot predicted probabilities as confusion matrix"""
    tn_index = np.where((y_true==0) & (model.predict(X)==0))[0]
    fp_index = np.where((y_true==0) & (model.predict(X)==1))[0]
    tp_index = np.where((y_true==1) & (model.predict(X)==1))[0]
    fn_index = np.where((y_true==1) & (model.predict(X)==0))[0]
    y_pred_proba = model.predict_proba(X)[:,1]
    
    tn = hv.Histogram(np.histogram(y_pred_proba[tn_index]), label='True negative')
    fp = hv.Histogram(np.histogram(y_pred_proba[fp_index]), label='False positive')
    tp = hv.Histogram(np.histogram(y_pred_proba[tp_index]), label='True positive')
    fn = hv.Histogram(np.histogram(y_pred_proba[fn_index]), label='False negative')
    
    return hv.Layout([tn,fp,fn,tp]).cols(2).opts(hv.opts.Histogram(show_grid=True, xlim=(0,1)))
    
    
if __name__ == '__main__':
    
    df_listing = (pd
                  .read_csv('listing.csv')
                  .assign(listing_dt = lambda x: pd.to_datetime(x['listing_dt']))
                  .pipe(fix_categorical_information)
                  .pipe(drop_bedroom_outliers)
                  .pipe(drop_bathroom_outliers)
                 )

    df_revision = (pd
               .read_csv('revision.csv')
               .assign(price_revision_dt = lambda x: pd.to_datetime(x['price_revision_dt']))
              )

    display(df_revision.get('listing_id').value_counts())
    
    high_revision = (df_revision
                     .groupby('listing_id')
                     .count()
                     .query("revised_purchase_price > 10")
                     .sort_values("revised_purchase_price", ascending=False)
                     .reset_index()
                    )
                     
    display(high_revision.merge(df_listing, on='listing_id'))
    
    top_listing_ids = high_revision['listing_id'].iloc[5:] 
    (df_revision
     .query("listing_id in @top_listing_ids")
     .groupby('listing_id')
     .apply(lambda x: display(hv.Scatter(x, 'price_revision_dt', 'revised_purchase_price',
                                        label=str(x['listing_id'].iloc[0]))
                              .sort('price_revision_dt')
                              .opts(width=400, show_grid=True)
                             )
           )
    )
    
    
    display(hv.Curve(df_revision.groupby('price_revision_dt').count(), 'price_revision_dt', 'revised_purchase_price').opts(width=600, show_grid=True, tools=['hover']))
    
    
    # Repeated listing id in df_revision some high number of changes on a single property
    # Top 5 repeated listings (total 1214 revisions) are from Richmond possibly 3 properties only. The revised values alternate between 2 levels
    # Calculate revision based on price since revisions happen without a change in price listing_id = 1887087, perhaps some other attribute gets revised
    # Other multiple revision properties look okay in terms of revision proces so we can keep them and only take last revision.
    # 
    # Can consider number of revisions as a value to Poisson regress upon in future
    # Many revisions happen on 26 Dec
    # Revisions seem to have a pattern by day of the week
    # Late 2020 many price revisions (perhaps due to pandemic and not due to propery attributes) we can weigh 2020 with low weightage since 2021 and 2022 revisions were fairly constant
    # 
    
    # Analysing revisions against listing features
    df_merged = (df_listing
                 .merge(df_revision
                        .assign(revision_count = lambda x: x
                                                             .assign(count=1)
                                                             .get(['listing_id', 'count'])
                                                             .groupby('listing_id')
                                                             .transform(np.sum)
                                                             .values
                                )
                        .pipe(drop_revision_outliers),
                        on='listing_id', how='left')
                 .fillna({'revision_count':0})
                 .pipe(add_features)
                 .query("revision < 1")
                 .assign(revision = lambda x: x['revision'].map({-1:1,0:0}),
                        log10_asking_price = lambda x: np.log10(x['asking_price']))
                 .drop(['day_listed', 'time_listed', 'week_listed'], axis=1)
                )
    
    display(hv
            .Scatter(df_merged.sample(frac=0.4, random_state=42), 'log10_asking_price', 'revision_count')
            .opts(alpha=0.5, show_grid=True, ylim=(-5,20), xrotation=90)
           )
    
    display(hv
            .Scatter(df_merged.sample(frac=0.4, random_state=42), 'bedroom_count', 'revision_count')
            .opts(alpha=0.5, show_grid=True, ylim=(-5,20), xrotation=90)
           )
    
    var = 'property_type'
    display(hv
            .Bars(df_merged
                  .groupby([var, 'location'])
                  .mean(),
                  kdims=[var, 'location'], vdims='revision')
            .sort('revision')
            .opts(alpha=0.5, show_grid=True, xrotation=90, width=800, height=350)
           )

    # It appears that the lower the price the higher number of revisions property has
    # Flats have more revisions on average compared to houses
    # Location may have an influence on number of revisions
    # Leasehold vs unknown are generally same across locations but freehold properties have generally lower proportions of revisions
    
    
    np.random.seed(42)
    df_dataset_ = (df_listing
                  .merge(df_revision
                         .pipe(drop_revision_outliers)
                         .groupby('listing_id')
                         .last(), 
                         how='left', on='listing_id')
                  .sort_values(['location', 'listing_dt'])
                  .pipe(add_features)
                  .assign(log10_asking_price = lambda x: np.log10(x['asking_price']))
                  .query("revision < 1") # < 2% data points where price increased so we'll drop them for now
                  .sample(frac=0.5, random_state=42)
                  .reset_index(drop=True)
                 )
    logging.warning("Using only a fraction of total data for training")
    
    features = ['property_type', 'tenure', 'bedroom_count', 'location', 'time_listed', 'log10_asking_price', 'day_listed', 'week_listed']
    
    target = 'revision'
    
    df_dataset = (df_dataset_
                  .get(features + [target])
                  .assign(revision = lambda x: x['revision'].map({-1:1,0:0}))
                  .dropna(subset=features)
                 )
    
    # Verify assumptions of LR
    
    X = df_dataset[features]
    y = df_dataset[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

    # Analyse class imbalance
    cb = ClassBalance(labels=['not revised', 'revised'])
    cb.fit(y_train, y_test)
    cb.show()
    
    sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

    estimator = LogisticRegressionCV(Cs=np.logspace(-8,4,20), cv=5, scoring='f1', n_jobs=-1, class_weight='balanced',max_iter=100, random_state=42)

    categorical_features = ['property_type', 'tenure', 'location']
    categorical_feature_engineering = ColumnTransformer([('me', MeanEncoder(), categorical_features), 
                                                         ('sine_day', sin_transformer(6), ['day_listed']), 
                                                        ('cos_day', cos_transformer(6), ['day_listed']),
                                                        ('sine_week', sin_transformer(53), ['week_listed']),
                                                        ('cos_week', cos_transformer(53), ['week_listed'])], 
                                                        remainder=MinMaxScaler())
    
    extended_columns = ['property_type', 'tenure', 'location', 'sin_day', 'cos_day', 'sin_week', 'cos_week', 'bedroom_count', 'time_listed', 'log10_asking_price']
    
    
    
    # -------------- Model building --------------- #
    dummy_model = DummyClassifier(strategy='stratified')
    dummy_model.fit(X_train, y_train)
    
    model = make_pipeline(categorical_feature_engineering, estimator)
    model.fit(X_train, y_train)
    
    # ------------- model analysis macro ------------- #
    # plot_confusion_matrix(dummy_model, X_train, y_train)
    
    plot_confusion_matrix(model, X_train, y_train)
    display(plot_cm_proba(model, X_train, y_train))

    display(pd.DataFrame(np.c_[extended_columns, model[-1].coef_[0]], columns=['name', 'coef']).sort_values('coef', ascending=False))
    
    display(model[-1].C_)
    RocCurveDisplay.from_estimator(model, X_train, y_train);
    
    df_X = (pd
            .DataFrame(model[:-1].transform(X_train), columns=extended_columns)
            .assign(target=y_train, 
                    proba=model.predict_proba(X_train)[:,1])
           )
    display(df_X.corr())
    
    # ------------ model analysis micro -------------- #
    
    display(X_train
            .assign(revision=y_train, y_pred=model.predict(X_train))
            .query("revision==0 and y_pred==1")
            .groupby(categorical_features)
            .count()
            .rename({'bedroom_count':'false positive count'}, axis=1)
            .sort_values(['false positive count'] + categorical_features, ascending=False)
            .get(['false positive count'])
            .head(10)
           )
    
    display(hv
            .Bars(model[0]
            .named_transformers_['me']
            .mapping_dict['location'])
            .sort('y')
            .opts(xrotation=90, width=600, alpha=0.5, ylabel='location_encoding', xlabel='location')
           )
    
    df_reading = (X_train
                  .assign(revision=y_train, y_pred=model.predict(X_train))
                  .query("revision==0") # Only take non revised properties since we're studying false positives
                  .query("location=='Reading' and property_type=='house' and tenure == 'freehold'")
                 )
    
    display(df_reading
            .query("y_pred==1")
            .drop('log10_asking_price', axis=1)
            .groupby(categorical_features)
            .agg(np.mean)
           )
    
    display(df_reading
            .query("y_pred==0")
            .drop('log10_asking_price', axis=1)
            .groupby(categorical_features)
            .agg(np.mean)
           )

    list_o = []
    for y_pred in [0,1]:
        list_o.append(hv.Histogram(np.histogram(df_reading.query("y_pred==@y_pred").get('time_listed'), density=True)).opts(alpha=0.2))
        
    hv.Overlay(list_o)
    
    
    # Focusing on false positives which are too high, the simple model may have missed information encoded in non linear form
    # Looking at the top contributors in False positives, houses and flats in Reading have a large number of fp
    # Digging down further and looking at numerical features in Reading data, we can see that most distinguishing feature in mispredicted instances compared to correctly predicted ones is time_listed
    # For all the instances 
    
    
    # --------------- Introducing new interactions ----------- # 
    interaction_features = ['location', 'time_listed']
    index_interaction_features = [i for i,col in enumerate(extended_columns) if col in interaction_features]
    interactions = ColumnTransformer([('interactions', PolynomialFeatures(interaction_only=True), index_interaction_features)], remainder='passthrough')
    
    model = make_pipeline(categorical_feature_engineering, interactions, estimator)

    model.fit(X_train, y_train)
    
    plot_confusion_matrix(model, X_train, y_train)
    display(plot_cm_proba(model, X_train, y_train))

    # ------------- Evaluating the update ----------- # 
    display(X_train
            .assign(revision=y_train, y_pred=model.predict(X_train))
            .query("revision==0 and y_pred==1")
            .groupby(categorical_features)
            .count()
            .sort_values(['bedroom_count'] + categorical_features, ascending=False)
            .head()
           )
    
    
    
    
    
