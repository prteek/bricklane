from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
import holoviews as hv
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from IPython.display import display
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(funcName)s : %(message)s")

hv.extension('bokeh')
pd.set_option("display.max_rows",300)


def fix_categorical_information(df_listing):
    """Filter rows and columns based on analysis of categorical_columns"""

    df = (df_listing
          .copy()
          .drop('listing_type', axis='columns')
          .replace({'Newcastle upon Tyne': 'Newcastle Upon Tyne'})
         )
    return df


def drop_bedroom_outliers(df_listing):
    """Drop incorrect data identified by analysis of bedroom data"""
    df = (df_listing
          .copy()
          .query("~ ((bedroom_count >= 10) & (property_type=='flat'))")
          .query("~ ((bedroom_count >= 10) & (asking_price <= 500000))")
          .query("~ ((bedroom_count == 0) & (asking_price >= 300000))")
          .reset_index(drop=True)
         )
    
    return df


def drop_bathroom_outliers(df_listing):
    """Drop incorrect data identified by analysis of bathroom data"""
    df = (df_listing
          .copy()
          .query("~ ((bathroom_count >= 15) & (bedroom_count <= 5))")
          .reset_index(drop=True)
         )
    
    return df

    
if __name__ == '__main__':
    
    df_listing_ = pd.read_csv('listing.csv').assign(listing_dt = lambda x: pd.to_datetime(x['listing_dt']))
  
    df_listing    
    
    df_listing.info() # No null values in data except in bathroom_count (82124)
    
    df_listing['listing_id'].unique().shape # One entry per listing
    
    # Analysing data in each column
    categorical_columns = ['listing_type', 'property_type', 'tenure', 'location']
    for col in categorical_columns[1:]:
        display(df_listing[col].value_counts())
        
    # All listing_type are buy only so we can drop this column
    # property_type : {house, flat}
    # tenure_type : {freehold, unknown, leasehold}
    # location : Newcastle upon Tyne has only 1 property so we should merge this in its proper place by renaming to Newcastle Upon Tyne
    
    # Analysing numeric columns
    numeric_columns = ['bedroom_count', 'bathroom_count']
    for col in numeric_columns:
        display(df_listing[col].value_counts().sort_index())
    
    # There are outliers in these columns 
    # To make a sanity check it's worth looking at prices and property_type of these
    df_bedroom_outlier = df_listing.query("bedroom_count > 10").sort_values('bedroom_count')
    
    display(df_bedroom_outlier)
    
    display(df_bedroom_outlier.query("(bedroom_count >=10) & (property_type=='flat')").sort_values('bedroom_count'))
    display(df_bedroom_outlier.query("(bedroom_count >=10) & (asking_price <= 500000)").sort_values('bedroom_count'))
    
    df_no_bedroom = df_listing.query("bedroom_count == 0").sort_values("asking_price")
    display(hv.Histogram(np.histogram(df_no_bedroom.get("asking_price"), bins=np.linspace(1000000,10000000,20))))
    
    display(df_no_bedroom.query("asking_price >= 300000").get("asking_price").value_counts().sort_index())
    
    
    # 182500 bedroom flat doesn't look right. We'll drop this
    # Multiple 46 bedroom flats in Newcastle seem to be duplicate entries with same price. Also these may not be flats afterall so we should remove these from data
    # On the same lines it's useful to just remove flats which have 10 or more bedrooms. It would remove 44 properties but if there is something wrong in this data it can introduce costly inconsistencies due to relatively high value of these properties
    # It also seems implausible to have a property with 10 or more bedroom priced under 500000 £. There are 31 such properties and we can drop these
    # There are also possible duplicates e.g. listing_id = 8666232 and 8664263, 17552583 and 17530497
    # There are many properties listed with 0 bedrooms, we can check their prices etc. to see if there is something non-sensical
    # Some 0 bedroom properties are more than 500000 £ (244) and one of them is 32000000 £. Most of the properties are under 300000. We can consider dropping properties that are more than 300000 £ and have 0 bedrooms. (Still some odd properties which have > 4 bathrooms but these are very few)
    
    
    df_bathroom_outlier = df_listing.query("bathroom_count >= 15").sort_values('bedroom_count')
    
    display(df_bathroom_outlier)
    
    # There are 4 properties which have 20 bathrooms but < 5 bedrooms. These can be removed from dataset
    
    # Next we can analyse price data
    df_listing = (df_listing
                  .pipe(fix_categorical_information)
                  .pipe(drop_bedroom_outliers)
                  .pipe(drop_bathroom_outliers)
                 )
    
    display(df_listing[['asking_price']].describe())
    display(hv.Histogram(np.histogram(np.log10(df_listing['asking_price'])))
            .opts(xlabel='log_asking_price'))
    
    
    # asking_price ranges from 100000 to 19000000
    # elongated tail so need to use medians instead of mean
    
    # Which locations have the highest property prices?
    # The prices may differ perhaps by property type
    
    var = 'bedroom_count'
    df_plot = (df_listing
               .get(['location', var, 'asking_price'])
               .groupby([var, 'location'])
               .median()
               .sort_values([var, 'asking_price'], ascending=False)
              )
    
    hv.Bars(df_plot, kdims=[var, 'location'], vdims='asking_price').opts(width=800, height=400, xrotation=90, show_grid=True).sort('asking_price')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    df_revision = pd.read_csv('revision.csv').assign(price_revision_dt = lambda x: pd.to_datetime(x['price_revision_dt']))
    
    ds_listing = hv.Dataset(df_listing
                            .sample(10000)
                            .get(['listing_dt', 'asking_price', 'location'])
                            .sort_values(['location', 'listing_dt'])
                           )
    
    ds_listing.to(hv.Curve, 'listing_dt', 'asking_price').overlay('location').opts(width=600, height=350, show_grid=True, show_legend=True)


    dfg = df_listing.groupby(['location', 'property_type']).mean().get('asking_price').reset_index()

    dsg = hv.Dataset(dfg)

    dsg.to(hv.Bars, ['property_type', 'location'], 'asking_price').opts(width=1600, height=400, xrotation=90).sort('asking_price')


    # No bedrooms faulty data
    # No bathrooms incorrect data
    # Repeated listing id in df_revision some high number of changes on a single property
    # price revision positive and negative, multiple revisions vs single revision
    
    np.random.seed(42)
    df_dataset = (df_listing
                  .merge(df_revision
                         .groupby('listing_id')
                         .last(), 
                         how='left', on='listing_id')
                  .sort_values(['location', 'listing_dt'])
                  .assign(revision = lambda x: np.sign(x['revised_purchase_price'] - x['asking_price']),
                         time_listed = lambda x: (x['listing_dt'].max() - x['listing_dt']).dt.days)
                  .fillna({"revision":0}) # Nans where price was not revised
                  .query("revision < 1") # < 2% data points where price increased so we'll drop them for now
                  .get(['property_type', 'tenure', 'bedroom_count', 'location', 'time_listed', 'revision'])
                  .reset_index(drop=True)
                  .sample(frac=1)
                 )
    
    
    # logging.warning("Using only a slice of total data")
    
    features = ['property_type', 'tenure', 'bedroom_count', 'location', 'time_listed']
    target = 'revision'
    
    df_dataset_ = df_dataset.dropna(subset=features)
    
    X = df_dataset_[features]
    y = df_dataset[target].map({-1:1,0:0}).values
    
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)
    sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

    estimator = LogisticRegression(C=0.1, class_weight='balanced', max_iter=500)
    ohe = ColumnTransformer([('ohe', OneHotEncoder(drop='if_binary', sparse=False), ['property_type', 'tenure', 'location'])], remainder='passthrough')
    model = make_pipeline(ohe, estimator)
        
    model.fit(X_train, y_train)
    
    confusion_matrix(y_train, model.predict(X_train))
    
    
    
    
    
    
    
    