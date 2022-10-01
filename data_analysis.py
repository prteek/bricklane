import numpy as np
import pandas as pd
import holoviews as hv
from IPython.display import display
from sklearn.linear_model import LinearRegression, RANSACRegressor
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


def make_reg_plot(df_data, col_x, col_y):
    """Make a regression line plot on passed columns and return plot along with coefficient and intercept of regression"""
    df_data = df_data.copy().sort_values(col_x).reset_index(drop=True)
    if col_x == 'listing_dt':
        X = df_data.get(col_x) - df_data.get(col_x).iloc[0]
        X = X.dt.days.values.reshape(-1,1)
    else:
        X = df_data.get(col_x).reshape(-1,1)
        
    y = df_data.get(col_y)
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    X_plot = df_data.sort_values(col_x).get(col_x)
    sc = hv.Curve((X_plot, y_pred))
    print(model.score(X,y))
    return sc, model.coef_, model.intercept_


def plot_daily_price(df, agg_func=np.mean, grouping_col='property_type'):
    """Plot daily price aggregated daily by agg_func, plot sparately for each value in grouping_col"""
    agg = (df
           .get(['listing_dt', grouping_col, 'asking_price'])
           .groupby([grouping_col, 'listing_dt'])
           .aggregate(agg_func)
           .sort_values('listing_dt')
           .reset_index()
          )

    o_ = (hv.Scatter(hv.Dataset(agg, kdims=['listing_dt', grouping_col], vdims='asking_price'))
          .groupby(grouping_col)
          .overlay()
          .opts(hv.opts.Scatter(alpha=0.5, width=600, height=350, show_grid=True, tools=['hover']))
         )

    dist = []
    for group, df_group in agg.groupby(grouping_col):
        dist.append(hv.Distribution(df_group, 'asking_price'))
    
    overlay = o_ << hv.Overlay(dist).opts(show_grid=True, width=150, height=350)
    return overlay


if __name__ == '__main__':
    
    df_listing = pd.read_csv('listing.csv').assign(listing_dt = lambda x: pd.to_datetime(x['listing_dt']))
  
    df_listing    
    
    df_listing.info() # No null values in data except in bathroom_count (82124)
    
    df_listing['listing_id'].unique().shape # One entry per listing
    
    # ------------ Categorical columns -------------- #
    categorical_columns = ['listing_type', 'property_type', 'tenure', 'location']
    for col in categorical_columns[1:]:
        display(df_listing[col].value_counts())
        
    # All listing_type are buy only so we can drop this column
    # property_type : {house, flat}
    # tenure_type : {freehold, unknown, leasehold}
    # location : Newcastle upon Tyne has only 1 property so we should merge this in its proper place by renaming to Newcastle Upon Tyne
    
    # ------------ Numeric columns -------------- #
    numeric_columns = ['bedroom_count', 'bathroom_count']
    for col in numeric_columns:
        display(df_listing[col].value_counts().sort_index())
    
    # There are outliers in these columns 
    # To make a sanity check it's worth looking at prices and property_type of these
    
    # ------------ Bedroom data -------------- # 
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
    
    # ------------ Bathroom data -------------- # 
    df_bathroom_outlier = df_listing.query("bathroom_count >= 15").sort_values('bedroom_count')
    
    display(df_bathroom_outlier)
    
    # There are 4 properties which have 20 bathrooms but < 5 bedrooms. These can be removed from dataset
    
    # ------------ Asking price data -------------- # 
    df_listing = (df_listing
                  .pipe(fix_categorical_information)
                  .pipe(drop_bedroom_outliers)
                  .pipe(drop_bathroom_outliers)
                 )
    
    display(df_listing[['asking_price']].describe())
    display(hv.Histogram(np.histogram(np.log10(df_listing['asking_price'])))
            .opts(xlabel='log_asking_price'))
    
    for location, df_location in df_listing.groupby('location'):
        overlay = plot_daily_price(df_location, np.mean, 'property_type')
        display(overlay.opts(hv.opts.Scatter(title=location)))
    
    # asking_price ranges from 100000 to 19000000, seems like capped at lower end specially for Wolverhampton
    # elongated tail so need to use medians instead of mean and probably log of asking price values
    # Gillingham has flat data which appears capped at the top starting around pandemic
    # Leicester has few outliers earlier in 2020 with very high prices
    
    # ------------ Questions -------------- #
    # Which locations have the highest property prices?
    # The prices may differ perhaps by property type
    
    var = 'property_type'
    df_plot = (df_listing
               .assign(log10_asking_price = lambda x: np.log10(x['asking_price']))
               .join(df_listing
                     .get(['location', var, 'asking_price'])
                     .groupby([var, 'location'])
                     .transform(np.median)
                     .rename({'asking_price':'median_price'}, axis=1)
                    )
                .get([var, 'location', 'asking_price', 'log10_asking_price', 'median_price'])
                .sort_values([var, 'median_price'], ascending=False)
              )
    

    display(hv
            .BoxWhisker(df_plot, kdims=['location', var], vdims='log10_asking_price')
            .groupby(var)
            .overlay()
            .opts(hv.opts.BoxWhisker(width=800, height=400, xrotation=90, show_grid=True, show_legend=False, outlier_color='gray', outlier_alpha=0.2, ylabel='log10 (asking_price)'))
            )
    
    
    # Which locations have experienced the fastest growth in asking prices over the period ?
    
    # We can group by date to remove any variation during the day since we're interested in longer term behavior
        
    coef_list = []
    
    for location, df_location in df_listing.groupby('location'):
        overlay = plot_daily_price(df_location, lambda x: np.log10(np.mean(x)), 'property_type')
        
        regs = []
        for property_type, df_property in df_location.groupby('property_type'):
            reg_, coef, intercept = make_reg_plot(df_property.assign(log10_asking_price=lambda x: np.log10(x['asking_price'])), 
                                                  'listing_dt', 
                                                  'log10_asking_price')
            display(f"{property_type} - Slope: {coef[0]}, Intercept: {round(intercept,2)}")
            
            coef_list.append({'location': location, 'property_type': property_type,
                             'coef': coef[0], 'intercept': intercept})
            regs.append(reg_)
            
        o = overlay.opts(hv.opts.Scatter(title=location))*hv.Overlay(regs)
        display(o.opts(title=location))
        
    
    df_coef = (pd.
               DataFrame(coef_list)
               .assign(growth_2_yr = lambda x: 10**(x['coef']*2*365))
               .sort_values(['property_type', 'growth_2_yr'], ascending=False)
              )

    display(df_coef)
    
    for property_type, df_property in df_coef.groupby('property_type'):
        o = (hv
             .Bars(df_property, 'location', 'growth_2_yr', label=property_type)
             .opts(width=600, height=350, xrotation=90, tools=['hover'])
            )
        display(o*hv.HLine(1).opts(color='black', line_dash='dashed'))
    
    
    
    
    
    
    
    
    
    