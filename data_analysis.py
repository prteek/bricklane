import numpy as np
import pandas as pd
import holoviews as hv
from IPython.display import display
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, power_transform
from sklearn.pipeline import make_pipeline
from scipy import stats
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


def make_reg_plot(df, col_x, col_y, outlier_threshold=2):
    """Make a regression line plot on passed columns and return plot along with coefficient and intercept of regression"""
    df_data = df.copy().sort_values(col_x).reset_index(drop=True)
    if col_x == 'listing_dt':
        X = df_data.get(col_x) - df_data.get(col_x).iloc[0]
        X = X.dt.days.values.reshape(-1,1)
    else:
        X = df_data.get(col_x).reshape(-1,1)
        
    y = df_data.get(col_y).values
    # filter_index = np.where((y>=-outlier_threshold) & (y<=outlier_threshold))[0]
    # X,y = X[filter_index], y[filter_index]
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    X_plot = df_data.get(col_x)
    sc = hv.Curve((X_plot, y_pred))
    print("r2_score:", model.score(X,y))
    return sc, model.coef_, model.intercept_


def plot_daily_price(df, agg_func=np.mean, transform_func=lambda x:x, grouping_col='property_type'):
    """Plot daily price aggregated daily by agg_func, then transforming aggregated asking_price by transform_func and finally plot separately for each value in grouping_col"""
    agg = (df
           .get(['listing_dt', grouping_col, 'asking_price'])
           .groupby([grouping_col, 'listing_dt'])
           .aggregate(agg_func)
           .sort_values('listing_dt')
           .reset_index()
          )

    overlay_list = []
    for group, df_group in agg.groupby(grouping_col):
        df_group['asking_price'] = transform_func(df_group['asking_price'].values.reshape(-1,1))
        scatter = hv.Scatter(hv.Dataset(df_group, 'listing_dt', 'asking_price', label=group))
        dist = hv.Distribution(df_group, 'asking_price')
        overlay_list.append(scatter << dist)

    overlay = (hv.Overlay(overlay_list)
               .collate()
               .opts(hv.opts.Scatter(alpha=0.5, width=600, height=350, 
                                     show_grid=True, tools=['hover']))
              )

    return overlay


def growth_days(df_coef, days=365*2):
    """Estimate growth over numer of days using coefficients from box-cox transformation and line fit"""
    numerator = df_coef['normalize_lambda']*df_coef['normalized_std']*df_coef['coef']*days
    denominator = (df_coef['normalize_lambda']*df_coef['normalized_std']*(0 + df_coef['intercept']) 
                   + 1 + df_coef['normalized_mean']*df_coef['normalize_lambda']
                  )
    return (1 + numerator/denominator)**(1/df_coef['normalize_lambda'])


def inverse_transform(yt, df_coef):
    """Inverse transform from power domain to price domain for sanity check on data. 
    Assumption is that Power transform uses Box-Cox transformation yt = (y**l - 1)/l and then scaling transformed values by mean and std"""
    y = (yt*df_coef['normalized_std']*df_coef['normalize_lambda']
         + 1
         + df_coef['normalized_mean'] * df_coef['normalize_lambda']
        )**(1/df_coef['normalize_lambda'])

    return y
    

def qqplot(data):
    """Plot qq-plot for data to check normality"""
    x,y = stats.probplot(data)[0]
    
    scatter = hv.Scatter((x,y)).opts(width=400, height=350, show_grid=True, xlabel='Theoretical quantiles', ylabel='Sample quantiles')
    line = hv.Curve(zip([np.min(x), np.max(x)], [np.min(y), np.max(y)])).opts(color='red')
    
    overlay = scatter*line
    return overlay.opts(title='Q-Q Plot')
    

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
    
    grouping_col = 'property_type'
    for location, df_location in df_listing.groupby('location'):
        overlay = plot_daily_price(df_location, agg_func=np.median, grouping_col=grouping_col)
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
            .opts(hv.opts.BoxWhisker(width=800, height=400, xrotation=90, show_grid=True, show_legend=False, outlier_color='gray', outlier_alpha=0.2, ylabel='log10 (asking_price)', box_alpha=0.5))
            )
    
    
    # Which locations have experienced the fastest growth in asking prices over the period ?
    
    # We can group by date to remove any variation during the day since we're interested in longer term behavior
    # We will split the data into house and flat separately to assess growth rates
    
    coef_list = [] 
    grouping_col = 'property_type'
    for location, df_location in df_listing.groupby('location'):
        # First we plot prices in a transformed domain where they are approximately normal
        overlay = plot_daily_price(df_location, 
                                   agg_func=np.median, 
                                   transform_func=lambda x: power_transform(x, method='box-cox', standardize=True), 
                                   grouping_col=grouping_col)
        
        regs = []
        df_date_reduced = (df_location
                           .groupby([grouping_col, 'listing_dt'])
                           .median()
                           .reset_index()
                          )
        for group, df_group in df_date_reduced.groupby(grouping_col):
            # For each group that we're interested in fit a line to data and get coefficients
            normalize = PowerTransformer(method='box-cox', standardize=True).fit(df_group['asking_price'].values.reshape(-1,1))
            
            y_org = df_group['asking_price'].values.reshape(-1,1)
            y_transformed = (y_org**(normalize.lambdas_[0]) -1)/normalize.lambdas_[0]
            
            df_group['target'] = normalize.transform(df_group['asking_price'].values.reshape(-1,1))

            reg_, coef, intercept = make_reg_plot(df_group, 'listing_dt', 'target')
            display(f"{group} - Slope: {coef[0]}, Intercept: {round(intercept,2)}")
            
            coef_list.append({'location': location, 'group': group,
                             'coef': coef[0], 'intercept': intercept, 
                              'normalize_lambda': normalize.lambdas_[0],
                             'normalized_mean': np.mean(y_transformed),
                             'normalized_std': np.std(y_transformed)})
            regs.append(reg_)
            display(qqplot(df_group['target'].values))
            
        o = overlay.opts(hv.opts.Scatter(title=location, ylabel='Transformed asking_price'))*hv.Overlay(regs)
        display(o)
        
    
    # Estimating Growth rate from line fit 
    df_coef = (pd.
               DataFrame(coef_list)
               .assign(growth_975_days = lambda x: growth_days(x, days=(365*2 + 8*30)))
               .sort_values(['group', 'growth_975_days'], ascending=False)
              )

    display(df_coef)
    
    colormap = {'flat':'#377eb8', 'house':'#e41a1c'}
    # colormap = {'freehold':'#377eb8', 'leasehold':'#e41a1c', 'unknown':'984ea3'}
    for group, df_group in df_coef.groupby('group'):
        o = (hv
             .Bars(df_group, 'location', 'growth_975_days', label=group.capitalize())
             .opts(width=600, height=350, 
                   xrotation=90, tools=['hover'], 
                   color=colormap[group], alpha=0.5, show_legend=False,
                   ylabel='Growth (over 975 days)')
            )
        display(o*hv.HLine(1).opts(color='black', line_dash='dashed'))
    
    
    # ------------- Sanity check on calulated growth values ---------- # 
    list_res = []
    for i, row in df_coef.iterrows():
        yt = np.array([row['intercept'], row['coef']*975+row['intercept']])
        y = inverse_transform(yt, row)
        res = {'lower_pred': y[0], 'upper_pred':y[1], 'group':row.group, 
               'location':row.location, 'growth_pred': row.growth_975_days}
        df_location = df_listing.query("location==@row.location")
        df_group = (df_location
                    .groupby(['property_type', 'listing_dt'])
                    .median()
                    .reset_index()
                    .query("property_type == @row.group")
                    .sort_values('listing_dt')
                   )
    
        res['lower_act'] = df_group.iloc[:5]['asking_price'].median()
        res['upper_act'] = df_group.iloc[-5:]['asking_price'].median()
        res['growth_act'] = res['upper_act']/res['lower_act']
        
        list_res.append(res)
        
    df_res = pd.DataFrame(list_res)
    
    display(hv.Scatter(df_res, 'lower_act', 'lower_pred').opts(show_grid=True, height=350, width=400, xrotation=90))
    display(hv.Scatter(df_res, 'upper_act', 'upper_pred').opts(show_grid=True, height=350, width=400, xrotation=90))
    display(hv.Scatter(df_res, 'growth_act', 'growth_pred').opts(show_grid=True, height=350, width=400, xrotation=90))
    
    
    
    