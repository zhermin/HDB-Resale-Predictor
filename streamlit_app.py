import os, time, random
from typing import Any
from datetime import datetime
from calendar import monthrange
import pandas as pd

import streamlit as st
### To test run the app on localhost
# $ pip install streamlit
# $ streamlit run streamlit_app.py

#--------------------------------- Page Config --------------------------------#

st.set_page_config(
    page_title='HDB Resale Price Predictor',
    page_icon='house.png', # Icon from Flaticon, created by surang
    layout='wide',
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    }
)

hide_streamlit_style = '''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
'''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#----------------------------- Data Preprocessing -----------------------------#

@st.cache
def load_data(data_file: str) -> tuple[pd.DataFrame]:
    '''Load and cache dataframes to avoid reloading constantly'''
    df = pd.read_csv(os.path.join(os.getcwd(), data_file))
    df_trim = df.drop(['block', 'street_name', 'remaining_lease', 'Year*', 'Month*'], axis=1)

    X = df_trim.drop(['resale_price'], axis=1)
    y = df_trim['resale_price']

    return df_trim, X, y

df, X, y = load_data('hdb_2017-2022.csv')

#------------------- Initial and Randomized State Variables -------------------#

def randomize() -> None:
    '''Randomize user input state variables'''
    st.session_state['month'] = datetime.strptime(random.choice(all_months), '%Y-%m')
    st.session_state['town'] = random.randint(0, len(df['town'].unique())-1)
    st.session_state['flat_type'] = random.randint(0, len(df['flat_type'].unique())-1)
    st.session_state['storey_range'] = random.randint(0, len(df['storey_range'].unique())-1)
    st.session_state['flat_model'] = random.randint(0, len(df['flat_model'].unique())-1)
    st.session_state['floor_area_sqm'] = random.randint(min_floor_area, max_floor_area)
    st.session_state['lease_commence_date'] = random.randint(min_lease_date, max_lease_date)

## Month Input Setup
all_months = sorted(df['month'])
first_month_strptime = datetime.strptime(all_months[0], '%Y-%m')
last_month = all_months[-1]
last_month_strptime = datetime.strptime(last_month, '%Y-%m')

## Floor Area Sqm Setup
min_floor_area = int(round(min(df['floor_area_sqm']), -1))
max_floor_area = int(round(max(df['floor_area_sqm']), -1))

## Lease Commence Date Setup
min_lease_date = int(sorted(df['lease_commence_date'])[0])
max_lease_date = int(sorted(df['lease_commence_date'])[-1])

#------------------------------- State Variables ------------------------------#

if 'month' not in st.session_state: st.session_state['month'] = last_month_strptime
if 'town' not in st.session_state: st.session_state['town'] = 0
if 'flat_type' not in st.session_state: st.session_state['flat_type'] = 0
if 'storey_range' not in st.session_state: st.session_state['storey_range'] = 0
if 'flat_model' not in st.session_state: st.session_state['flat_model'] = 0
if 'floor_area_sqm' not in st.session_state: st.session_state['floor_area_sqm'] = min_floor_area
if 'lease_commence_date' not in st.session_state: st.session_state['lease_commence_date'] = min_lease_date

if 'predictions' not in st.session_state: st.session_state['predictions'] = []

if 'prev_pred' not in st.session_state: st.session_state['prev_pred'] = 0
if 'prev_runtime' not in st.session_state: st.session_state['prev_runtime'] = 0

#----------------------------------- Sidebar ----------------------------------#

st.sidebar.write('# Choose your Model')

data_scaling = st.sidebar.selectbox(
    label='Select Scaling Method*',
    options=[
        'Raw',
        'Standard Scaling',
        'MinMax Scaling',
        'Normalization',
    ]
)

model_name = st.sidebar.selectbox(
    label='Select Model',
    options=[
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Decision Tree Regression',
        'XGBoost',
        'Support Vector Regression',
        'Artificial Neural Network',
    ],
    index=0,
)

st.sidebar.write("**Don't use Raw Scaling if using SVR*")

st.sidebar.write('# Choose your HDB')

randomize = st.sidebar.button(
    label='Randomize!',
    on_click=randomize,
)

month = st.sidebar.date_input(
    label='Select Month & Year',
    min_value=first_month_strptime,
    max_value=datetime.strptime(f'{last_month}-{monthrange(last_month_strptime.year, last_month_strptime.month)[1]}', '%Y-%m-%d'),
    value=st.session_state['month'],
).strftime('%Y-%m')

town = st.sidebar.selectbox(
    label='Select Town',
    options=sorted(df['town'].unique()),
    index=st.session_state['town'],
)

flat_type = st.sidebar.selectbox(
    label='Select Flat Type',
    options=sorted(df['flat_type'].unique()),
    index=st.session_state['flat_type'],
)

storey_range = st.sidebar.selectbox(
    label='Select Storey Range',
    options=sorted(df['storey_range'].unique()),
    index=st.session_state['storey_range'],
)

flat_model = st.sidebar.selectbox(
    label='Select Flat Model',
    options=sorted(df['flat_model'].unique()),
    index=st.session_state['flat_model'],
)

floor_area_sqm = st.sidebar.slider(
    label='Select Floor Area (sqm)',
    min_value=min_floor_area,
    max_value=max_floor_area,
    value=st.session_state['floor_area_sqm'],
)

lease_commence_date = st.sidebar.slider(
    label='Select Lease Commence Date',
    min_value=min_lease_date,
    max_value=max_lease_date,
    value=st.session_state['lease_commence_date'],
)

#---------------------------- User Input Test Data ----------------------------#

test = pd.DataFrame([[
    month,
    town,
    flat_type,
    storey_range,
    floor_area_sqm,
    flat_model,
    lease_commence_date,
]], columns=X.columns)

#-------------------------------- Data Scaling --------------------------------#

if data_scaling == 'Standard Scaling':
    from sklearn.preprocessing import StandardScaler
    scaling = StandardScaler()
elif data_scaling == 'MinMax Scaling':
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler()
elif data_scaling == 'Normalization':
    from sklearn.preprocessing import Normalizer
    scaling = Normalizer()
elif data_scaling == 'Raw':
    scaling = None

#---------------------------------- Encoders ----------------------------------#

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

def model_pipeline(X: pd.DataFrame, y: pd.Series, model: Any, scaling: Any) -> Pipeline:
    ordinal_cols = ['month', 'flat_type', 'storey_range']
    onehot_cols = ['town', 'flat_model']

    transformer = make_column_transformer(
        (OrdinalEncoder(), ordinal_cols),
        (OneHotEncoder(), onehot_cols),
        remainder='passthrough',
    )

    pipeline = make_pipeline(
        transformer,
        FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
        scaling,
        model,
    )

    pipeline.fit(X, y)
    return pipeline

#-------------------------------- Predictions ---------------------------------#

@st.cache(allow_output_mutation=True, show_spinner=False)
def model_load(model_name: str):
    '''Loads machine learning model saved in repo

    Flag 'allow_out_mutation=True' tells Streamlit that the output will change
    and that we are aware of it. This is necessary to load the Keras model. 

    Args
    ----
        model_name (str): Selected trained .h5 ANN model

    Returns
    -------
        loaded_model (Keras model): Trained machine learning model from my repo
    '''

    from keras.models import load_model
    loaded_model = load_model(model_name)
    loaded_model.make_predict_function()
    return loaded_model

start = time.perf_counter()
if model_name == 'Linear Regression':
    from sklearn.linear_model import LinearRegression
    model = model_pipeline(
        X=X, y=y,
        model=LinearRegression(),
        scaling=scaling,
    )
elif model_name == 'Ridge Regression':
    from sklearn.linear_model import Ridge
    alpha = 0
    model = model_pipeline(
        X=X, y=y,
        model=Ridge(alpha),
        scaling=scaling,
    )
elif model_name == 'Lasso Regression':
    from sklearn.linear_model import Lasso
    alpha = 1
    model = model_pipeline(
        X=X, y=y,
        model=Lasso(alpha),
        scaling=scaling,
    )
elif model_name == 'Decision Tree Regression':
    from sklearn.tree import DecisionTreeRegressor
    model = model_pipeline(
        X=X, y=y,
        model=DecisionTreeRegressor(criterion='mse', max_depth=12, min_samples_leaf=1),
        scaling=scaling,
    )
elif model_name == 'XGBoost':
    import xgboost
    model = model_pipeline(
        X=X, y=y,
        model=xgboost.XGBRegressor(),
        scaling=scaling,
    )
elif model_name == 'Support Vector Regression':
    from sklearn import svm
    model = model_pipeline(
        X=X, y=y,
        model=svm.LinearSVR(C=1_000, max_iter=100_000),
        scaling=scaling,
    )
elif model_name == 'Artificial Neural Network':
    model = model_pipeline(
        X=X, y=y,
        model=model_load('ann_model_93.8%.h5'),
        scaling=scaling,
    )

cur_pred = model.predict(test)[0] if model_name != 'Artificial Neural Network' else model.predict(test)[0][0]

#--------------------------------- Main Body ----------------------------------#

# By: `Tam Zher Min`  
# Email: `tamzhermin@gmail.com`

st.write('''
# HDB Resale Prices Predictor

This is a prediction tool built using Streamlit as an extension to the analysis \
done to forecast Singapore's HDB resale prices using classic regression techniques \
such as linear regression, tree-based regression, support vector regression and \
artificial neural networks. The models are trained on historical prices from \
2017 to 2022 retrieved from [Data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices "Historical Resale Flat Prices").

Feel free to use the *Randomize!* button at the sidebar to select the parameters for your HDB! 
'''
)

st.write('# Prediction')

metrics_row = st.columns(2)
diff_pred = 0 if st.session_state["prev_pred"] == 0 else (cur_pred-st.session_state["prev_pred"])/st.session_state["prev_pred"]
with metrics_row[0]:
    st.metric(
        label=f'Resale Price',
        value=f'S${cur_pred:,.2f}',
        delta=f'{diff_pred:.2%}',
    )
cur_runtime = time.perf_counter() - start
diff_runtime = 0 if st.session_state["prev_runtime"] == 0 else -(cur_runtime-st.session_state["prev_runtime"])/st.session_state["prev_runtime"]
with metrics_row[1]:
    st.metric(
        label=f'Runtime',
        value=f'{cur_runtime:.2f}s',
        delta=f'{diff_runtime:.2%}',
    )

st.write('## Previous Results')

prediction_row = [*test.values.tolist()[0], cur_pred, diff_pred, cur_runtime, diff_runtime]
st.session_state['predictions'].append(prediction_row)
st.dataframe(pd.DataFrame(
        st.session_state['predictions'],
        columns=[*[col.replace('_', ' ').title() for col in df.columns], 'Price Diff (%)', 'Runtime (s)', 'Runtime Diff (%)']
    ).style.format({
        'Resale Price': '{:,.2f}',
        'Price Diff (%)': '{:,.2%}',
        'Runtime (s)': '{:.2f}',
        'Runtime Diff (%)': '{:.2%}',
    })
)

st.write('---')

with st.expander(f'View Raw Data'):
    st.write(df)

#---------------------- Update Previous State Variables -----------------------#

st.session_state['prev_pred'] = cur_pred
st.session_state['prev_runtime'] = cur_runtime