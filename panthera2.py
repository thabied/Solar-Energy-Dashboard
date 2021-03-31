import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# INITIATE DASH APP
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server

# CREATE AND EDIT DATAFRAMES
df = pd.read_csv('df.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)

def add_season(row):
  '''add the season in Belgium based on the month'''
  
  if row['month'] == 12 or row['month'] == 1 or row['month'] == 2:
    season = 'Winter'
  elif row['month'] == 3 or row['month'] == 4 or row['month'] == 5:
    season = 'Spring'
  elif row['month'] == 6 or row['month'] == 7 or row['month'] == 8:
    season = 'Summer'
  elif row['month'] == 9 or row['month'] == 10 or row['month'] == 11:
    season = 'Autumn'

  return season

df['Season'] = df.apply(add_season, axis = 1)
df['datetime'] = pd.to_datetime((df.year*10000+df.month*100+df.day).apply(str),format='%Y%m%d')

def add_daily(row):
  '''ADD PART OF THE DAY AS A LABEL BASED ON TIME'''
  
  if row['hour'] >= 0 and row['hour'] < 12:
    label = 'Morning'
  elif row['hour'] >= 12 and row['hour'] <= 17:
    label = 'Afternoon'
  elif row['hour'] > 17 and row['hour'] <= 23:
    label = 'Night'

  return label

weather = pd.read_csv('weathernew.csv')
weather = weather[['clock','visibility','temp','wind','weather']]
weather['hour'] = weather['clock'].astype(str).str[0:2]
weather['hour'] = weather['hour'].astype(float)
weather['label'] = weather.apply(add_daily, axis = 1)

# CREATE BUUBLE CHART ANIMATION
fig = px.scatter(weather, x="wind", y="visibility", animation_frame="hour",animation_group='weather',
           size=abs(weather["temp"]), color="label", title='Visibility (km) and Wind(km/h) over a 24h period')
fig.update_layout(paper_bgcolor='HoneyDew')

# TRAIN RANDOM FOREST MODEL
model = RandomForestRegressor()
model.fit(df[['temp','humidity','visibility']], df['cum_power'])

# CREATE LAYOUT
app.layout = html.Div([

    # HEADING
    dbc.Container(html.H2("A Love Story between a House in Antwerp and the Sun", style={'textAlign':'center',
     'backgroundColor':'lemonchiffon'}), fluid=True),

    # SUB-HEADINGS
    dbc.Container(html.H6("Once upon a time an Engineer gifted his loyal house a Solar Panel. In an act of gratitude, the House provided the engineer and his family with energy to sustain themselves",
     style={'textAlign':'center', 'backgroundColor':'peachpuff'}), fluid=True),

    dbc.Container(html.H6("This is their story",
     style={'textAlign':'center', 'backgroundColor':'peachpuff'}), fluid=True),

    html.Br(),

    # STORY TEXT
    dbc.Container(html.H6('In order to transport the photon messages from the Sun to the Solar Panel, the spirits of the Clouds; Wind and Pressure perform a dance'),
    style={'textAlign':'center', 'backgroundColor':'Pink'}, fluid=True),

    dbc.Container(html.H6('Click Play to see them dance'),
    style={'textAlign':'center', 'backgroundColor':'Pink'}, fluid=True),

    html.Br(),

    # BUBBLE CHART ANIMATION
    dbc.Container(dbc.Col(dcc.Graph(id='play', figure=fig, style={'height':420}), width=800), fluid=True),

    html.Br(),

    # STORY TEXT
    dbc.Container(html.H6('Throughout the seasons their relationship grew stronger, with the aid of the Temperature and Humidity spirits, and the Engineer grew ever grateful '),
    style={'textAlign':'center', 'backgroundColor':'Pink'}, fluid=True),

    dbc.Container(html.H6('Take a look through some of their most cherished moments over the years'),
    style={'textAlign':'center', 'backgroundColor':'Pink'}, fluid=True),

    html.Br(),

    # INPUT OPTIONS
    dbc.Container(dbc.Row(dbc.Col([dcc.RadioItems(id='input',
    options=[
        {'label': '2012', 'value': 2012},
        {'label': '2013', 'value': 2013},
        {'label': '2014', 'value': 2014},
        {'label': '2015', 'value': 2015},
        {'label': '2016', 'value': 2016},
        {'label': '2017', 'value': 2017},
        {'label': '2018', 'value': 2018},
        {'label': '2019', 'value': 2019}],
    value=2012,
    labelStyle={'textAlign':'center'}),
    ], width={"size": 10, "offset": 4})), fluid=True),

    html.Br(),

    # TIME-SERIES AND BAR GRAPH
    dbc.Row([
        dbc.Col(dcc.Graph(id='time',style={'height': 350}), width=450),
        dbc.Col(dcc.Graph(id='seasons',style={'height': 350}), width=450),
    ], justify='center'),

    # 3D SCATTER PLOT
    dbc.Container(dcc.Graph(id='3d', style={'height': 550}), fluid=True),

    html.Br(),

    # STORY TEXT
    dbc.Container(html.H6('After some calculations, the Engineer and some RandomForests developed a formula to predict the number of messages the Sun will send the following day'),
    style={'textAlign':'center', 'backgroundColor':'Thistle'}, fluid=True),

    dbc.Container(html.H6('The formula involves input from the 3 main spirits. Based on different inputs from all 3, see how many messages the Sun will send tomorrow'),
    style={'textAlign':'center', 'backgroundColor':'Thistle'}, fluid=True),

    html.Br(),

    # PREDICTION SLIDERS
    html.H4(children=['Temperature']),

    dcc.Slider(
        id='X1_slider',
        min=-7,
        max=23,
        step=1,
        value=13,
        marks={-7:'-7 °C',8: '8 °C',23:'23 °C'},
        ),

    html.H4(children=['Humidity']),

    dcc.Slider(
        id='X2_slider',
        min=0,
        max=100,
        step=1,
        value=60,
        marks={0:'0 %',50:'50 %', 100:'100 %'}
    ),

    html.H4(children=['Visibility']),

    dcc.Slider(
        id='X3_slider',
        min=0,
        max=20,
        step=1,
        value=15,
        marks={0:'0 km',10:'10 km',20:'20 km'}
    ),
    
    # PREDICTION
    html.H2(id="prediction_result",style={'textAlign':'center'}),

    html.Br(),

    # STORY TEXT
    dbc.Container(html.H6('Word had spread across the globe of this love story and more people started buying this wonderful gift for their trusty houses',
     style={'textAlign':'center', 'backgroundColor':'peachpuff'}), fluid=True),

    dbc.Container(html.H6('Please give thanks to Frank (https://www.kaggle.com/fvcoppen) for writing this story',
     style={'textAlign':'center', 'backgroundColor':'peachpuff'}), fluid=True),

    dbc.Container(html.H2('The End', style={'textAlign':'center',
     'backgroundColor':'lemonchiffon'}), fluid=True),


])

# CREATE GRAPH CALLBACKS
@app.callback(
    Output(component_id='time', component_property='figure'),
    Output(component_id='seasons', component_property='figure'),
    Output(component_id='3d', component_property='figure'),
    Input(component_id='input', component_property='value')
)

# CREATE GRAPH UPDATE FUNCTION
def update_graphs(year):

    data = df.copy()
    data = data[data['year'] == year]

    # CREATE TIME-SERIES PLOT
    time_series = px.line(data, x="datetime", y='Elec_kW', title='Solar Power and Electricity Usage (kWh) over a Year')
    time_series.add_scatter(x=data['datetime'], y=data['cum_power'],mode='lines', name='Solar Production')
    time_series.update_layout(paper_bgcolor='HoneyDew')

    # CREATE BAR GRAPH
    monthseason = data.groupby('Season').mean().reset_index()[['Season','cum_power']]
    seasons_bar = px.bar(monthseason, x='cum_power', y='Season',orientation='h', title='Solar Power (kWh) per Season')
    seasons_bar.update_layout(paper_bgcolor='HoneyDew')

    # CREATE 3D SCATTER PLOT
    threed = px.scatter_3d(data, x="temp", y="humidity", z="cum_power", color='Season',size_max=0.03, opacity=0.7, title='Solar Power (kWh) vs Humidity (%) vs Temperature (C)')
    threed.update_layout(paper_bgcolor='HoneyDew')

    return time_series, seasons_bar, threed

# CREATE MODEL CALLBACKS
@app.callback(
  Output(component_id="prediction_result",component_property="children"),
  [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value")])

# CREATE MODEL UPDATE FUNCTION
def update_prediction(X1, X2, X3):
  '''update prediction value based on slider inputs'''

  input_X = np.array([X1,X2,X3]).reshape(1,-1)
  
  prediction = model.predict(input_X)[0]
  
  return "Prediction: {} kWh".format(round(prediction,1))

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)