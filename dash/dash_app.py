import dash
import dash_bootstrap_components
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
from dash.dependencies import Input, Output
import main
import visuals

# get the main config file that defines the update process
config_main = main.get_config(main.ROOT_PATH + 'config/config_main.json')

# get the list of stocks from the config file. Used in the drop-down selector
stock_lst = [{'label': s['stock']['name'], 'value': s['stock']['name']} for s in config_main]

# start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], prevent_initial_callbacks=False)
app.config['suppress_callback_exceptions'] = True

# LAYOUT ------------------------------------------------------------------------------------------------
app.layout = dash_bootstrap_components.Container([
    html.Div([
        # header row
        html.H4("LSTM/Prophet Ensemble Model for Market Price Predictions"),
        # stock/crypto selection row
        dbc.Row([
            dbc.Col(html.H5('Select Stock/Crypto'), width={'size': 6, 'order': 1}),
            dbc.Col(dcc.Dropdown(
                id='dd_select_stock',
                options=stock_lst,
                value=stock_lst[0]['value'],
                placeholder="Select stock/crypto",
            ), width={'size': 6, 'order': 2})
        ]),
        # Actual/Predicted Values Plot Metrics, Model Configuration Settings
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='plot_predictions'), width={'size': 9, 'order': 1}, align='center'),
            dbc.Col(dcc.Markdown(id='md_model_config'), width={'size': 3, 'order': 2}, align='center')
        ]),
        html.Hr(),
    ])
], fluid=False)


# CALLBACKS--------------------------------------------------------------------------------------------------
@app.callback(
    Output('plot_predictions', 'figure'),
    Input('dd_select_stock', 'value')
)
def update_plot(stock):
    """
    Updates the actual/predicted plot
    :param stock: The stock selected in the dd_select_stock dropdown
    :return: Renders the plot_predictions plot
    """
    for s in config_main:
        if s['stock']['name'] == stock:
            df_hist_path = main.ROOT_PATH + 'data/' + s['model']['df_hist']
            df = pd.read_pickle(df_hist_path)

            return visuals.plot_actual_ensemble(s['stock']['name'], df)


@app.callback(
    Output('md_model_config', 'children'),
    Input('dd_select_stock', 'value')
)
def update_model_config(stock):
    """
    Prints the model configuration details
    :param stock: The stock selected in the dd_select_stock dropdown
    :return: Prints the model configuration details in the md_model_config markdown object
    """
    for s in config_main:
        if s['stock']['name'] == stock:
            stock_type = s['stock']['type']
            name = s['stock']['name']
            transform = s['stock']['transform']
            shift = s['stock']['shift']
            n_steps = s['model']['n_steps']
            n_predict = s['model']['n_predict']
            feature_path = s['model']['features']

            # Get the features used in the final model
            df_features = pd.read_pickle("data/" + feature_path)
            features = [f for f in df_features.columns if f not in ['symbol']]

            config_str = f""" 
            #### Model Config  
            **Name**:{name} | {stock_type}  
            **Number of Input Days**:{n_steps}  
            **Number of Predicted Days**:{n_predict}  
            **Data Transformation**:{transform}  
            **Lagged Features**:{shift}  
            **Input Features**:  
            {features}
            """
            return config_str


app.run_server(debug=True)
