import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_actual_ensemble(name, df, model='ens'):
    """
    Plots the prices as a time-series showing actual/predicted values with daily and
    cumulative prediction errors
    Input:
      name: the name of the stock/crypto
      df: the historical dataframe (output from roll_predictions)
      model: model to plot daily errors for (ens,fbp,lstm,last_price)
    """
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Actual vs Predicted Closing Price', 'Daily Error', 'Cumulative Error'))
    # Actual prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.actual,
        fill='tozeroy',
        mode='lines',
        line=dict(color="#ccc"),
        name='Actual'),
        row=1, col=1
    )
    # predicted prices
    # ensemble
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.ens,
        mode='lines',
        line_color='orange',
        name='Ensemble'),
        row=1, col=1
    )
    # lstm
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.lstm,
        mode='lines',
        line_color='#7cbf5a',
        name='LSTM'),
        row=1, col=1
    )
    # FB prophat
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.fbp,
        mode='lines',
        line_color='skyblue',
        name='FB Prophet'),
        row=1, col=1
    )

    # daily error
    # bar chart
    colors = {'ens': 'orange', 'fbp': 'skyblue', 'lstm': '#7cbf5a', 'last_close': 'crimson'}
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[f'{model}_diff'],
        name=f'{model} Error',
        marker_color=colors[model]
    ), row=2, col=1)

    # cumulative errors
    # ensemble
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{model}_cum'],
        fill='tozeroy',
        line_color='orange',
        mode='lines',
        name='Cumulative Error- Ensemble Model'
    ), row=3, col=1)

    # LSTM
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['lstm_cum'],
        fill='tozeroy',
        line_color='#7cbf5a',
        mode='lines',
        name='Cumulative Error-LSTM Model'
    ), row=3, col=1)

    # FB p
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['fbp_cum'],
        fill='tozeroy',
        line_color='skyblue',
        mode='lines',
        name='Cumulative Error- FB Prophet Model'
    ), row=3, col=1)

    fig.update_layout(height=600,
                      width=800,
                      template='plotly_white',
                      title_text=f"{name}: Actual Vs. Predicted Prices")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.5,
        xanchor="left",
        x=0.2
    ))
    return fig
