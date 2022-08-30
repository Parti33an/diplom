"""
This app uses NavbarSimple to navigate between three different pages.

dcc.Location is used to track the current location. A callback uses the current
location to render the appropriate page content. The active prop of each
NavLink is set automatically according to the current pathname. To use this
feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State, MATCH, ALL

import plotly.graph_objs as go
from plotly.subplots import make_subplots

#import  tinkoff.invest as ti
from tinkoff.invest.constants import INVEST_GRPC_API
import tinkoff_report as tr
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#pd.set_option("display.max_rows", None)
#from pandas import DataFrame
TINKOFF_REF = 'https://www.tinkoff.ru/invest/'
TINKOFF_LOGO = 'https://www.cdn-tinkoff.ru/frontend-libraries/pfcore-static-assets/logos/main-logo.svg'


HOME = ("Токен", "/")
PORTFOLIO = ("Состояние портфеля", "/portfolio")
ANALYZE = ("Исторический анализ", "/analyze")
PREDICT = ("Прогноз", "/predict")

def str_to_dt(date):
    date=date[:10]
    a = [int(i) for i in date.split('-')]
    return datetime(*a)

def initdata():
    return {'TOKEN': '', 'flag_Token': False, 'accounts': list(), 'chosen': None}


DATA = initdata()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

modal = html.Div(
    [
         dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Ошибка!!!"), close_button=False
                ),
                dbc.ModalBody(
                    "Токен не введен или недействителен!!! Попробуйте ещё раз..."
                ),
                dbc.ModalFooter(dbc.Button("Закрыть", id="close-dismiss")),
            ],
            id="modal-dismiss",
            keyboard=False,
            backdrop="static",
        ),
    ],
)

page_layout = html.Div(
    [
        dcc.Location(id="url"),
        dbc.NavbarSimple(
            children=[
                dbc.NavLink(HOME[0], href=HOME[1], active="exact"),
                dbc.NavLink(PORTFOLIO[0], href=PORTFOLIO[1], active="exact"),
                dbc.NavLink(ANALYZE[0], href=ANALYZE[1], active="exact"),
                dbc.NavLink(PREDICT[0], href=PREDICT[1], active="exact"),

            ],
            brand="Анализируй это!!!",
            color="primary",
            dark=True,
        ),
        dbc.Container(html.Div(id="home-page"), className="pt-2"),
        dbc.Container(html.Div(id="portfolio-page"), className="pt-2"),
        html.Div(id="analyze-page"),
        html.Div(id="predict-page"),
        modal
    ]
)

home_page = [dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.CardBody(html.H5("Введите ваш токен", className="card-text")),
                        width=2,
                        ),
                    dbc.Col(
                    html.A( html.Img(src=TINKOFF_LOGO),
                            href=TINKOFF_REF,
                        )
                        ),         
                    dbc.Col(
                        dbc.Input(type='text', id = 'token-input'),
                        width= 7,
                    ),
                    dbc.Col(
                        dbc.Button("Отправить", id="token-button", className="me-2", n_clicks=0),
                        width=1,
                        ),
                ],
                 className="g-0 d-flex align-items-center",
            )
        ],
        ),
        html.Div(id='token-output')]
        
def graph_portfolio_value(count, to_ = datetime.now()):
    
    fig = go.Figure()
        
    if count:
        period = count.candles[list(count.candles.keys())[0]] #дни берем по датам торговли долларом
        x = period.date[(period['date']>=np.datetime64(count.opened_date)) & (period['date']< np.datetime64(to_))]
        value = [count.get_portfolio_by_date(to_ = i)['Стоимость, руб'].sum() for i in x]
        deposit = [count.get_money(to_ = i)['payment'].sum() for i in x]
        fig.add_trace(go.Bar(x = x, y = deposit, name = 'Депозит'))
        fig.add_trace(go.Bar(x = x, y = value, name = 'Величина портфеля'))
        fig.update_xaxes(
                #tickangle = 90,
                title_text = "Дата",
                title_font = {"size": 16},
                title_standoff = 25)

        fig.update_yaxes(
                title_text = "Сумма, рублей",
                title_standoff = 15,
                title_font = {"size": 16},)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01),
            height=700,
            )
    return fig

def graph_profit_instruments(count, currency, to_ = datetime.now()):
    fig = go.Figure()
    if count:
        instruments = []
        for i in count.trading_figis:
            if count.instruments.get(i):
                if count.instruments.get(i)['currency'] == currency:
                    instruments.append(i)
        x = [count.instruments.get(i)['name'] for i in instruments]
        profit = [count.get_profit(ins, to_) for ins in instruments]
        df = pd.DataFrame({'x': x, 'y': profit})
        df["color"] = np.where(df['y']<0, 'red', 'green')
        fig.add_trace(go.Bar(x = df['x'],
                             y = df['y'], name = 'Прибыль',
                             marker_color = df['color']))
        fig.update_xaxes(
                tickangle = 90,
                title_text = "Инструменты",
                title_font = {"size": 16},
                title_standoff = 0)

        fig.update_yaxes(
                nticks=10,
                title_text = "Сумма в валюте инструмента",
                title_standoff = 0,
                title_font = {"size": 16},)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,),
            height=700,
            )
    return fig

def graph_activity_instrument(count, figi, to_ = datetime.now()):
    fig = go.Figure()
    if count:
        #print(to_, 'figi=', figi)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        data = count.candles[figi] #дни берем по датам торговли долларом
        price = data[(data['date']>=np.datetime64(count.opened_date)) & (data['date']< np.datetime64(to_))]
        fig.add_trace(
            go.Scatter(x=price['date'], y=price['close'], name="Цена инструмента", ), 
            secondary_y=False)
       
        mask = np.where((count.data['figi']==figi) & (count.data['otype']==22) & (count.data['date']< np.datetime64(to_) ), True, False)
        # plot a scatter chart by specifying the x and y values
        # Use add_trace function to specify secondary_y axes.
        sell = count.data[mask]
        if (sell.shape[0]>0):
            sell['quantity'] = sell.apply(lambda x: x.quantity - x.quantity_rest, axis=1)
            fig.add_trace(
            go.Scatter(x=sell['date'], y=sell['price'], name="Цена продажи", marker_symbol=34,
                                marker_line_color="green", marker_color="green",
                                marker_line_width=2, marker_size=12,            
            mode="markers"), 
            secondary_y=False)

            # Use add_trace function and specify secondary_y axes = True.
            fig.add_trace(go.Scatter(x = sell['date'],
                                    y = sell['quantity'], name = 'Объем продажи',
                                    marker_line_color="green", marker_color="green", marker_size=10,
                                    mode="markers"), 
                                    secondary_y=True)

        mask = np.where((count.data['figi']==figi) & (count.data['otype']==15) & (count.data['date']< np.datetime64(to_)), True, False)
        buy = count.data[mask]
        #print(buy)
        if (buy.shape[0]>0):
            buy['quantity'] = buy.apply(lambda x: x.quantity - x.quantity_rest, axis=1)
            fig.add_trace(
            go.Scatter(x=buy['date'], y=buy['price'], name="Цена покупки", 
                    marker_symbol=34,
                                marker_line_color="red",
                                marker_line_width=2, marker_size=12,
            mode="markers"), 
            secondary_y=False)

            fig.add_trace(go.Scatter(x = buy['date'],
                                    y = buy['quantity'], name = 'Объем покупки',              
                                    marker_line_color="red", marker_color="red", marker_size=10,
                                        mode="markers"), 
                                        secondary_y=True)
        # Adding title text to the figure
        fig.update_layout(
             height=700,
            legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,),
            )
        
        fig.update_xaxes(title_text="Дата")
        fig.update_yaxes(title_text="Стоимость акций, Д1, закрытие", secondary_y=False)
        fig.update_yaxes(title_text="Объем сделки", secondary_y=True)
    return fig

def graph_balance(count, to_ = datetime.now()):
    fig = go.Figure()
    portfolio = count.get_portfolio_by_date(to_)
    mask = np.where((portfolio['Количество']>0), True,False)
    data=portfolio[mask]
    data['value']=data.apply(lambda x: x['Количество'] * x['Цена, руб'], axis=1)
    labels = data['Наименование']
    values = data['value']
    fig.add_trace(go.Pie(labels=labels, values=values, name="Баланс активов"))
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.8, hoverinfo="label+percent+name")
    fig.update_layout(
    title_text=f"Баланс активов на {to_}",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text=f'{round(data.value.sum())} рублей', x=0.5, y=0.5, font_size=16, showarrow=False),],
    height=700,
    )
    return fig

def report_analyze():
    global DATA
    #account = data['chosen']
    #if account:
    
    #else:
    #    return [html.Br(), html.H1("Ещё нет данных! Введите токен!!!", style = {'text-align': 'center'})]
    #print
    #if DATA['chosen']:
    analyze_page = [
            dbc.Container(
                    dbc.Row(
                    [   dbc.Col(width = 2),
                        dbc.Col(html.H5('Выберите дату окончания периода:')),
                        dbc.Col(
                                html.Div(
                                    dcc.DatePickerSingle(
                                    id='date-picker',
                                    min_date_allowed=datetime(1995, 8, 5),
                                    max_date_allowed=datetime.now(),
                                    display_format = "DD/MM/YYYY",
                                    first_day_of_week = 1,
                                    date=datetime.now()
                                    ), 
                                    ),
                        ), 
                    ], className="g-0 d-flex align-items-center",
                    )
            ),
            html.H4(id='output'),
            dbc.Container(
                html.Div(children=[
                html.H4('Величина портфеля'),
                html.Label('для подробной информации нажмите на график'),
                dcc.Graph(id='portfolio_value'),
            ]), className="pt-2", fluid = True),

            dbc.Container(
            [
            html.Div([
            html.Div(children=[
                html.Br(),
                html.H4('Доходность по инструментам'),
                 dbc.Row(
                    [
                    dbc.Col(html.H6('Валюта инструмента')),
                    dbc.Col(dbc.Select(id="selectCurrency", 
                            options=[{"label": 'Рубль', "value": 'rub' },
                            {"label": 'Доллар США', "value": 'usd' },
                            {"label": 'Евро', "value": 'eur' }
                            ],
                            value = 'rub')),
                    dbc.Col(width = 6), #
                    ], className="g-0 d-flex align-items-center",
                    ),
                dcc.Graph(id='profit_instruments'),

                html.Br(),
                html.H4('Активность по инструменту'),
                html.Div(id = 'instrument_choice'),
                dcc.Graph(id='activity_instrument'),
                ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Br(),
                dcc.Graph(id='balance'),

                html.Br(),
                #html.Label('Состав портфеля'),
                html.Div(id = 'portfolio_table'),

                html.Br(),
                #html.Label('Разное'),
                html.Div(id = 'another'),
            ], style={'padding': 10, 'flex': 1})
            ], style={'display': 'flex', 'flex-direction': 'row'})
        ], className="pt-2", fluid = True)
        ]
    #else:
    #    analyze_page =  [html.Br(), html.H1("Нет доступных данных! Введите токен!!!", style = {'text-align': 'center'})]
    return analyze_page

def report_portfolio(data, c = 0):
    accounts = data['accounts']
    if len(accounts)>0:
        return [ 
        dbc.Row(
            [
                dbc.Col(html.H5('Выберите портфель из доступных по токену:  ')),
                dbc.Col(dbc.Select(id="selectAccount", options=[{"label": account.name, "value": f'{i}'} for i, account in enumerate(accounts) ], value = str(c))), #
            ], className="g-0 d-flex align-items-center",
            ),
            html.Div(id="report"),
            #html.Div(id='choice'),
    ]
    else:
        return [html.Br(), html.H1("Нет доступных данных! Введите токен!!!", style = {'text-align': 'center'})]

app.layout = page_layout
'''app.validation_layout = html.Div([
    homepage_layout,
    portfolio_header
    ])
'''
@app.callback( Output('portfolio_table', 'children'),
               Output('balance', 'figure') ,
               [Input("portfolio_value", "clickData"),
               ])
def portfolio_value_click(data):
    global DATA
    count = DATA['chosen']
    if data:
        day=str_to_dt(data['points'][0]['x'])
        portfolio_table = [
                        html.H6(f'Состав портфеля на {day}'),
                        dbc.Table.from_dataframe(count.get_portfolio_by_date(day)),
                        html.H6('Стоимость портфеля: {} рублей'.format(round(count.get_portfolio_by_date(day)['Стоимость, руб'].sum())))
                        ]
        return portfolio_table, graph_balance(count, day)
    return [], go.Figure()
    

@app.callback(  Output('portfolio_value', 'figure') , 
                Output('profit_instruments', 'figure') ,
                [Input("date-picker", "date"),
                 Input("selectCurrency", "value"),
                ])
def analyze_page_content(day, currency):
    global DATA
    portfolio_table = []
    count = DATA['chosen']
    return graph_portfolio_value(count, str_to_dt(day)), graph_profit_instruments(count, currency, str_to_dt(day)) 

@app.callback(  Output('instrument_choice', 'children'), 
                [Input("date-picker", "date")])
def select_instrument(day):
    global DATA
    instruments=[]
    choice=html.Div()
    if DATA['chosen']:
        instruments = []
        for i in DATA['chosen'].trading_figis:
            if DATA['chosen'].instruments.get(i):
                instruments.append(i)
        choice = dbc.Select(id="selectInstrument",
                options=[{"label": DATA['chosen'].instruments[ins]['name'], "value": ins} for ins in instruments], value = instruments[0]) #
    return choice

@app.callback(  Output('activity_instrument', 'figure'), 
                [Input("selectInstrument", "value"),
                Input("date-picker", "date")])
def activity_instrument(figi, day):
    global DATA
    return graph_activity_instrument(DATA['chosen'], figi, str_to_dt(day))
    
@app.callback(
                Output("home-page", "children") , 
                Output("portfolio-page", "children"),
                Output("analyze-page", "children"),
                Output("predict-page", "children"),
             [Input("url", "pathname")])
def render_page_content(pathname):
    global DATA
    #print("On link ---> DATA['chosen'] = ", DATA)

    if pathname == HOME[1]:
        return home_page, None, None, None
    elif pathname == PORTFOLIO[1]:
        return None, report_portfolio(DATA) , None, None                              #
    elif pathname == ANALYZE[1]:
        return None, None, report_analyze(), None
    elif pathname == PREDICT[1]:
        return None, None, None, html.P(f"Oh cool, this is page {PREDICT[0]}!")
    return html.Div(
        [
            html.H1("404: Не найден", className="text-danger"),
            html.Hr(),
            html.P(f"Ссылка {pathname} не распознана..."),
        ],
        className="p-3 bg-light rounded-3",
        ), None, None, None

@app.callback(Output('token-output', 'children'),
              Output("modal-dismiss", "is_open"),          
              Input('token-button', 'n_clicks'),
              State('token-input', 'value'),
              Input("close-dismiss", "n_clicks"),
              State("modal-dismiss", "is_open"),          
              )
def home_page_output(n_open, input, n_close, is_open):
    global DATA
    output = [html.Br(), html.H4("Нет данных...")]
    modal = False
    
    if input and n_open:
        responce = tr.get_accounts(input)
        if not isinstance(responce, str):
            DATA = initdata()
            DATA['accounts'] = responce
            DATA['TOKEN'] = input
            DATA['flag_Token'] = True
        else:
            print(str(responce))
            DATA['flag_Token'] = False    
    else:
        DATA['flag_Token'] = False

    if (n_open and not DATA['flag_Token']):
        modal = True
    if n_close and is_open:
        modal = not is_open

    #print(DATA['accounts'], '  длина:', len(DATA['accounts']))
    if len(DATA['accounts'])>0:
        output = [html.Br(), html.H5('По токену {} доступны:'.format(DATA['TOKEN']))] 
        for count in DATA['accounts']:
            output.append(html.H6(f'Портфель {count.name}, дата открытия счета: {count.opened_date} '))
            #output.append(html.H5(f'Для более детальной информации перейдите по } '))
       # [html.P(f'Портфель {account.name}, дата открытия счета: {account.opened_date} ') for account in DATA['accounts']]] - прибалвение списка в выдачу дает ошибку почему то?!
    return output, modal 

@app.callback(Output("report", 'children'),
             Input("selectAccount", "value")
             )
def portfolio_page_output(choice):
    global DATA
    DATA['chosen'] = DATA['accounts'][int(choice)]
    return  [dbc.Card(html.Div(
            [html.Br(),
            dbc.Row(
            [dbc.Col(width=1), dbc.Col(html.H5(f"Портфель {account.name}", style ={'font-style': 'italic'}))]),
            dbc.Table.from_dataframe(account.get_portfolio_by_date() , striped=True, bordered=True, hover=True, color = 'info' if i == int(choice) else 'light'),
            html.H6(f'Внесено денег: {account.get_money().payment.sum()}'),
            html.H6(f'Комиссия брокера за операции: {account.get_comissions_rub().payment.sum()}'),
            html.H6(f'Плата за перенос позиций: {account.get_margin_fee().payment.sum()}'),
            html.H6(f'Налога заплачено: {account.get_taxes().payment.sum()}'),
            html.H6(f'Доход по фьючерсам: {account.get_varmargin().payment.sum()}'),
            html.H6(f'Дивидендный доход: {account.get_dividends().payment.sum()}'),
            html.H5('Текущая прибыль: {} рублей'.format(round(account.get_portfolio_by_date()['Стоимость, руб'].sum()-account.get_money().payment.sum(),2))),
            html.Br(),
            ]
            )) for i, account in enumerate(DATA['accounts'])] #, f"Выбран портфель {account.name}"
    

if __name__ == "__main__":
    app.run_server(port=9000)
