import  tinkoff.invest as ti
from tinkoff.invest.constants import INVEST_GRPC_API

from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np
import pandas as pd
#pd.set_option("display.max_rows", None)
from pandas import DataFrame
#import warnings
#warnings.filterwarnings('ignore')

from tinkoff.invest.services import InstrumentsService, MarketDataService, InstrumentIdType
from tinkoff.invest.exceptions import RequestError

RUB = 'rub'

from tinkoff.invest import (
    AccessLevel,
    AccountStatus,
    CandleInstrument,
    Client,
    AsyncClient,
    MarketDataRequest,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
    GenerateBrokerReportRequest,
    GetBrokerReportRequest,
    OperationsResponse,
    Operation,
    OperationType,
    OperationState,
    InstrumentStatus
)

TAXES = ( 
        OperationType.OPERATION_TYPE_BOND_TAX, 
        OperationType.OPERATION_TYPE_TAX, 
        OperationType.OPERATION_TYPE_DIVIDEND_TAX,
        OperationType.OPERATION_TYPE_TAX_CORRECTION,
        OperationType.OPERATION_TYPE_BENEFIT_TAX,
        OperationType.OPERATION_TYPE_TAX_PROGRESSIVE,
        OperationType.OPERATION_TYPE_BOND_TAX_PROGRESSIVE,
        OperationType.OPERATION_TYPE_DIVIDEND_TAX_PROGRESSIVE,
        OperationType.OPERATION_TYPE_BENEFIT_TAX_PROGRESSIVE,
        OperationType.OPERATION_TYPE_TAX_CORRECTION_PROGRESSIVE,
        OperationType.OPERATION_TYPE_TAX_REPO_PROGRESSIVE, 
        OperationType.OPERATION_TYPE_TAX_REPO,
        OperationType.OPERATION_TYPE_TAX_REPO_HOLD,
        OperationType.OPERATION_TYPE_TAX_REPO_REFUND,
        OperationType.OPERATION_TYPE_TAX_REPO_HOLD_PROGRESSIVE,
        OperationType.OPERATION_TYPE_TAX_REPO_REFUND_PROGRESSIVE,
        OperationType.OPERATION_TYPE_TAX_CORRECTION_COUPON)

def datetime_pd(from_: datetime = None, to_: datetime = datetime.now()):
    return np.datetime64(from_), np.datetime64(to_), 
    
class Account:
    instruments = dict()
    currencies = dict()
    init = False
    
    @classmethod
    def _get_instruments(cls, token):
        try:
            with Client(token) as client:
                shares = client.instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL).instruments
                for i in shares: 
                    cls.instruments[i.figi]= {
                            'currency': i.currency,
                            'ticker':i.ticker,
                            'name' : i.name,
                            'country': i.country_of_risk_name,
                            'sector': i.sector,    
                            'instrument_type': 'share'    
                            }
                bonds = client.instruments.bonds(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL).instruments
                for i in bonds: 
                    cls.instruments[i.figi]={
                            'currency': i.currency,
                            'ticker':i.ticker,
                            'name' : i.name,
                            'country': i.country_of_risk_name,
                            'sector': i.sector,    
                            'instrument_type': 'bond',     
                            'nominal': cls._cast_money(i.nominal)
                            }
                etfs = client.instruments.etfs(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL).instruments
                for i in etfs: 
                    cls.instruments[i.figi]={
                            'currency': i.currency,
                            'ticker':i.ticker,
                            'name' : i.name,
                            'country': i.country_of_risk_name,
                            'sector': i.sector,    
                            'instrument_type': 'etf'    
                            }
                futures = client.instruments.futures(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL).instruments
                for i in futures: 
                    cls.instruments[i.figi]={
                            'currency': i.currency,
                            'ticker':i.ticker,
                            'name' : i.name,
                            'country': i.country_of_risk_name,
                            'sector': i.sector,    
                            'instrument_type': 'future'    
                            }
                    # 'BBG0013HGFT4': {'currency': 'rub', 'ticker': 'USD000UTSTOM', 'name': 'Доллар США', 'country': '', 'instrument_type': 'currency'}
                    #'BBG0013HJJ31': {'currency': 'rub', 'ticker': 'EUR_RUB__TOM', 'name': 'Евро', 'country': '', 'instrument_type': 'currency'}
                    #currencies = client.instruments.currencies().instruments #
                    #for i in currencies: 
                cls.currencies['BBG0013HGFT4']={
                            'currency': RUB,
                            'ticker':'USD000UTSTOM',
                            'name' : 'Доллар США',
                            'country': '',
                            'instrument_type': 'currency'    
                            }
                cls.currencies['BBG0013HJJ31']={
                            'currency': RUB,
                            'ticker':'EUR_RUB__TOM',
                            'name' : 'Евро',
                            'country': '',
                            'instrument_type': 'currency'    
                            }
                cls.init = True
        except RequestError as e:
            return str(e)
        
            
    def __init__(self, TOKEN, account):
        self.token = TOKEN
        #self.client = client
        self.usdrur = None
        self.account_id = account.id
        self.name = account.name
        self.candles=dict()
        self.rub = 0
        self.opened_date = account.opened_date
        self.closed_date = account.closed_date
        self.status = account.status
        if(not self.init):
            Account._get_instruments(self.token)
        self.data = self._get_operations_df()
        self.trading_figis = list(self.data.figi.unique())
        self.trading_figis.remove('')
        for figi in self.trading_figis:
            self.candles[figi] = self.get_candles(figi, from_ = self.opened_date, to_ = datetime.now())

    def get_candles(self, figi, from_ = datetime(2015,1,1) , to_ = datetime.now()):
        start_year = from_.year
        cur_year = to_.year
        data = []
        try:
            with Client(self.token) as client:
                for i in range(start_year, cur_year+1):
                    u = client.market_data.get_candles(figi=figi, from_ = datetime(i,1,1), to = datetime(i,12,31), interval =5 )
                    if len(u.candles) == 0:
                        continue
                    for candle in u.candles:
                        if candle.is_complete == True:
                            record = self.candle_todict(candle)
                            data.append(record)
                df = pd.DataFrame(data) 
                df["date"]=pd.to_datetime(df.date).dt.tz_localize(None)
                return df
        except RequestError as e:
            return str(e)
                        

    def candle_todict(self, c):
        r = {
            'date': c.time,
            'close': self._cast_money(c.close),
        }
        return r

    def _operation_todict(self, o : Operation):
        """
        Преобразую PortfolioPosition в dict
        :param p:
        :return:
        """
        if(o.figi!=''):
            
            if self.currencies.get(o.figi):
                ins = self.currencies
            elif self.instruments.get(o.figi):
                ins = self.instruments
            else:
                print(o.figi)
            ticker = ins[o.figi]['ticker']
            name = ins[o.figi]['name']
            country = ins[o.figi]['country']
        else:
            ticker = 'RUB'
            name = 'рубль'
            country = 'Россия'
            
        r = {
            'date': o.date,
            'type': o.type,
            'otype': o.operation_type,
            'currency': o.currency,
            'instrument_type': o.instrument_type,
            'figi': o.figi,
            'ticker': ticker,
            'name' : name,
            'quantity': o.quantity,
            'quantity_rest': o.quantity_rest,
            'state': o.state,
            'payment': self._cast_money(o.payment, False),
            'price': self._cast_money(o.price, False),
            'country': country,
        }
        return r
    
    @classmethod
    def _cast_money(cls, v, to_rub=True):
        """
        https://tinkoff.github.io/investAPI/faq_custom_types/
        :param to_rub:
        :param v:
        :return:
        """
        r = v.units + v.nano / 1e9
        #if to_rub and hasattr(v, 'currency') and getattr(v, 'currency') == 'usd':
        #    r *= self.get_usdrur()
        return r

    def _get_operations_df(self) -> Optional[DataFrame]:
        """
        Преобразую PortfolioResponse в pandas.DataFrame
        :param account_id:
        :return:
        """
        data=[]
        try:
            with Client(self.token) as client:
                r: OperationsResponse = client.operations.get_operations(
                    account_id=self.account_id,
                    from_= self.opened_date,
                    to  = datetime.now()
                )
                
                if len(r.operations) < 1: return None
                
                for p in r.operations:
                    #print(p)
                    ins=None
                    if p.state == OperationState.OPERATION_STATE_EXECUTED:
                        record = self._operation_todict(p)
                        data.append(record)
                        
                df = pd.DataFrame(data)
                # https://www.datasciencelearner.com/numpy-datetime64-to-datetime-implementation/
                df["date"]=pd.to_datetime(df.date).dt.tz_localize(None)
                #df["date"]=pd.to_datetime(df.date, unit="ns", utc=False)
                return df
        except RequestError as e:
            return str(e)
    
    def get_money(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        
        #https://dev-gang.ru/article/sravnenie-daty-i-vremeni-v-pythons-czasovymi-pojasami-i-bez-nih-wkbsv8ew17/
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) 
                         & ((self.data['otype']==OperationType.OPERATION_TYPE_INPUT) | (self.data['otype']==OperationType.OPERATION_TYPE_OUTPUT)), True, False)
        return self.data[mask][['date','payment']]
    
    def get_comissions_rub(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) &
                         ((self.data['otype']==OperationType.OPERATION_TYPE_BROKER_FEE) |
                         (self.data['otype']==OperationType.OPERATION_TYPE_SERVICE_FEE))
                          & (self.data['currency']==RUB), True, False)
        
        return self.data[mask][['date','payment']]

    def get_margin_fee(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) 
                         & (self.data['otype']==OperationType.OPERATION_TYPE_MARGIN_FEE) , True, False)
        
        return self.data[mask][['date','payment']]

    def get_taxes(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) & (self.data['otype'].isin(TAXES)) , True, False)
        return self.data[mask][['date','payment']]

    def get_varmargin(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) 
                         & ((self.data['otype']==OperationType.OPERATION_TYPE_ACCRUING_VARMARGIN) |
                            (self.data['otype']==OperationType.OPERATION_TYPE_WRITING_OFF_VARMARGIN)
                          ), True, False)
        return self.data[mask][['date','payment']]
    
    def get_dividends(self, from_: datetime = None, to_ : datetime = datetime.now() ):
        if (from_ == None):
            from_ = self.opened_date
        from_, to_ = datetime_pd(from_ = from_, to_ = to_)
        mask = np.where((self.data['date'] >= from_) & (self.data['date'] < to_) 
                         & ((self.data['otype']==OperationType.OPERATION_TYPE_DIVIDEND)
                        | (self.data['otype']==OperationType.OPERATION_TYPE_DIV_EXT )), True, False)
        return self.data[mask][['date','payment']]
   
    def instrument_position(self, figi , to_ : datetime = datetime.now() ):
        _, to_ = datetime_pd(to_ = to_)
        mask = np.where((self.data['figi']==figi) & (self.data['date'] < to_) 
                        & (self.data['otype'].isin([OperationType.OPERATION_TYPE_DELIVERY_BUY, OperationType.OPERATION_TYPE_BUY, OperationType.OPERATION_TYPE_BUY_MARGIN])), True, False)
        chosenbuy = self.data[mask]
        buy = chosenbuy['quantity'].sum() - chosenbuy['quantity_rest'].sum() 
        
        mask = np.where((self.data['figi']==figi) & (self.data['date'] < to_) 
                        & (self.data['otype'].isin([OperationType.OPERATION_TYPE_DELIVERY_SELL, OperationType.OPERATION_TYPE_SELL, OperationType.OPERATION_TYPE_SELL_MARGIN])), True, False)
        chosensell = self.data[mask]
        sell = chosensell['quantity'].sum() - chosensell['quantity_rest'].sum() 
        return buy - sell
    
    def currency_position(self, figi , to_ : datetime = datetime.now()): # валютные комиссии сюда же идут
        #to_ = np.datetime64(to_)
        _, to_ = datetime_pd(to_ = to_)
        
        currency = self.currencies[figi]['ticker'][:3].lower()
        mask = np.where((self.data.currency == currency) & (self.data['date'] < to_), True,False)
        return self.instrument_position(figi , to_ = to_ ) + self.data[mask]['payment'].sum() 
    
    def get_price(self, figi , to_ : datetime = datetime.now() ):
        #amount = self.instrument_position(figi, to_ = to_)
        
        if (to_.year == datetime.now().year and to_.month == datetime.now().month and to_.day == datetime.now().day):
            try:
                with Client(self.token) as client:
                    u = client.market_data.get_last_prices(figi=[figi])
                    cur_price = self._cast_money(u.last_prices[0].price)
            except RequestError as e:
                return str(e)
        else:
            _, pdto_ = datetime_pd(to_ = to_)
            data = self.candles[figi]
            price = data[data['date'] < pdto_]
            #cost = amount * price['close'][price.shape[0]-1]
            cur_price = price['close'][price.shape[0]-1]
        
        if self.instruments.get(figi):
            if self.instruments[figi]['instrument_type']=='bond':
                cur_price = self.instruments[figi]['nominal'] * cur_price/100
                
            if self.instruments[figi]['currency']!= RUB:
                for c in self.currencies:
                    if (self.currencies[c]['ticker'][:3].lower() == self.instruments[figi]['currency']):
                        return cur_price * self.get_price(c, to_ = to_)
        return cur_price
    
    def get_profit(self, figi, to_ : datetime = datetime.now() ): #кроме валюты
        if self.instruments.get(figi):
            amount = self.instrument_position(figi, to_ = to_) * self.get_price(figi, to_ = to_)
            _, to_ = datetime_pd(to_ = to_)
            mask = np.where((self.data['figi']==figi) & (self.data['date'] < to_) & (self.data['otype'].isin([OperationType.OPERATION_TYPE_DELIVERY_BUY, OperationType.OPERATION_TYPE_BUY, OperationType.OPERATION_TYPE_BUY_MARGIN])), True, False)
            buy = self.data[mask]['payment'].sum()
            mask = np.where((self.data['figi']==figi) & (self.data['date'] < to_) & (self.data['otype'].isin([OperationType.OPERATION_TYPE_DELIVERY_SELL, OperationType.OPERATION_TYPE_SELL, OperationType.OPERATION_TYPE_SELL_MARGIN])), True, False)
            sell = self.data[mask]['payment'].sum()
            return amount + sell + buy
        else:
            return 0

    #def get_currency(self, ticker):
    
    def get_positions(self, to_ : datetime = datetime.now() ):
        positions = []
        _, to_ = datetime_pd(to_ = to_)
        #shares = self.trading_figis:
        for figi in self.trading_figis:
            if self.instruments.get(figi):
                amount = self.instrument_position(figi , to_ = to_ )
                if amount!=0:
                    r = {
                        'instrument_type': self.instruments[figi] ['instrument_type'],
                        'figi': figi,
                        'ticker':self.instruments[figi] ['ticker'],
                        'currency': self.instruments[figi] ['currency'],
                        'name' : self.instruments[figi] ['name'],
                        'quantity': amount
                        }
                    positions.append(r)

        for figi in self.currencies:
            amount = self.currency_position(figi , to_ = to_ ) # + self.data[(self.data.currency == self.currency['currency'])]['payment'].sum()
            if amount!=0:
                r = {
                    'instrument_type': self.currencies[figi] ['instrument_type'],
                    'figi': figi,
                    'ticker':self.currencies[figi] ['ticker'],
                    'currency': self.currencies[figi] ['currency'],
                    'name' : self.currencies[figi] ['name'],
                    'quantity': amount
                    }
                positions.append(r)
        
        # добавляем рублевую позицию
        mask = np.where(((self.data.otype == OperationType.OPERATION_TYPE_BUY) |
                         (self.data.otype == OperationType.OPERATION_TYPE_SELL)) &
                        (self.data.instrument_type != 'futures') &
                        (self.data.currency == RUB ) & (self.data.date < to_), True, False)
        r = {
            'instrument_type': '', #self.instruments[''] ['instrument_type'],
            'figi': '',
            'ticker':'RUB', #self.instruments[''] ['ticker'],
            'currency':'',
            'name' : 'Рубль', #self.instruments[''] ['name'],
            'quantity': self.get_money(to_ = to_)['payment'].sum() 
                        + self.get_comissions_rub(to_ = to_)['payment'].sum() 
                        + self.get_margin_fee(to_ = to_)['payment'].sum() 
                        + self.get_taxes(to_ = to_)['payment'].sum()
                        + self.get_varmargin(to_ = to_)['payment'].sum()
                        + self.get_dividends(to_ = to_)['payment'].sum()
                        + self.data[mask]['payment'].sum()
            }
        positions.append(r)
        return pd.DataFrame(positions)
    
    def get_portfolio_by_date(self, to_ : datetime = datetime.now() ):
        portfolio = self.get_positions(to_ = to_)
        portfolio['quantity'] = portfolio.apply(lambda x: round(x.quantity, 2) if x.quantity!=round(x.quantity, 0) else x.quantity, axis=1)
        portfolio['price'] = portfolio.apply(lambda x: round(self.get_price(x.figi, to_ = to_), 2) if (x.figi!='') else 1, axis=1)
        portfolio['amount'] = portfolio.apply(lambda x: round(x.quantity * x.price, 2) if x.instrument_type!='future' else 0, axis=1)
        portfolio['instrument_type'] = portfolio.apply(lambda x: 'акция' if x.instrument_type == 'share' else x.instrument_type, axis=1)
        portfolio['instrument_type'] = portfolio.apply(lambda x: 'облигация' if x.instrument_type == 'bond' else x.instrument_type, axis=1)
        portfolio['instrument_type'] = portfolio.apply(lambda x: 'фьючерс' if x.instrument_type == 'future' else x.instrument_type, axis=1)
        portfolio['instrument_type'] = portfolio.apply(lambda x: 'валюта' if x.instrument_type == 'currency' else x.instrument_type, axis=1)
        portfolio.rename(columns={'instrument_type': 'Актив', 'currency': 'Валюта', 'name': 'Наименование', 'quantity': 'Количество', 'price': 'Цена, руб', 'amount': 'Стоимость, руб'}, inplace=True)

        return portfolio
    
def get_accounts(token):
    accounts = []
    try:
        with Client(token) as client:
            counts = client.users.get_accounts().accounts
            for count in counts:
                if not (count.access_level == ti.AccessLevel.ACCOUNT_ACCESS_LEVEL_NO_ACCESS):
                    #print(f"Счет {count.name} доступен")
                    accounts.append(Account(token, count))
            return accounts
    except RequestError as e:
        return str(e)
    