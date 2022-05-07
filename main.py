from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.app import MDApp
import yfinance as yf
from matplotlib import pyplot as plt
from datetime import date
from joblib import load
from sklearn.model_selection import train_test_split
import dataframe_image as dfi

Builder.load_file('main.kv')


class MainScreen(Screen):
    pass


class MDSearchScreen(Screen):
    def get_data(self):
        ticker = self.ids.search_field.text
        stock = yf.Ticker(ticker)
        df = yf.download(ticker, start='2021-09-01')
        company_name = str(stock.get_info()['longName'])
        self.ids["comp_name"].text = company_name
        close = 'Close: ' + str(round(df.iloc[-1]['Close'], 4))
        stock_open = 'Open:  ' + str(round((df.iloc[-1]['Open']), 4))
        high = 'Highest(today):  ' + str(round(df.iloc[-1]['High'], 4))
        low = 'Lowest(today):  ' + str(round(df.iloc[-1]['Low'], 4))
        volume = 'Volume:  ' + str(round(df.iloc[-1]['Volume'], 4))
        self.ids["search_label1"].text = stock_open + '\n\n' + high + '\n\n' + low + '\n\n' + volume + '\n\n' + close
        plt.figure(figsize=(7, 3))
        plt.plot(df['Close'], label='Close', linestyle='dashed', color='red')
        plt.ylabel("Price")
        today = date.today().strftime("%B %d, %Y")
        plt.title(f"Close of {company_name} over time-Generated on {today}")
        plt.legend()
        plt.savefig('./my_plot.png')
        self.ids["plot1"].source = './my_plot.png'


class MDIndex(Screen):
    pass


class TestPred(Screen):
    pass


class TrainPred(Screen):
    pass


class ErrorCalc(Screen):
    pass


def prev_returns(df, prev):
    names = []
    for i in range(1, prev + 1):
        df['Prev_Day' + str(i)] = df['Close'].shift(i)
        names.append('Prev_Day' + str(i))
    return names


def info_str(stock):
    stock = yf.Ticker(stock)
    currency = str(stock.get_info()['currency'])
    timezone = str(stock.get_info()['exchangeTimezoneName'])
    day_avg200 = str(stock.get_info()['twoHundredDayAverage'])
    week_low52 = str(stock.get_info()['fiftyTwoWeekLow'])
    reg_market_price = str(stock.get_info()['regularMarketPrice'])
    return currency, timezone, day_avg200, week_low52, reg_market_price


class PredSP500(Screen):
    def on_enter(self, *args):
        df = yf.download('^GSPC', start='2022-03-02')
        lagnames = prev_returns(df, 2)
        df.dropna(inplace=True)
        train, test = train_test_split(df, shuffle=False,
                                       test_size=0.7, random_state=0)
        model = load('models/lrml.joblib')
        test['Predicted Close'] = model.predict(test[lagnames])
        n_day_prev = test[['Close', 'Predicted Close']].tail(8)
        n_day_prev['Date'] = n_day_prev.index.strftime("%c")
        first_colum = n_day_prev.pop('Date')
        n_day_prev.insert(0, 'Date', first_colum)
        n_day_prev.reset_index(drop=True, inplace=True)
        dfi.export(n_day_prev, 'df_styled.png')
        self.ids['sp_img'].source = './df_styled.png'
        currency, timezone, day_avg200, week_low52, reg_market_price = info_str('^GSPC')
        self.ids['data1'].text = 'Currency: ' + currency
        self.ids['data2'].text = 'Timezone: ' + timezone
        self.ids['data3'].text = '200 Day Avg: ' + day_avg200
        self.ids['data4'].text = '52 Week Low: ' + week_low52
        self.ids['data5'].text = 'Market Price: ' + reg_market_price


class PredDJ(Screen):
    def on_enter(self, *args):
        df = yf.download('^DJI', start='2022-03-02')
        lagnames = prev_returns(df, 2)
        df.dropna(inplace=True)
        train, test = train_test_split(df, shuffle=False,
                                       test_size=0.7, random_state=0)
        model = load('models/lrml_dj.joblib')
        test['Predicted Close'] = model.predict(test[lagnames])
        n_day_prev = test[['Close', 'Predicted Close']].tail(8)
        n_day_prev['Date'] = n_day_prev.index.strftime("%c")
        first_colum = n_day_prev.pop('Date')
        n_day_prev.insert(0, 'Date', first_colum)
        n_day_prev.reset_index(drop=True, inplace=True)
        dfi.export(n_day_prev, 'df_styled1.png')
        self.ids['sp_img1'].source = './df_styled1.png'
        currency, timezone, day_avg200, week_low52, reg_market_price = info_str('^DJI')
        self.ids['data6'].text = 'Currency: ' + currency
        self.ids['data7'].text = 'Timezone: ' + timezone
        self.ids['data8'].text = '200 Day Avg: ' + day_avg200
        self.ids['data9'].text = '52 Week Low: ' + week_low52
        self.ids['data10'].text = 'Market Price: ' + reg_market_price


class PredNSD(Screen):
    def on_enter(self, *args):
        df = yf.download('^IXIC', start='2022-03-02')
        lagnames = prev_returns(df, 2)
        df.dropna(inplace=True)
        train, test = train_test_split(df, shuffle=False,
                                       test_size=0.7, random_state=0)
        model = load('models/lrml_nsd.joblib')
        test['Predicted Close'] = model.predict(test[lagnames])
        n_day_prev = test[['Close', 'Predicted Close']].tail(8)
        n_day_prev['Date'] = n_day_prev.index.strftime("%c")
        first_colum = n_day_prev.pop('Date')
        n_day_prev.insert(0, 'Date', first_colum)
        n_day_prev.reset_index(drop=True, inplace=True)
        dfi.export(n_day_prev, 'df_styled2.png')
        self.ids['sp_img2'].source = './df_styled2.png'
        currency, timezone, day_avg200, week_low52, reg_market_price = info_str('^IXIC')
        self.ids['data11'].text = 'Currency: ' + currency
        self.ids['data12'].text = 'Timezone: ' + timezone
        self.ids['data13'].text = '200 Day Avg: ' + day_avg200
        self.ids['data14'].text = '52 Week Low: ' + week_low52
        self.ids['data15'].text = 'Market Price: ' + reg_market_price


class StockApp(MDApp):
    from kivy.core.window import Window
    Window.size = (470, 900)

    def build(self):
        self.title = 'Plutus - Stock Market Forecast'
        self.theme_cls.theme_style = "Dark"
        sm = ScreenManager()
        sm.add_widget(MainScreen())
        sm.add_widget(MDSearchScreen(name='Search'))
        sm.add_widget(MDIndex(name='Predict1'))
        sm.add_widget(MainScreen(name='Main'))
        sm.add_widget(TestPred(name='TestPred'))
        sm.add_widget(TrainPred(name='TrainPred'))
        sm.add_widget(ErrorCalc(name='ErrorCalc'))
        sm.add_widget(PredSP500(name='PredSP500'))
        sm.add_widget(PredDJ(name='PredDJ'))
        sm.add_widget(PredNSD(name='PredNSD'))
        return sm


if __name__ == '__main__':
    StockApp().run()
