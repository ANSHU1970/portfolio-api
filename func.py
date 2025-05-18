import collections
import json
import urllib
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as smf
import time
import os
from eod import EodHistoricalData
from fredapi import Fred
from pandas.tseries import offsets
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.risk_models import CovarianceShrinkage
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from pandas.tseries.offsets import MonthEnd, BMonthEnd
from datetime import datetime, timedelta
from typing import List,Dict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

executor = ThreadPoolExecutor(max_workers=10)

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option("display.max_rows", 50, "display.max_columns", 6, "display.precision", 2)

api_key = os.getenv("API_KEY")
fred_api = os.getenv("FRED_API")

def get_cutoff_date(data):
    today = pd.Timestamp.today()
    today_date = today.normalize()
    if today_date != today_date + BMonthEnd(0):
        cutoff_date = (today_date - BMonthEnd(1)).date()
    else:
        cutoff_date = (today_date + BMonthEnd(0)).date()
    cutoff_date = cutoff_date.strftime('%Y-%m-%d')
    data_cutoff = data.loc[:cutoff_date]
    return data_cutoff
def get_only_cutoff_date():
    today = pd.Timestamp.today()
    today_date = today.normalize()
    if today_date != today_date + BMonthEnd(0):
        cutoff_date = (today_date - BMonthEnd(1)).date()
    else:
        cutoff_date = (today_date + BMonthEnd(0)).date()
    return cutoff_date
def get_index_constituents():
    sp500_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    nas100_list = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
    nas100_list.set_index('Symbol', inplace=True)
    sp500_list.to_csv('sp500_components.csv')
    nas100_list.to_csv('nasdaq100_components.csv')
    print(sp500_list.head())

def fred_data():
    dest_path = "./"
    if os.path.exists(dest_path + "rfr.csv"):
        print("rfr.csv already exists. Skipping FRED data download.")
        return

    dest_path = "./"
    fred = Fred(api_key=fred_api)
    rfr = pd.DataFrame(fred.get_series('DTB3'))
    rfr.rename(columns={0: 'rfr'}, inplace=True)
    rfr = rfr.ffill()
    rfr = rfr.map(lambda x: x * .01)
    rfr.to_csv(dest_path + "rfr.csv")
def mean_variance_optimization_unconstrained(data_view, risktgt, counts, max_wt):
    try:
        if data_view.shape[1] > 1:
            mu = mean_historical_return(data_view, frequency=12, log_returns=True)
            S = CovarianceShrinkage(data_view, frequency=12).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg)
            if counts == 'MV':
                weights = ef.min_volatility()
            elif counts == 'MVL':
                ef.add_constraint(lambda x: x <= max_wt)
                weights = ef.min_volatility()
            elif counts == 'RA':
                weights = ef.max_quadratic_utility(risk_aversion=1)
            elif counts == 'RAL':
                ef.add_constraint(lambda x: x <= max_wt)
                weights = ef.max_quadratic_utility(risk_aversion=1)
            elif counts == 'SR':
                weights = ef.max_sharpe(risk_free_rate=0.02)
            elif counts == 'SRL':
                ef.add_constraint(lambda x: x <= max_wt)
                weights = ef.max_sharpe(risk_free_rate=0.02)
            else:
                print(f"Error in the Optimization Loop")
                raise ValueError
            return ef.clean_weights()
        elif data_view.shape[1] == 1:
            print(f"Only One Asset")
            if data_view.columns[0] != 'SHY':
                data_view['SHY'] = 0.75
            oDict = collections.OrderedDict({data_view.columns[0]: 0.25, data_view.columns[1]: 0.75})
            return oDict
        else:
            print(f"Zero Asset")
            data_view['SHY'] = 1.0
            oDict = collections.OrderedDict({data_view.columns[0]: 1.0})
            return oDict
    except Exception as e:
        print(f"{e}")
def mean_variance_optimization_unconstrained2(data_view, cons_sec, cons_pos, pos_lim, secmin, secmax, mapper):
    try:
        if data_view.shape[1] > 1:
            mu = mean_historical_return(data_view, frequency=12, log_returns=True)
            S = CovarianceShrinkage(data_view, frequency=12).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg)
            asset_upper = {'gtaa': secmax}
            asset_lower = {'gtaa': secmin}
            if cons_sec != 0:
                ef.add_sector_constraints(sector_mapper=mapper, sector_lower=asset_lower, sector_upper=asset_upper)
            elif cons_pos != 0:
                ef.add_constraint(lambda x: x <= pos_lim)
            else:
                pass
            weights = ef.min_volatility()
            return ef.clean_weights()
        elif data_view.shape[1] == 1:
            print(f"Only One Asset")
            if data_view.columns[0] != 'SHY':
                data_view['SHY'] = 0.75
            oDict = collections.OrderedDict({data_view.columns[0]: 0.25, data_view.columns[1]: 0.75})
            return oDict
        else:
            print(f"Zero Asset")
            data_view['SHY'] = 1.0
            oDict = collections.OrderedDict({data_view.columns[0]: 1.0})
            return oDict
    except Exception as e:
        print(f"{e}")
def hrp_optimization_unconstrained(data_view):
    try:
        if data_view.shape[1] > 1:
            mu = data_view.pct_change().fillna(0)
            hrp = HRPOpt(mu)
            weights = hrp.optimize()
            return hrp.clean_weights()
        elif data_view.shape[1] == 1:
            print(f"Only One Asset")
            oDict = collections.OrderedDict({data_view.columns[0]: 1.0})
            return oDict
        else:
            print(f"Zero Asset")
            oDict = collections.OrderedDict({'No Asset': np.nan})
            return oDict
    except Exception as e:
        print(f"{e}")
def get_stock_fdata(s):
    try:
        urls = f"https://eodhistoricaldata.com/api/fundamentals/{s}.US?api_token={api_key}"
        response = urllib.request.urlopen(urls)
        data = json.loads(response.read())
        return data['Highlights']
    except Exception as e:
        print(f"Error fetching {s}: {e}")
        return None
def fetch_fundamentals_parallel(tickers):
    def fetch(sym):
        try:
            url = f"https://eodhistoricaldata.com/api/fundamentals/{sym}.US?api_token={api_key}"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            return sym, data.get('Highlights')
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            return sym, None

    futures = {executor.submit(fetch, sym): sym for sym in tickers}
    f_dict = {}
    for future in as_completed(futures):
        sym, result = future.result()
        if result:
            f_dict[sym] = result
    return pd.DataFrame.from_dict(f_dict, orient='index')

def request_index_constituents(iname, cutoff):
    urls = f"https://eodhistoricaldata.com/api/fundamentals/{iname}.US?api_token={api_key}"
    response = urllib.request.urlopen(urls)
    data = json.loads(response.read())
    df = pd.DataFrame.from_dict(data['ETF_Data']['Holdings'])
    df = df.fillna(np.nan).T.dropna()
    df.sort_values(by="Assets_%", ascending=False, inplace=True)
    if '' in df['Sector'].values:
        df = df[df['Sector'] != '']
    tickers = [s.split(".")[0] for s in df.index.tolist() if s.split(".")[0] != 'GOOG']
    fdata_df = fetch_fundamentals_parallel(tickers)
    if fdata_df.empty:
        print("No fundamental data retrieved.")
        return pd.DataFrame()
    flds = ['QuarterlyRevenueGrowthYOY', 'EBITDAMargin', 'ProfitMargin', 'ReturnOnEquityTTM']
    fdata_df = fdata_df[flds].dropna(how='any')
    fdata_df = fdata_df.astype(float)
    rev_filter = fdata_df[fdata_df['QuarterlyRevenueGrowthYOY'] >= 0.1]
    ebitda_filter = rev_filter[rev_filter['ReturnOnEquityTTM'] >= 0.2]
    profit_filter = ebitda_filter[ebitda_filter['ProfitMargin'] >= 0.15]
    profit_filter.sort_values(by='MarketCapitalization', ascending=False, inplace=True)
    if len(profit_filter) < cutoff:
        dummy_df = profit_filter.copy()
        dummy_df.loc['SHY', :] = 0.0
    else:
        dummy_df = profit_filter.iloc[:cutoff]
        dummy_df.loc['SHY', :] = 0.0
    return dummy_df
def fetch_price_parallel(tickers, client):
    def fetch(ticker):
        try:
            closing_prices = client.get_prices_eod(ticker, order='a')
            df = pd.DataFrame(closing_prices)
            df.set_index('date', inplace=True)
            return ticker, df.adjusted_close
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            return ticker, None

    futures = {executor.submit(fetch, t): t for t in tickers}
    results = {}
    for future in as_completed(futures):
        ticker, data = future.result()
        if data is not None:
            results[ticker] = data
    return pd.DataFrame(results)
def request_fdata_general(sid):
    time.sleep(1.5)
    urls = "https://eodhistoricaldata.com/api/eod-bulk-last-day/US?api_token=61d74fc2a90056.68029297&filter=extended&" \
           "symbols=" + sid + "&fmt=json"
    response = urllib.request.urlopen(urls)
    data = json.loads(response.read())
    df_bulk = pd.DataFrame.from_dict(data)
    return df_bulk
class investmentUniverse:
    def __init__(self, fname, tickers, univ_path, univs):
        if fname in ['large', 'small', 'tech', 'wealthx']:
            self.sdate = '2018-12-31'
        else:
            self.sdate = '2006-12-31'
        self.fname = fname
        self.univ_path = univ_path
        self.client = EodHistoricalData(api_key)
        self.tickers = tickers
    def generate_investment_universe(self):
        univ_df = pd.read_excel(self.univ_path + "models.xlsx", sheet_name=None, index_col=[0])
        inv_univ = univ_df['universe']
        pres_df = univ_df['preservation']
        univ_filter = pres_df.loc[pres_df['preservation'] > 0, 'preservation']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[temp_df['risk_5y'] < 5]
        temp_df.to_csv(self.univ_path + "pres_preservation.csv")
        univ_filter = pres_df.loc[pres_df['conservative income'] > 0, 'conservative income']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] < 9)]
        temp_df.to_csv(self.univ_path + "pres_cinc.csv")
        pres_df = univ_df['income']
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] <= 13)]
        temp_df.to_csv(self.univ_path + "inc_cons.csv")
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 8) & (temp_df['risk_5y'] <= 14)]
        temp_df.to_csv(self.univ_path + "inc_mod.csv")
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 9) & (temp_df['risk_5y'] <= 15)]
        temp_df.to_csv(self.univ_path + "inc_enh.csv")
        pres_df = univ_df['growth_income']
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] <= 16)]
        temp_df.to_csv(self.univ_path + "gni_cons.csv")
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 6) & (temp_df['risk_5y'] <= 17)]
        temp_df.to_csv(self.univ_path + "gni_mod.csv")
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 6) & (temp_df['risk_5y'] <= 18)]
        temp_df.to_csv(self.univ_path + "gni_enh.csv")
        pres_df = univ_df['growth']
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 10) & (temp_df['risk_5y'] <= 20)]
        temp_df.to_csv(self.univ_path + "gro_cons.csv")
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 12) & (temp_df['risk_5y'] <= 25)]
        temp_df.to_csv(self.univ_path + "gro_mod.csv")
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 14)]
        temp_df.to_csv(self.univ_path + "gro_enh.csv")
    def fetch_closing_prices(self):
        # if os.path.exists(self.univ_path + 'adj_close.csv'):
        #     print("adj_close.csv already exists. Skipping price fetching.")
        #     return
        closing_df = fetch_price_parallel(self.tickers, self.client)
        closing_df.index = pd.to_datetime(closing_df.index)
        clean_data = get_cutoff_date(closing_df)
        closing_df = clean_data.loc[self.sdate:, self.tickers]
        closing_df.to_csv(self.univ_path + 'adj_close.csv')
    def fetch_moving_averages(self, ma_type='sma'):
        # if os.path.exists(self.univ_path + 'moving_average.csv') and os.path.exists(self.univ_path + 'moving_average2.csv'):
        #     print("Moving average files already exist. Skipping download.")
        #     return
        # def fetch_ma(ticker, period):
        #     try:
        #         sma_data = self.client.get_instrument_ta(ticker, function=ma_type, period=period)
        #         df = pd.DataFrame(sma_data)
        #         df.set_index('date', inplace=True)
        #         df.rename(columns={ma_type: ticker}, inplace=True)
        #         return ticker, df
        #     except Exception as e:
        #         print(f"Failed to fetch MA for {ticker}: {e}")
        #         return ticker, pd.DataFrame()

        
        def fetch_ma(ticker, period):
            try:
                today = datetime.today()
                today_str = today.strftime('%Y-%m-%d')

                
                sma_data = self.client.get_instrument_ta(
                    ticker,
                    function=ma_type,
                    period=period,
                    from_date=today_str,
                    to_date=today_str
                )
                df = pd.DataFrame(sma_data)

                
                if df.empty:
                    
                    fallback_day = today - timedelta(days=1)
                    
                    if fallback_day.weekday() == 6:  
                        fallback_day = today - timedelta(days=2)
                   
                    elif fallback_day.weekday() == 5:  
                        fallback_day = today - timedelta(days=1)

                    fallback_str = fallback_day.strftime('%Y-%m-%d')

                    
                    sma_data = self.client.get_instrument_ta(
                        ticker,
                        function=ma_type,
                        period=period,
                        from_date=fallback_str,
                        to_date=fallback_str
                    )
                    df = pd.DataFrame(sma_data)

                
                if not df.empty:
                    df.set_index('date', inplace=True)
                    df.rename(columns={ma_type: ticker}, inplace=True)
                    return ticker, df
                else:
                    print(f"No data found for {ticker} even after fallback")
                    return ticker, pd.DataFrame()
            except Exception as e:
                print(f"Failed to fetch MA for {ticker}: {e}")
                return ticker, pd.DataFrame()


        sma_dict = {}
        sma_dict2 = {}

        # Period 50
        futures_50 = {executor.submit(fetch_ma, s, 50): s for s in self.tickers}
        for future in as_completed(futures_50):
            s, df = future.result()
            if not df.empty:
                sma_dict[s] = df[s]

        # Period 20
        futures_20 = {executor.submit(fetch_ma, s, 20): s for s in self.tickers}
        for future in as_completed(futures_20):
            s, df = future.result()
            if not df.empty:
                sma_dict2[s] = df[s]

        ta_sma_df = pd.DataFrame(sma_dict)
        ta_sma_df2 = pd.DataFrame(sma_dict2)

        ta_sma_df.to_csv(self.univ_path + 'moving_average.csv')
        ta_sma_df2.to_csv(self.univ_path + 'moving_average2.csv')
        print(ta_sma_df)
    def benchmarks_closing_prices(self):
        if os.path.exists(self.univ_path + 'bm_prices.csv') and os.path.exists(self.univ_path + 'bm_raw_px.csv'):
            print("Benchmark price files already exist. Skipping benchmark download.")
            return
        bm_dict = {}
        bm_univ = ['SP500TR.INDX', 'NDX.INDX', 'VTHR.US', 'VBTLX.US', 'VTSAX.US', 'DGSIX.US', 'AW01.INDX',
                   'NTUHCB.INDX', 'NTUICB.INDX', 'DJCBP.INDX', 'AGG.US', 'CABNX.US', 'AAANX.US', 'NAVFX.US', 'XLSR.US',
                   'VEA.US', 'EEM.US', 'SPY.US', 'SML.INDX', 'VB.US', 'VTI.US']
        for s in bm_univ:
            print(s)
            if s == 'SP500TR.INDX':
                bm_prices = self.client.get_prices_eod(s, order='a')
                temp_df = pd.DataFrame(bm_prices)
                temp_df.set_index('date', inplace=True)
                bm_df = pd.DataFrame(index=temp_df.index)
                bm_df[s] = temp_df['adjusted_close']
            else:
                bm_prices = self.client.get_prices_eod(s, order='a')
                temp_df = pd.DataFrame(bm_prices)
                temp_df.set_index('date', inplace=True)
                bm_df[s] = temp_df['adjusted_close']
        bm_df.to_csv(self.univ_path + 'bm_raw_px.csv')
        bm_px = bm_df.loc[self.sdate:, :]
        cols = {'SP500TR.INDX': 'S&P 500 Total Return Index', 'NDX.INDX': 'NASDAQ 100 Index',
                'VTHR.US': 'Russell 3000 Index', 'VBTLX.US': 'Vanguard Total Bond Market',
                'VTSAX.US': 'Vanguard Total Stock Mkt', 'DGSIX.US': 'DFA Global All 60/40',
                'AW01.INDX': 'MSCI All World Index'}
        bm_px = bm_px.rename(columns=cols)
        bm_px.to_csv(self.univ_path + "bm_prices.csv")
class investmentModels:
    def __init__(self, univs, sname, srcpath, destpath, q=0.5, window=90, adv_fees=1.0, benchmark='^GSPC',opt_type = "zscore"):
        self.sname = sname
        self.q = q
        self.out_path2 = destpath
        self.out_path = srcpath
        self.window = window
        self.fees = adv_fees / 1200
        self.benchmark = '^GSPC'
        self.client = EodHistoricalData(api_key)
        self.univ_df = univs
        self.universe = list(self.univ_df.index)
        if sname in ['large', 'small', 'tech', 'wealthx']:
            self.sdate = '2018-12-31'
        else:
            self.sdate = '2006-12-31'
    def fetch_famafrench_factors(self):
        output_path = os.path.join(self.out_path, 'parsed_famafrench.csv')
        if os.path.exists(output_path):
            print("Fama-French data already exists. Skipping download.")
            return
        try:
            ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
            os.makedirs(self.out_path, exist_ok=True)
            zip_path = os.path.join(self.out_path, 'fama_french.zip')
            output_path = os.path.join(self.out_path, 'parsed_famafrench.csv')
            urllib.request.urlretrieve(ff_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(self.out_path)
                zip_contents = zip_file.namelist()
                print(f"[DEBUG] ZIP Contents: {zip_contents}")
                csv_file = next((f for f in zip_contents if f.lower().endswith('.csv')), None)
            if not csv_file:
                raise FileNotFoundError("No CSV file found inside the Fama-French ZIP archive.")
            factors_path = os.path.join(self.out_path, csv_file)
            ff_factors = pd.read_csv(factors_path, skiprows=3, index_col=0)
            ff_factors = ff_factors[:-1]
            null_idx = ff_factors.isnull().idxmax().iloc[0]
            ff_factors = ff_factors.loc[:null_idx]
            ff_factors.dropna(inplace=True)
            ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m') + pd.offsets.MonthEnd()
            ff_factors = ff_factors.map(lambda x: float(x) / 100)
            ff_factors.to_csv(output_path)
            os.remove(zip_path)
        except Exception as e:
            raise RuntimeError(f"Error in fetch_famafrench_factors: {str(e)}")
    def ff_regression(self):
        ff_factor = pd.read_csv(self.out_path + 'parsed_famafrench.csv', index_col=[0], parse_dates=True)
        portfolio_nav = pd.read_csv(self.out_path + 'portfolio_nav.csv', index_col=[0], parse_dates=True)
        

        portfolio_df = portfolio_nav.pct_change().fillna(0)
        alpha, beta, rsq_adj, p_alpha = [], [], [], []
        for counter in range(len(portfolio_df.columns)):
            selected_port = portfolio_df.loc[:, portfolio_df.columns[counter]]
            all_data = pd.merge(pd.DataFrame(selected_port), ff_factor, how='inner', left_index=True, right_index=True)
            all_data.rename(columns={"Mkt-RF": "mkt_excess"}, inplace=True)
            all_data['port_excess'] = all_data[portfolio_df.columns[counter]] - all_data['RF']
            all_data = all_data.iloc[-60:]
            model = smf.formula.ols(formula="port_excess ~ mkt_excess + SMB + HML", data=all_data).fit()
            alpha.append(model.params['Intercept'] * 12)
            beta.append(model.params['mkt_excess'])
            rsq_adj.append(model.rsquared_adj)
            p_alpha.append(model.pvalues['Intercept'])
        return alpha, beta, rsq_adj, p_alpha
    def calculate_inverse_vol(self, data):
        self.data = data
        df_close = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        def calculate_rolling_returns(data, period):
            return (data.iloc[-1] / data.iloc[0]) - 1
        def calculate_rolling_std(data, period):
            return data.std() * np.sqrt(252)
        rolling_ret = df_close.rolling(window=self.window).apply(calculate_rolling_returns, args=(self.window,))
        rolling_std = rolling_ret.rolling(window=self.window).apply(np.std)
        inverse_weights = rolling_std.map(lambda x: 1 / x)
        return inverse_weights
    def filter_assets_based_on_moving_averages(self):
        df_close = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        df_close = df_close.loc['2008-01-31':]
        bm_prices = pd.read_csv('benchmark.csv', index_col=[0], parse_dates=True)
        bm_select = bm_prices[self.benchmark]
        def calculate_rolling_returns(data, period):
            return ((data.iloc[-1] / data.iloc[0]) ** (250 / period)) - 1
        df_combined_px = df_close.copy()
        df_combined_px['bm'] = bm_select
        df_rolling_ret = df_combined_px.rolling(window=self.window).apply(calculate_rolling_returns,
                                                                          args=(self.window,))
        df_delta = df_rolling_ret.sub(df_rolling_ret.bm, axis=0)
        positive_delta = df_delta.gt(0)
        positive_relative_returns = df_delta[positive_delta]
        filter = positive_relative_returns.quantile(self.q, axis=1, numeric_only=True)
        delta_filter = positive_relative_returns.sub(filter, axis=0)
        positive_filter_df = delta_filter.gt(0)
        filtered_pos_rel_return = delta_filter[positive_filter_df]
        filtered_pos_rel_return_monthly = filtered_pos_rel_return.resample('ME', closed='right').last()
        filtered_pos_rel_return_monthly.to_csv(self.out_path + f'{self.sname}_ew_holdings.csv')
        df_return = df_close.resample('ME', closed='right').last()
        df_return = df_return.pct_change().fillna(0).shift(-1)
        portfolio_ret = df_return[filtered_pos_rel_return_monthly.notnull()].mean(axis=1).shift(1).fillna(0)
        inv_vol = self.calculate_inverse_vol(df_close)
        inv_vol = inv_vol.resample('ME', closed='right').last()
        selected_asset = inv_vol[filtered_pos_rel_return_monthly.notnull()]
        inverse_weights = selected_asset.div(selected_asset.sum(axis=1), axis=0)
        inverse_wt_port = df_return.mul(inverse_weights, axis=1).sum(axis=1).shift(1).fillna(0)
        inverse_weights.to_csv(self.out_path + f'{self.sname}_inverse_holdings.csv')
        wts_check = inverse_weights.sum(axis=1)
        def clean_wts(data): return data.dropna().apply(lambda x: x / data.sum())
        delta_wts = filtered_pos_rel_return.apply(clean_wts, axis=1)
        delta_wts = delta_wts.resample('ME', closed='right').last()
        delta_wts.to_csv(self.out_path + f'{self.sname}_alpha_holdings.csv')
        alpha_wt_port = df_return.mul(delta_wts, axis=1).sum(axis=1).shift(1).fillna(0)
        z_df = pd.DataFrame(
        filtered_pos_rel_return_monthly.apply(lambda row: zscore(row, nan_policy='omit'), axis=1).tolist(),
        index=filtered_pos_rel_return_monthly.index,
        columns=filtered_pos_rel_return_monthly.columns
        )
        df_cdf = z_df.apply(norm.cdf)
        sum_t = df_cdf.sum(axis=1)
        zscore_wts = df_cdf.div(sum_t, axis=0)
        zscore_wts.to_csv(self.out_path + f'{self.sname}_zscore_holdings.csv')
        zscore_wt_port = df_return.mul(zscore_wts, axis=1).sum(axis=1).shift(1).fillna(0)
        bm_ret = bm_prices.loc[portfolio_ret.index[0]:, self.benchmark]
        combined_ret_df = pd.DataFrame(portfolio_ret, columns=['portfolio'])
        combined_ret_df.loc[:, 'inverse'] = inverse_wt_port
        combined_ret_df.loc[:, 'alpha'] = alpha_wt_port
        combined_ret_df.loc[:, 'zscore'] = zscore_wt_port
        combined_ret_df.loc[:, 'bm'] = bm_ret.resample('ME', closed='right').last().pct_change()
        filter = positive_relative_returns.quantile(.2, axis=1, numeric_only=True)
        filtered_pos_rel_return = positive_relative_returns[positive_relative_returns.lt(filter, axis=0)]
        filtered_pos_rel_return_monthly = filtered_pos_rel_return.resample('ME', closed='right').last()
        filtered_pos_rel_return_monthly.to_csv(self.out_path + f'{self.sname}_lower_quintile_holdings.csv')
        pr2 = df_return[filtered_pos_rel_return_monthly.notnull()].mean(axis=1).shift(1).fillna(0)
        combined_ret_df.loc[:, 'lower_quintile'] = pr2
        combined_ret_df = combined_ret_df.map(lambda x: x - self.fees)
        combined_ret_df.iloc[0] = 0.0
        combined_ret_df = 100 * combined_ret_df.add(1).cumprod()
        combined_ret_df = combined_ret_df[['portfolio', 'lower_quintile', 'inverse', 'alpha','zscore']].iloc[:-1]
        bm_ret = bm_prices.loc[combined_ret_df.index[0]:combined_ret_df.index[-1]]
        bm_ret = bm_ret.resample('ME', closed='right').last().pct_change().fillna(0)
        bm_ret = 100 * bm_ret.add(1).cumprod()
        combined_ret_df = pd.concat([combined_ret_df, bm_ret], axis=1, sort=False)
        combined_ret_df.to_csv(self.out_path + 'portfolio_nav.csv')
    def generate_optimized_monthly_weights(self, counts):
        df_sma = pd.read_csv(self.out_path + 'monthly_holdings.csv', index_col=[0], parse_dates=True)
        df_close = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        df_close = df_close.loc[:, df_sma.columns]
        df_close = df_close.loc[self.sdate:]
        monthly_price = df_close.resample('M', closed='right').last()
        monthly_price = monthly_price.loc[df_sma.index[0]: df_sma.index[-1], :]
        uncon_mvo_opt_wts = {}
        uncon_hrp_opts_wts = {}
        for counter in range(len(monthly_price) - 12):
            temp = df_sma.iloc[counter + 12]
            sym_list = temp[temp == True].index.to_list()
            sample_data = monthly_price.loc[monthly_price.index[counter]:monthly_price.index[counter + 12], sym_list]
            sample_data.dropna(axis=1, inplace=True)
            if self.opt_type == 'uc_mvo':
                uncon_mvo_opt_wts.update({monthly_price.index[counter + 12]:
                                              mean_variance_optimization_unconstrained(sample_data, self.tgtrisk,
                                                                                       counts, self.maxwts)})
        if self.opt_type == 'uc_mvo':
            umvo_wts_df = pd.DataFrame.from_dict(uncon_mvo_opt_wts).T
            umvo_wts_df = pd.DataFrame(umvo_wts_df, columns=df_close.columns)
            wts_cash = umvo_wts_df[umvo_wts_df.isnull().sum(axis=1) == len(umvo_wts_df.columns)]
            umvo_wts_df.loc[wts_cash.index, ['SHY']] = 1.0
            umvo_wts_df.to_csv(self.out_path2 + '{}_unconstrained_mvo_wts.csv'.format(self.sname))
            umvo_wts_df.to_csv(self.out_path + '{}_unconstrained_mvo_wts.csv'.format(self.sname))
        else:
            unc_hrp_wts_df = pd.DataFrame.from_dict(uncon_hrp_opts_wts).T
            unc_hrp_wts_df = unc_hrp_wts_df[df_close.columns]
            wts_cash = unc_hrp_wts_df[unc_hrp_wts_df.isnull().sum(axis=1) == len(unc_hrp_wts_df.columns)]
            unc_hrp_wts_df.loc[wts_cash.index, ['SHY']] = 1.0
            unc_hrp_wts_df.to_csv(self.out_path + '{}_unconstrained_hrp_wts.csv'.format(self.sname))
    def portfolio_backtest(self, adv_fees=1.0):
        global wts_df
        price_df = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        resampled_px = price_df.resample('ME', closed='right').last()
        bm_prices = pd.read_csv(self.out_path + 'bm_prices.csv', index_col=[0], parse_dates=True)
        resampled_bm = bm_prices.resample('ME', closed='right').last()
        resampled_bm['BMUS6040'] = 0.6 * resampled_bm['S&P 500 Total Return Index'] + 0.4 * resampled_bm[
            'Vanguard Total Bond Market']
        if self.opt_type == 'uc_mvo':
            wts_df = pd.read_csv(self.out_path + '{}_unconstrained_mvo_wts.csv'.format(self.sname), index_col=[0],
                                 parse_dates=True)
        elif self.opt_type == 'uc_hrp':
            wts_df = pd.read_csv(self.out_path + '{}_unconstrained_hrp_wts.csv'.format(self.sname), index_col=[0],
                                 parse_dates=True)
        sdate = wts_df.index[0]
        resampled_px_adj = resampled_px.loc[sdate:].copy()
        monthly_ret = resampled_px_adj.pct_change()
        wts_col = wts_df.columns.to_list()
        if 'No Asset' in wts_col:
            wts_df.drop('No Asset', axis=1, inplace=True)
            wts_col = wts_df.columns.to_list()
        port_ret = monthly_ret.shift(-1).multiply(wts_df).shift(1).sum(axis=1)
        returns_df = pd.DataFrame(port_ret, columns=[self.opt_type])
        returns_df['eq_wt'] = resampled_px.pct_change().fillna(0).mean(axis=1)
        monthly_fees = (0.01 * adv_fees) / 12
        returns_df = returns_df.map(lambda x: x - monthly_fees)
        returns_df.iloc[0] = 0.0
        returns_df = pd.merge(returns_df, resampled_bm.pct_change().fillna(0), left_index=True, right_index=True)
        returns_df.iloc[0] = 0.0
        portfolio_nav = 100 * returns_df.add(1).cumprod()
        # portfolio_nav.to_csv(self.out_path + 'portfolio_nav.csv')
        output_path = os.path.join(self.out_path, 'portfolio_nav.csv')
        portfolio_nav.to_csv(output_path)
        # returns_df.to_csv(self.out_path + 'backtest_returns.csv')
        maxwts = wts_df.max(axis=1)
        print(maxwts.sort_values(ascending=False))
        print(wts_df.min(axis=1).sort_values(ascending=False))
        print('*' * 100)
        print(wts_df.iloc[-5:].fillna(0).T)
    def portfolio_analytics(self):
        # read_navs = pd.read_csv(self.out_path + "portfolio_nav.csv", index_col=[0], parse_dates=True)
        read_navs = pd.read_csv(os.path.join(self.out_path, "portfolio_nav.csv"), index_col=[0], parse_dates=True)
        bm_px = pd.read_csv(self.out_path + 'rfr.csv', index_col=[0], parse_dates=True)
        bm_px = bm_px.resample('ME', closed='right').last()
        bm_px = bm_px.loc[read_navs.index[0]: read_navs.index[-1], :]
        frame2 = read_navs.copy()
        try:
            stats_df = pd.DataFrame(columns=frame2.columns)
            yearly = frame2.copy()
            if pd.Series(frame2.index[0]).dt.is_year_end[0]:
                yearly = frame2.resample('YE', closed='right').last().pct_change()
            else:
                dummy_yr = frame2.index[0] + offsets.YearEnd(-1)
                yearly.loc[dummy_yr, :] = yearly.iloc[0]
                yearly.sort_index(inplace=True)
                yearly = yearly.resample('YE', closed='right').last().pct_change()
            portfolio_nav = frame2.copy()
            N = 12
            rfr_ts = bm_px.iloc[-(N + 1):]
            rfr = rfr_ts.mean()
            one_yr = portfolio_nav.iloc[-(N + 1):]
            r1 = 1 + (one_yr.pct_change().dropna())
            r1 = r1.cumprod().iloc[-1] ** (12 / N) - 1
            risk1 = one_yr.iloc[1:].pct_change().std() * np.sqrt(12)
            sharpe1 = r1.apply(lambda x: (x - rfr))
            sharpe1 = sharpe1.divide(risk1, axis=0)
            f1 = False
            N = 36
            if len(portfolio_nav) < N:
                f3 = True
            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                three_yr = portfolio_nav.iloc[-(N + 1):]
                r3 = 1 + (three_yr.pct_change().dropna())
                r3 = r3.cumprod().iloc[-1] ** (12 / N) - 1
                risk3 = three_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe3 = r3.apply(lambda x: (x - rfr))
                sharpe3 = sharpe3.divide(risk3, axis=0)
                f3 = False
            N = 60
            if len(portfolio_nav) < N:
                f5 = True
            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                five_yr = portfolio_nav.iloc[-(N + 1):]
                r5 = 1 + (five_yr.pct_change().dropna())
                r5 = r5.cumprod().iloc[-1] ** (12 / N) - 1
                risk5 = five_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe5 = r5.apply(lambda x: (x - rfr))
                sharpe5 = sharpe5.divide(risk5, axis=0)
                f5 = False
            N = 120
            if len(portfolio_nav) < N:
                f10 = True
            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                ten_yr = portfolio_nav.iloc[-(N + 1):]
                r10 = 1 + (ten_yr.pct_change().dropna())
                r10 = r10.cumprod().iloc[-1] ** (12 / N) - 1
                risk10 = ten_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe10 = r10.apply(lambda x: (x - rfr))
                sharpe10 = sharpe10.divide(risk10, axis=0)
                f10 = False
            N = len(frame2) - 1
            rfr = bm_px.mean()
            ri = 1 + (frame2.pct_change().dropna())
            ri = ri.cumprod().iloc[-1] ** (12 / N) - 1
            riski = frame2.pct_change().std() * np.sqrt(12)
            sharpei = ri.apply(lambda x: (x - rfr))
            sharpei = sharpei.divide(riski, axis=0)
            stats_df.loc['ytd', :] = yearly.loc[yearly.index[-1], :].values
            stats_df.loc['ytd', :] = stats_df.loc['ytd', :].apply(lambda x: round(x, 4))
            if f1:
                stats_df.loc['annualized_return_1y', :] = 0
                stats_df.loc['annualized_risk_1y', :] = 0
                stats_df.loc['sharpe_ratio_1y', :] = 0
            else:
                stats_df.loc['annualized_return_1y', :] = r1.apply(lambda x: round(x, 4)).values
                stats_df.loc['annualized_risk_1y', :] = risk1.apply(lambda x: round(x, 4)).values
                stats_df.loc['sharpe_ratio_1y', :] = sharpe1.apply(lambda x: round(x, 4)).values.flatten()
            if f3:
                stats_df.loc['annualized_return_3y', :] = 0
                stats_df.loc['annualized_risk_3y', :] = 0
                stats_df.loc['sharpe_ratio_3y', :] = 0
            else:
                stats_df.loc['annualized_return_3y', :] = r3.apply(lambda x: round(x, 4)).values
                stats_df.loc['annualized_risk_3y', :] = risk3.apply(lambda x: round(x, 4)).values
                stats_df.loc['sharpe_ratio_3y', :] = sharpe3.apply(lambda x: round(x, 4)).values.flatten()
            if f5:
                stats_df.loc['annualized_return_5y', :] = 0
                stats_df.loc['annualized_risk_5y', :] = 0
                stats_df.loc['sharpe_ratio_5y', :] = 0
            else:
                stats_df.loc['annualized_return_5y', :] = r5.apply(lambda x: round(x, 4)).values
                stats_df.loc['annualized_risk_5y', :] = risk5.apply(lambda x: round(x, 4)).values
                stats_df.loc['sharpe_ratio_5y', :] = sharpe5.apply(lambda x: round(x, 4)).values.flatten()
            if f10:
                stats_df.loc['annualized_return_10y', :] = 0
                stats_df.loc['annualized_risk_10y', :] = 0
                stats_df.loc['sharpe_ratio_10y', :] = 0
            else:
                stats_df.loc['annualized_return_10y', :] = r10.apply(lambda x: round(x, 4)).values
                stats_df.loc['annualized_risk_10y', :] = risk10.apply(lambda x: round(x, 4)).values
                stats_df.loc['sharpe_ratio_10y', :] = sharpe10.apply(lambda x: round(x, 4)).values.flatten()
            stats_df.loc['annualized_return_inception', :] = ri.apply(lambda x: round(x, 4)).values
            stats_df.loc['annualized_risk_inception', :] = riski.apply(lambda x: round(x, 4)).values
            stats_df.loc['sharpe_ratio_inception', :] = sharpei.apply(lambda x: round(x, 4)).values.flatten()
            stats_df.loc['cagr', :] = round((frame2.iloc[-1] / frame2.iloc[0] - 1), 4)
            stats_df.loc['$_growth', :] = round(frame2.iloc[-1], 4)
            roll_max = portfolio_nav.rolling(min_periods=1, window=12).max()
            rolling_drawdown = portfolio_nav / roll_max - 1.0
            max_dd = rolling_drawdown.min()
            asset_1 = portfolio_nav.loc[:, portfolio_nav.columns[0]]
            dd_end1 = np.argmax(np.maximum.accumulate(asset_1) - asset_1)
            dd_start1 = np.argmax(asset_1[:dd_end1])
            asset_2 = portfolio_nav.loc[:, portfolio_nav.columns[1]]
            dd_end2 = np.argmax(np.maximum.accumulate(asset_2) - asset_2)
            dd_start2 = np.argmax(asset_2[:dd_end2])
            worst_ret = yearly.min().values
            stats_df.loc['avgdd', :] = abs(round(rolling_drawdown.mean(), 2)).values
            stats_df.loc['maxdd', :] = abs(round(max_dd, 2)).values
            stats_df.loc['skew', :] = round(frame2.pct_change().skew(), 2).values
            stats_df.loc['kurtosis', :] = round(frame2.pct_change().kurtosis(), 2).values
            stats_df.loc['min_ret', :] = yearly.min().apply(lambda x: round(x, 4)).values
            stats_df.loc['max_ret', :] = yearly.max().apply(lambda x: round(x, 4)).values
            alpha, beta, rsq_adj, p_alpha = self.ff_regression()
            stats_df.loc['alpha', :] = alpha
            stats_df.loc['beta', :] = beta
            stats_df.loc['rsq_adj', :] = rsq_adj
            stats_df.loc['p_value_alpha', :] = p_alpha
            yearly.dropna(inplace=True)
            yearly.index = pd.to_datetime(yearly.index).strftime('%Y')
            yearly.loc['ytd', :] = stats_df.loc['ytd', :].values
            yearly.loc['since_inception', :] = stats_df.loc['annualized_return_inception', :].values
            writer = pd.ExcelWriter(self.out_path + '{}_analysis_output.xlsx'.format(self.sname),
                                    engine='xlsxwriter')
            stats_df.to_excel(writer, sheet_name="port_statistics")
            portfolio_nav.index = portfolio_nav.index.strftime('%m/%d/%Y')
            frame2.index = frame2.index.strftime('%m/%d/%Y')
            frame2.to_excel(writer, sheet_name='time_series')
            yearly.to_excel(writer, sheet_name="yearly")
            df_roll_return = frame2.rolling(36).apply(lambda x: x.iloc[-1] / x.iloc[0]) ** (1 / 3) - 1
            df_roll_return.to_excel(writer, sheet_name="rolling_returns")
            writer.close()
            return stats_df
        except Exception as e:
            print("Error Occurred in generating stats_df dataframe", e)
            raise ValueError
    def bear_mkt_analysis(self):
        bearmkt_dates = pd.read_csv(self.out_path + 'bear_mkt_dates.csv', index_col=[0], parse_dates=True)
        read_navs = pd.read_csv(self.out_path + "portfolio_nav.csv", index_col=[0], parse_dates=True)
        idx_ls = ['dotcom', 'sept_911', 'financial_crisis', 'flash_crash2010', 'fomc_oct2018', 'covid_2020']
        per_ret = []
        for lbls in range(len(idx_ls)):
            b1 = bearmkt_dates.iloc[lbls, 0]
            e1 = bearmkt_dates.iloc[lbls, 1]
            temp = read_navs.loc[b1:e1]
            val = temp.iloc[-1] / temp.iloc[0] - 1
            per_ret.append(val.values.tolist())
        bear_mkt_df = pd.DataFrame(per_ret, index=idx_ls, columns=read_navs.columns)
        bear_mkt_df.to_csv(self.out_path + f"bear_mkt_{self.sname}.csv")
    def rebalance_trades(self, param_sheet, dest_dir, cutoff_date):
        upload_cols = ['Model', 'Description', 'Symbol', 'Target Percent', 'Maximum Percent', 'Minimum Percent']
        ls_strategy = param_sheet.index.tolist()
        for s in ls_strategy:
            wt_scheme = param_sheet.loc[s, 'wt_scheme']
            holdings_file = f"{self.out_path}{s}_{wt_scheme}_holdings.csv"
            if not os.path.exists(holdings_file):
                print(f"Skipping {s} - holdings file not found")
                continue
            try:
                wts_df = pd.read_csv(holdings_file, index_col=[0], parse_dates=True)
                wts_df = wts_df.loc[:cutoff_date, :]
                trades = wts_df.iloc[-1].dropna()
                if wt_scheme == 'ew':
                    trades.loc[:] = 1 / trades.count()
                trades = trades.map(lambda x: x / sum(trades))
                trades.iloc[0] = (1 - trades.sum()) + trades.iloc[0]
                df = pd.DataFrame(index=trades.index, columns=upload_cols)
                df['Model'] = param_sheet.loc[s, 'model_name']
                df['Description'] = param_sheet.loc[s, 'Description']
                df['Symbol'] = trades.index
                df['Target Percent'] = trades.values
                df['Target Percent'] = df['Target Percent'].apply(lambda x: x * 100)
                df['Minimum Percent'] = 'global'
                df['Maximum Percent'] = 'global'
                df.set_index('Model', drop=True, inplace=True)
                df.to_csv(os.path.join(dest_dir, f"{s}_upload.csv"))
                print(f"Trade file created for {s}")
            except Exception as e:
                print(f"Error processing {s}: {str(e)}")
    def generate_file_to_upload(self):
        def calibrate_data(data):
            tot = data.sum()
            count = 0
            adj_val = []
            for val in data:
                if val != 0.0 and count == 0:
                    val = val + round((1 - tot), 2)
                    adj_val.append(val)
                    count = 1
                else:
                    adj_val.append(val)
            return adj_val
        df = pd.read_csv(self.out_path + self.sname + '_unconstrained_mvo_wts.csv', index_col=[0], parse_dates=True)
        df.fillna(0, inplace=True)
        df = round(df.div(df.sum(axis=1), axis=0), 2)
        adj_lst = []
        for cntr in range(len(df)):
            adj_lst.append(calibrate_data(df.iloc[cntr]))
        df = pd.DataFrame(adj_lst, columns=df.columns, index=df.index)
        tt = df.stack().to_frame()
        tt = tt.reset_index()
        tt.set_index(tt.columns[0], inplace=True)
        tt = tt.rename(columns={tt.columns[0]: 'Symbol', tt.columns[1]: 'Target Weight'})
        tt.index.name = 'Date'
        tt = tt[tt['Target Weight'] != 0]
        tt.to_csv(self.out_path + self.sname + '_upload.csv'), 0



def process_custom_model_with_tickers(
    model_name: str,
    tickers: List[str],
    weights: List[float],
    all_params: pd.DataFrame,
    cutoff_date: datetime,
    univ_df: Dict[str, pd.DataFrame],
    output_dir: str = "./"
):
    from func import investmentUniverse, investmentModels, fred_data

    try:
        tickers = [t.replace('/', '-') for t in tickers]

        # ✅ Create DataFrame from tickers and weights
        dummy_df = pd.DataFrame(index=tickers)
        dummy_df.index.name = "ticker"
        dummy_df["weight"] = weights

        # ✅ Create output directory
        dest_dir = os.path.join(output_dir, str(cutoff_date))
        os.makedirs(dest_dir, exist_ok=True)

        # ✅ Extract model parameters
        q_thres = all_params.loc[model_name, 'thres_q']
        rolling_window = all_params.loc[model_name, 'days']
        wt_scheme = all_params.loc[model_name, 'wt_scheme']
        benchmark = all_params.loc[model_name, 'benchmark']
        adv_fees = 1.65 if model_name in ['large', 'tech', 'small', 'wealthx'] else 1.0

        # ✅ Initialize and run universe & model
        univ = investmentUniverse(model_name, tickers, "./", dummy_df)
        univ.fetch_closing_prices()
        univ.fetch_moving_averages()
        univ.benchmarks_closing_prices()

        model = investmentModels(dummy_df, model_name, "./", "./", q_thres, rolling_window, adv_fees, benchmark)
        model.fetch_famafrench_factors()
        fred_data()
        model.filter_assets_based_on_moving_averages()
        stats = model.portfolio_analytics()
        stats.to_csv(os.path.join(dest_dir, f"{model_name}.csv"))

        return {
            "status": "success",
            "output_file": os.path.join(dest_dir, f"{model_name}_analysis_output.xlsx"),
            "tickers": tickers
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
