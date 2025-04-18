'''Next version of trading_model_v1. the model is based on relative performance to its benchmark - 01/01/2025'''
"""Final Version revised on 3/31/2023. This version saves data on a external HDD"""
"""The model to filter universe based on Revenue, Profit and Earnings growth, and optimize portfolio for weights"""
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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed




pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option("display.max_rows", 50, "display.max_columns", 6, "display.precision", 2)

api_key = '61d74fc2a90056.68029297'
fred_api = 'ee9dfbde992f15f10f4a37ecc5809a7e'


# sdate = '2018-12-31'

# Function to determine the cutoff date
def get_cutoff_date(data):
    # Get today's date
    today = pd.Timestamp.today()

    # Remove the timestamp from today's date
    today_date = today.normalize()

    # Determine the cutoff date based on the conditions
    if today_date != today_date + BMonthEnd(0):  # If today is NOT the last business day of the month
        cutoff_date = (today_date - BMonthEnd(1)).date()  # Last business day of the previous month
    else:  # If today IS the last business day of the month
        cutoff_date = (today_date + BMonthEnd(0)).date()  # Last business day of the current month

    # Filter the time series based on the cutoff date
    cutoff_date = cutoff_date.strftime('%Y-%m-%d')
    data_cutoff = data.loc[:cutoff_date]

    return data_cutoff

def get_only_cutoff_date():
    # Get today's date
    today = pd.Timestamp.today()

    # Remove the timestamp from today's date
    today_date = today.normalize()

    # Determine the cutoff date based on the conditions
    if today_date != today_date + BMonthEnd(0):  # If today is NOT the last business day of the month
        cutoff_date = (today_date - BMonthEnd(1)).date()  # Last business day of the previous month
    else:  # If today IS the last business day of the month
        cutoff_date = (today_date + BMonthEnd(0)).date()  # Last business day of the current month

    return cutoff_date


def get_index_constituents():
    sp500_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    nas100_list = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')[4]
    nas100_list.set_index('Symbol', inplace=True)
    sp500_list.to_csv('sp500_components.csv')
    nas100_list.to_csv('nasdaq100_components.csv')
    print(sp500_list.head())


def fred_data():
    dest_path = "./"
    """Fetch 3-months treasury bill"""
    fred = Fred(api_key=fred_api)
    # Risk Free Rate: 3 - Months T-bill Index
    rfr = pd.DataFrame(fred.get_series('DTB3'))
    rfr.rename(columns={0: 'rfr'}, inplace=True)
    rfr = rfr.ffill()
    rfr = rfr.map(lambda x: x * .01)
    # rfr.to_csv("C:/Users/yprasad/proj_models/data/rfr.csv")
    rfr.to_csv(dest_path + "rfr.csv")


def mean_variance_optimization_unconstrained(data_view, risktgt, counts, max_wt):
    try:
        if data_view.shape[1] > 1:
            # if the filtered dataframe has only 1 asset, no need to optimize
            # mu = capm_return(data_view, frequency=12, log_returns=True)
            mu = mean_historical_return(data_view, frequency=12, log_returns=True)
            # mu = ema_historical_return(data_view, frequency=12, log_returns=True, span=12)
            S = CovarianceShrinkage(data_view, frequency=12).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg)
            # Asset class restrictions
            # ef.add_constraint(lambda x: x <= 0.4)
            # weights = ef.max_quadratic_utility(risk_aversion=1)
            # max_wt = 0.4
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
            # weights = ef.max_sharpe()
            # weights = ef.efficient_return(target_return=0.15)
            # weights = ef.efficient_risk(target_volatility=risktgt)
            return ef.clean_weights()

        elif data_view.shape[1] == 1:
            # Since the filtered data has only one name and its not Cash position, split the allocation between the
            # name and the cash to manage concentration risk
            print(f"Only One Asset")
            if data_view.columns[0] != 'SHY':
                data_view['SHY'] = 0.75
            oDict = collections.OrderedDict({data_view.columns[0]: 0.25, data_view.columns[1]: 0.75})
            return oDict

        else:
            # If no asset filters through signal, allocation all to cash
            print(f"Zero Asset")
            # oDict = collections.OrderedDict({'No Asset': np.nan})
            data_view['SHY'] = 1.0
            oDict = collections.OrderedDict({data_view.columns[0]: 1.0})
            return oDict
    except Exception as e:
        print(f"{e}")


def mean_variance_optimization_unconstrained2(data_view, cons_sec, cons_pos, pos_lim, secmin, secmax, mapper):
    try:
        if data_view.shape[1] > 1:
            # if the filtered dataframe has only 1 asset, no need to optimize
            # mu = capm_return(data_view, frequency=12, log_returns=True)
            mu = mean_historical_return(data_view, frequency=12, log_returns=True)
            # mu = ema_historical_return(data_view, frequency=12, log_returns=True, span=12)
            S = CovarianceShrinkage(data_view, frequency=12).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg)

            # Asset class restrictions
            # asset_lower = {'top_30_nas100': 0.5, 'top_15_sml600': 0.1, 'gtaa': 0.4}
            # asset_upper = {'top_30_nas100': 0.7, 'top_15_sml600': 0.25, 'gtaa': 0.6}
            asset_upper = {'gtaa': secmax}
            asset_lower = {'gtaa': secmin}
            if cons_sec != 0:
                ef.add_sector_constraints(sector_mapper=mapper, sector_lower=asset_lower, sector_upper=asset_upper)
            elif cons_pos != 0:
                ef.add_constraint(lambda x: x <= pos_lim)
            else:
                pass
            # ef.add_constraint(lambda x: x >= 0)
            # ef.add_constraint(lambda x: x <= 0.15)
            # weights = ef.max_quadratic_utility(risk_aversion=5)
            weights = ef.min_volatility()
            # weights = ef.max_sharpe()
            # weights = ef.efficient_return(target_return=0.09)
            # weights = ef.efficient_risk(target_volatility=0.1)
            return ef.clean_weights()

        elif data_view.shape[1] == 1:
            # Since the filtered data has only one name and its not Cash position, split the allocation between the
            # name and the cash to manage concentration risk
            print(f"Only One Asset")
            if data_view.columns[0] != 'SHY':
                data_view['SHY'] = 0.75
            oDict = collections.OrderedDict({data_view.columns[0]: 0.25, data_view.columns[1]: 0.75})
            return oDict

        else:
            # If no asset filters through signal, allocation all to cash
            print(f"Zero Asset")
            # oDict = collections.OrderedDict({'No Asset': np.nan})
            data_view['SHY'] = 1.0
            oDict = collections.OrderedDict({data_view.columns[0]: 1.0})
            return oDict
    except Exception as e:
        print(f"{e}")


def hrp_optimization_unconstrained(data_view):
    try:
        if data_view.shape[1] > 1:
            mu = data_view.pct_change().fillna(0)
            # S = CovarianceShrinkage(mu, frequency=12).ledoit_wolf()
            hrp = HRPOpt(mu)
            # hrp.add_constraint(lambda x: x <= 0.15)
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

# Threaded version of fetching fundamentals
def fetch_fundamentals_parallel(tickers):
    f_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(get_stock_fdata, sym): sym for sym in tickers}
        for future in as_completed(future_to_ticker):
            sym = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    f_dict[sym] = result
            except Exception as e:
                print(f"Error processing {sym}: {e}")
    return pd.DataFrame.from_dict(f_dict, orient='index')


# Optimized version of request_index_constituents
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
        closing_prices = client.get_prices_eod(ticker, order='a')
        df = pd.DataFrame(closing_prices)
        df.set_index('date', inplace=True)
        return ticker, df.adjusted_close

    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch, t) for t in tickers]
        for future in as_completed(futures):
            try:
                ticker, data = future.result()
                results[ticker] = data
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
    return pd.DataFrame.from_dict(results)

def request_fdata_general(sid):
    """Request fundamental data for the curated universe list"""
    time.sleep(1.5)
    urls = "https://eodhistoricaldata.com/api/eod-bulk-last-day/US?api_token=61d74fc2a90056.68029297&filter=extended&" \
           "symbols=" + sid + "&fmt=json"
    # urls = "https://eodhistoricaldata.com/api/fundamentals/{}.US?api_token={}&filter=Highlights".format(sid, api_key)
    response = urllib.request.urlopen(urls)
    data = json.loads(response.read())
    df_bulk = pd.DataFrame.from_dict(data)
    return df_bulk


class investmentUniverse:
    """API call to pull the index constituents, rank them by universe selection rules"""

    def __init__(self, fname, tickers, univ_path, univs):
        # self.univ_path = "C:/Users/yprasad/proj_models/model_portfolios/universe/"
        # self.client = EodHistoricalData(api_key)
        # # self.start = '1999-12-31'
        # self.start = '2014-07-30'
        # self.univ_df = pd.read_excel(self.univ_path + "models.xlsx", sheet_name=None, index_col=[0])
        # self.inv_univ = self.univ_df['universe']
        if fname in ['large', 'small', 'tech', 'wealthx']:
            self.sdate = '2018-12-31'
        else:
            self.sdate = '2006-12-31'

        self.fname = fname
        self.univ_path = univ_path
        # self.univ_path = "C:/Users/yprasad/proj_models/model_portfolios/cam_models/"
        self.client = EodHistoricalData(api_key)
        # self.start = '1999-12-31'
        # self.univ_df = pd.read_excel(self.univ_path + "universe.xlsx", index_col=[0], sheet_name=None)
        # self.sh_names = list(self.univ_df.keys())
        # self.params = self.univ_df[self.sh_names[0]]
        # self.tickers = list(self.univ_df['Ticker'])
        self.tickers = tickers

    # Module not required in this case
    def generate_investment_universe(self):
        univ_df = pd.read_excel(self.univ_path + "models.xlsx", sheet_name=None, index_col=[0])
        inv_univ = univ_df['universe']

        # create universe for preservation models
        pres_df = univ_df['preservation']
        # universe file for preservation - preservation model
        univ_filter = pres_df.loc[pres_df['preservation'] > 0, 'preservation']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[temp_df['risk_5y'] < 5]
        temp_df.to_csv(self.univ_path + "pres_preservation.csv")

        # universe file for preservation - preservation model
        univ_filter = pres_df.loc[pres_df['conservative income'] > 0, 'conservative income']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] < 9)]
        temp_df.to_csv(self.univ_path + "pres_cinc.csv")

        # create universe for income models
        pres_df = univ_df['income']
        # universe file for income - conservative model
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] <= 13)]
        temp_df.to_csv(self.univ_path + "inc_cons.csv")

        # universe file for income - moderate model
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 8) & (temp_df['risk_5y'] <= 14)]
        temp_df.to_csv(self.univ_path + "inc_mod.csv")

        # universe file for income - enhanced model
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 9) & (temp_df['risk_5y'] <= 15)]
        temp_df.to_csv(self.univ_path + "inc_enh.csv")

        # create universe for ginc models
        pres_df = univ_df['growth_income']
        # universe file for income - conservative model
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 4) & (temp_df['risk_5y'] <= 16)]
        temp_df.to_csv(self.univ_path + "gni_cons.csv")

        # universe file for ginc - moderate model
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 6) & (temp_df['risk_5y'] <= 17)]
        temp_df.to_csv(self.univ_path + "gni_mod.csv")

        # universe file for ginc - enhanced model
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] > 6) & (temp_df['risk_5y'] <= 18)]
        temp_df.to_csv(self.univ_path + "gni_enh.csv")

        # create universe for growth models
        pres_df = univ_df['growth']
        # universe file for growth - conservative model
        univ_filter = pres_df.loc[pres_df['conservative'] > 0, 'conservative']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 10) & (temp_df['risk_5y'] <= 20)]
        temp_df.to_csv(self.univ_path + "gro_cons.csv")

        # universe file for growth - moderate model
        univ_filter = pres_df.loc[pres_df['moderate'] > 0, 'moderate']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 12) & (temp_df['risk_5y'] <= 25)]
        temp_df.to_csv(self.univ_path + "gro_mod.csv")

        # universe file for growth - enhanced model
        univ_filter = pres_df.loc[pres_df['enhanced'] > 0, 'enhanced']
        temp_df = inv_univ[inv_univ.asset.apply(lambda x: x in univ_filter.index)]
        temp_df = temp_df[(temp_df['risk_5y'] >= 14)]
        temp_df.to_csv(self.univ_path + "gro_enh.csv")

    def fetch_closing_prices(self):
        """Fetches closing prices from EOD Historical API using threading"""
        closing_df = fetch_price_parallel(self.tickers, self.client)
        closing_df.index = pd.to_datetime(closing_df.index)
        clean_data = get_cutoff_date(closing_df)
        closing_df = clean_data.loc[self.sdate:, self.tickers]
        closing_df.to_csv(self.univ_path + 'adj_close.csv')

    def fetch_moving_averages(self, ma_type='sma'):
        """Fetched Moving average indicator from EOD Historical API"""
        sma_dict = {}
        sma_dict2 = {}

        for s in self.tickers:
            try:
                sma_list = self.client.get_instrument_ta(s, function=ma_type, period=50)
                sma_df = pd.DataFrame(sma_list)
                sma_df.set_index('date', inplace=True)
                sma_df.rename(columns={ma_type: s}, inplace=True)
                sma_dict.update(sma_df)
            except Exception as e:
                print(f"Could not fetch data for {s}")

        for s in self.tickers:
            sma_list2 = self.client.get_instrument_ta(s, function=ma_type, period=20)
            sma_df2 = pd.DataFrame(sma_list2)
            sma_df2.set_index('date', inplace=True)
            sma_df2.rename(columns={ma_type: s}, inplace=True)
            sma_dict2.update(sma_df2)

        ta_sma_df = pd.DataFrame.from_dict(sma_dict)
        ta_sma_df = ta_sma_df[self.tickers]
        ta_sma_df.to_csv(self.univ_path + 'moving_average.csv')

        ta_sma_df2 = pd.DataFrame.from_dict(sma_dict2)
        ta_sma_df2 = ta_sma_df2[self.tickers]
        ta_sma_df2.to_csv(self.univ_path + 'moving_average2.csv')

        print(ta_sma_df)

    def benchmarks_closing_prices(self):
        """Fetches closing prices from EOD Historical API. BIL inception - 5/30/2007 - Replace by FRED Data"""
        # start = '1999-12-31'
        bm_dict = {}
        # bm_univ = ['SP500TR.INDX', 'NDX.INDX', 'VTHR.US', 'VBTLX.US', 'VTSAX.US', 'DGSIX.US', 'AW01.INDX', 'PHTNX.US',
        # 		   'PHTJX.US', 'PLTQX.US', 'PHTYX.US', 'PHTUX.US', 'PLTNX.US', 'PLTHX.US', 'PLHHX.US']
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
            # bm_df = bm_df.set_index('date')
            # bm_dict.update({s: bm_df.adjusted_close})

        # bm_px = pd.DataFrame.from_dict(bm_dict)
        # TODO: Temporary fix on 8/31/2023. Fix date issue
        bm_df.to_csv(self.univ_path + 'bm_raw_px.csv')
        # bm_px = pd.read_csv(self.univ_path + 'temp_bm.csv', index_col=[0], parse_dates=True)
        bm_px = bm_df.loc[self.sdate:, :]
        # bm_px = bm_px[bm_univ].bfill()
        cols = {'SP500TR.INDX': 'S&P 500 Total Return Index', 'NDX.INDX': 'NASDAQ 100 Index',
                'VTHR.US': 'Russell 3000 Index', 'VBTLX.US': 'Vanguard Total Bond Market',
                'VTSAX.US': 'Vanguard Total Stock Mkt', 'DGSIX.US': 'DFA Global All 60/40',
                'AW01.INDX': 'MSCI All World Index'}

        bm_px = bm_px.rename(columns=cols)
        bm_px.to_csv(self.univ_path + "bm_prices.csv")


class investmentModels:
    ''':param:
        sname: model name
        q = quintile threshold for asset filters
        maxwts = maximumm holdings weight for the optimizer'''
    
    
    def __init__(self, univs, sname, srcpath, destpath, q=0.5, window=90, adv_fees=1.0, benchmark='^GSPC'):

        self.sname = sname
        self.q = q
        self.out_path2 = destpath
        self.out_path = srcpath
        self.window = window
        # converting to monthly
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
        import tempfile
    
        # 3 factors
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, 'fama_french.zip')
            
                # Download the file
                urllib.request.urlretrieve(ff_url, zip_path)
            
                # Extract the file
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
            
                # Read the CSV
                ff_factors = pd.read_csv(
                    os.path.join(temp_dir, 'F-F_Research_Data_Factors.csv'), 
                    skiprows=3, 
                    index_col=0
                )

                # Process the data
                ff_factors = ff_factors[:-1]
                ff_factors = ff_factors.loc[:ff_factors.isnull().idxmax()[0]]
                ff_factors.dropna(inplace=True)
                ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m')
                ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()
                ff_factors = ff_factors.map(lambda x: float(x) / 100)
            
                # Save to output path
                ff_factors.to_csv(os.path.join(self.out_path, 'parsed_famafrench.csv'))
            
        except Exception as e:
            
            raise

    def ff_regression(self):
        ff_factor = pd.read_csv(self.out_path + 'parsed_famafrench.csv', index_col=[0], parse_dates=True)
        # ff_factor = pd.read_csv('parsed_famafrench.csv', index_col=[0], parse_dates=True)

        # read the nav file to calculate the backtest returns
        portfolio_nav = pd.read_csv(self.out_path + 'portfolio_nav.csv', index_col=[0], parse_dates=True)

        # calculate the nav pct change for returns time series
        portfolio_df = portfolio_nav.pct_change().fillna(0)

        # portfolio_df = pd.read_csv(self.out_path + 'backtest_returns.csv', index_col=[0], parse_dates=True)
        # Merging the data
        alpha, beta, rsq_adj, p_alpha = [], [], [], []
        for counter in range(len(portfolio_df.columns)):
            selected_port = portfolio_df.loc[:, portfolio_df.columns[counter]]
            all_data = pd.merge(pd.DataFrame(selected_port), ff_factor, how='inner', left_index=True, right_index=True)
            # Rename the columns
            all_data.rename(columns={"Mkt-RF": "mkt_excess"}, inplace=True)
            # Calculate the excess returns
            all_data['port_excess'] = all_data[portfolio_df.columns[counter]] - all_data['RF']
            # Added 6/2/2022, last 60 months data
            all_data = all_data.iloc[-60:]

            # 3 factor
            model = smf.formula.ols(formula="port_excess ~ mkt_excess + SMB + HML", data=all_data).fit()
            # 5 Factor
            # model = smf.formula.ols(formula="port_excess ~ mkt_excess + SMB + HML + RMW + CMA", data=all_data).fit()
            # print(model.params)

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
        """Selects assets based on their moving average and cutoff threshold"""
        # df_avg = pd.read_csv(self.out_path + 'moving_average.csv', index_col=[0], parse_dates=True)
        # df_avg2 = pd.read_csv(self.out_path + 'moving_average2.csv', index_col=[0], parse_dates=True)
        df_close = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)

        # delete it
        # df_close = df_close.loc['2019-12-31':]
        df_close = df_close.loc['2008-01-31':]

        # read benchmark prices file
        bm_prices = pd.read_csv('benchmark.csv', index_col=[0], parse_dates=True)

        # select respective benchmark
        bm_select = bm_prices[self.benchmark]

        def calculate_rolling_returns(data, period):
            return ((data.iloc[-1] / data.iloc[0]) ** (250 / period)) - 1

        # create a dataframe with benchmark included for rolling returns
        df_combined_px = df_close.copy()
        df_combined_px['bm'] = bm_select

        # calculate rolling annualized returns
        df_rolling_ret = df_combined_px.rolling(window=self.window).apply(calculate_rolling_returns,
                                                                          args=(self.window,))

        # Relative Ranking. filter outperforming assets
        df_delta = df_rolling_ret.sub(df_rolling_ret.bm, axis=0)

        # Boolean dataframe for positive delta
        positive_delta = df_delta.gt(0)
        positive_relative_returns = df_delta[positive_delta]
        # positive_relative_returns = df_rolling_ret[df_rolling_ret.sub(df_rolling_ret.bm, axis=0) > 0]
        filter = positive_relative_returns.quantile(self.q, axis=1, numeric_only=True)

        # Dataframe of difference of positive performance rel to bm and quantile
        delta_filter = positive_relative_returns.sub(filter, axis=0)

        # Boolean dataframe for elements greater than quantile delta
        positive_filter_df = delta_filter.gt(0)

        # Dataframe for over the quantile filter. Final holdings
        filtered_pos_rel_return = delta_filter[positive_filter_df]

        # filtered_pos_rel_return = positive_relative_returns[positive_relative_returns.gt(filter, axis=0)]
        filtered_pos_rel_return_monthly = filtered_pos_rel_return.resample('ME', closed='right').last()

        # save the holdings
        filtered_pos_rel_return_monthly.to_csv(self.out_path + f'{self.sname}_ew_holdings.csv')

        # plot holdings counts
        # df = filtered_pos_rel_return_monthly.copy()
        # df['counts'] = df.count(axis=1)
        # df['counts'].hist(bins=20)
        # plt.show()

        df_return = df_close.resample('ME', closed='right').last()
        df_return = df_return.pct_change().fillna(0).shift(-1)
        portfolio_ret = df_return[filtered_pos_rel_return_monthly.notnull()].mean(axis=1).shift(1).fillna(0)

        # -------------- Alternative portfolio using inverse of weights ----------------------------#
        # calculate inverse vol weight
        inv_vol = self.calculate_inverse_vol(df_close)
        inv_vol = inv_vol.resample('ME', closed='right').last()
        selected_asset = inv_vol[filtered_pos_rel_return_monthly.notnull()]
        inverse_weights = selected_asset.div(selected_asset.sum(axis=1), axis=0)
        inverse_wt_port = df_return.mul(inverse_weights, axis=1).sum(axis=1).shift(1).fillna(0)
        inverse_weights.to_csv(self.out_path + f'{self.sname}_inverse_holdings.csv')
        # check if all weights adds to 1
        wts_check = inverse_weights.sum(axis=1)
        # -------------- Alternative portfolio using inverse of weights ---------------------------- #

        # ---------------Alternative weights using delta performance_____________________________ #
        def clean_wts(data): return data.dropna().apply(lambda x: x / data.sum())
        delta_wts = filtered_pos_rel_return.apply(clean_wts, axis=1)
        # delta_wts = filtered_pos_rel_return.apply(lambda x: x / filtered_pos_rel_return.sum(axis=1))
        delta_wts = delta_wts.resample('ME', closed='right').last()
        # save alpha weights
        delta_wts.to_csv(self.out_path + f'{self.sname}_alpha_holdings.csv')
        alpha_wt_port = df_return.mul(delta_wts, axis=1).sum(axis=1).shift(1).fillna(0)
        # ----------------------------------------------------------------------------------------- #

        # ---------------Alternative weights using Zscore _________________________________________ #

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
        # ----------------------------------------------------------------------------------------- #

        # Add underlying benchmark time series for comparison
        bm_ret = bm_prices.loc[portfolio_ret.index[0]:, self.benchmark]
        # bm_ret.resample('ME', closed='right').last().pct_change()

        # combined the dataframes
        combined_ret_df = pd.DataFrame(portfolio_ret, columns=['portfolio'])

        combined_ret_df.loc[:, 'inv_port'] = inverse_wt_port
        combined_ret_df.loc[:, 'alpha_port'] = alpha_wt_port
        combined_ret_df.loc[:, 'zscore_port'] = zscore_wt_port
        combined_ret_df.loc[:, 'bm'] = bm_ret.resample('ME', closed='right').last().pct_change()

        # Do the same for lower quantile to compare for true alpha
        filter = positive_relative_returns.quantile(.2, axis=1, numeric_only=True)
        filtered_pos_rel_return = positive_relative_returns[positive_relative_returns.lt(filter, axis=0)]
        filtered_pos_rel_return_monthly = filtered_pos_rel_return.resample('ME', closed='right').last()
        filtered_pos_rel_return_monthly.to_csv(self.out_path + f'{self.sname}_lower_quintile_holdings.csv')

        # combine all dataframes
        pr2 = df_return[filtered_pos_rel_return_monthly.notnull()].mean(axis=1).shift(1).fillna(0)
        combined_ret_df.loc[:, 'portfolio_lqunitile'] = pr2
        combined_ret_df = combined_ret_df.map(lambda x: x - self.fees)

        # clean the data and select the required timeseries
        combined_ret_df.iloc[0] = 0.0
        combined_ret_df = 100 * combined_ret_df.add(1).cumprod()
        # remove last data point if ran intra month to capture as of last month end data
        combined_ret_df = combined_ret_df[['portfolio', 'portfolio_lqunitile', 'inv_port', 'alpha_port',
                                           'zscore_port']].iloc[:-1]

        # add all other benchmarks to the nav dataframe
        bm_ret = bm_prices.loc[combined_ret_df.index[0]:combined_ret_df.index[-1]]
        bm_ret = bm_ret.resample('ME', closed='right').last().pct_change().fillna(0)
        bm_ret = 100 * bm_ret.add(1).cumprod()

        # combined all dataframe in one
        combined_ret_df = pd.concat([combined_ret_df, bm_ret], axis=1, sort=False)

        # save the NAV file
        combined_ret_df.to_csv(self.out_path + 'portfolio_nav.csv')

    def generate_optimized_monthly_weights(self, counts):
        df_sma = pd.read_csv(self.out_path + 'monthly_holdings.csv', index_col=[0], parse_dates=True)
        df_close = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        # Adjust close to match universe tickers
        df_close = df_close.loc[:, df_sma.columns]
        # ---------------
        df_close = df_close.loc[self.sdate:]
        # Resample daily price data to monthly
        monthly_price = df_close.resample('M', closed='right').last()
        # Run optimization as of last month closing price
        monthly_price = monthly_price.loc[df_sma.index[0]: df_sma.index[-1], :]
        uncon_mvo_opt_wts = {}
        uncon_hrp_opts_wts = {}

        # Loop through 12 months and generate MVO wts
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
            # New addition on 2/18
            umvo_wts_df = pd.DataFrame(umvo_wts_df, columns=df_close.columns)

            # umvo_wts_df = umvo_wts_df[df_close.columns]

            # 100% cash to no optimization solutions where all columns are null - Force cash
            wts_cash = umvo_wts_df[umvo_wts_df.isnull().sum(axis=1) == len(umvo_wts_df.columns)]
            umvo_wts_df.loc[wts_cash.index, ['SHY']] = 1.0
            # End
            umvo_wts_df.to_csv(self.out_path2 + '{}_unconstrained_mvo_wts.csv'.format(self.sname))
            umvo_wts_df.to_csv(self.out_path + '{}_unconstrained_mvo_wts.csv'.format(self.sname))
        # Replace None with np.nan in the dataframe
        # umvo_wts_df = umvo_wts_df.replace(to_replace=[None], value=np.nan, inplace=True)
        # temp = umvo_wts_df.round(2)
        # delta = 1 - temp.sum(axis=1)
        # # mins = temp.replace(0, np.nan).idxmin(axis=1)
        # # maxs = temp.replace(0, np.nan).idxmax(axis=1)
        # # minarr = np.array(mins)
        # # maxarr = np.array(maxs)
        # darr = np.array(delta)
        # darr = np.round(darr, 2)
        # absdarr = abs(100*darr).astype(int)
        # temp2 = pd.DataFrame(index=temp.index, columns=temp.columns)
        # for c in range(len(temp)):
        # 	if darr[c] > 0:
        # 		tt = temp.loc[temp.index[c]].sort_values(ascending=True)
        # 		tt.iloc[:absdarr[c]] = tt.iloc[:absdarr[0]] + .01
        # 		temp2.iloc[c, :] = tt
        #
        # 		# temp.loc[temp.index[c], minarr[c]] = temp.loc[temp.index[c], minarr[c]] + darr[c]
        # 	elif darr[c] < 0:
        # 		tt = temp.loc[temp.index[c]].sort_values(ascending=False)
        # 		tt.iloc[:absdarr[c]] = tt.iloc[:absdarr[0]] - .01
        # 		temp2.iloc[c, :] = tt
        # 		# temp.loc[temp.index[c], maxarr[c]] = temp.loc[temp.index[c], maxarr[c]] + darr[c]
        # 	else:
        #
        # 		temp2.iloc[c, :] = temp.iloc[c, :]
        #
        # print('hold')

        # to create bbg_upload and Ycharts upload file
        # df_rounded = df.div(df.sum(axis=1), axis=0).round(2)
        # temp = umvo_wts_df.copy()
        # temp.fillna(0, inplace=True)
        # t = temp.keys()[np.argmax(temp.values != 0, axis=1)]
        # temp.loc[temp.index[0], t[0]] + (1 - temp.sum(axis=1))
        # tt = umvo_wts_df.stack().to_frame()
        # tt = tt.reset_index()
        else:
            unc_hrp_wts_df = pd.DataFrame.from_dict(uncon_hrp_opts_wts).T
            unc_hrp_wts_df = unc_hrp_wts_df[df_close.columns]
            # 100% cash to no optimization solutions - Force cash
            wts_cash = unc_hrp_wts_df[unc_hrp_wts_df.isnull().sum(axis=1) == len(unc_hrp_wts_df.columns)]
            unc_hrp_wts_df.loc[wts_cash.index, ['SHY']] = 1.0
            # End
            unc_hrp_wts_df.to_csv(self.out_path + '{}_unconstrained_hrp_wts.csv'.format(self.sname))

    def portfolio_backtest(self, adv_fees=1.0):

        global wts_df
        # Asset prices
        price_df = pd.read_csv(self.out_path + 'adj_close.csv', index_col=[0], parse_dates=True)
        resampled_px = price_df.resample('ME', closed='right').last()

        # -------Read Benchmark Prices and create custom benchmarks--------
        bm_prices = pd.read_csv(self.out_path + 'bm_prices.csv', index_col=[0], parse_dates=True)
        resampled_bm = bm_prices.resample('ME', closed='right').last()
        resampled_bm['BMUS6040'] = 0.6 * resampled_bm['S&P 500 Total Return Index'] + 0.4 * resampled_bm[
            'Vanguard Total Bond Market']
        # resampled_bm['bm_slcap'] = 0.4 * resampled_bm['S&P 500 Total Return Index'] + 0.3 * resampled_bm[
        # 	'Vanguard Total Bond Market'] + 0.3 * resampled_bm['MSCI All World Index']
        # resampled_bm['bm_sgaf'] = 0.4 * resampled_bm['NASDAQ 100 Index'] + 0.3 * resampled_bm[
        # 	'Vanguard Total Bond Market'] + 0.3 * resampled_bm['MSCI All World Index']
        # resampled_bm['bm_seof'] = resampled_bm['Vanguard Total Stock Mkt'].copy()
        # resampled_bm['bm_gtaa'] = 0.4 * resampled_bm['Vanguard Total Bond Market'] \
        # 						  + 0.6 * resampled_bm['MSCI All World Index']
        # resampled_bm['bm_sector'] = resampled_bm['S&P 500 Total Return Index'].copy()
        # resampled_bm['bm_gbf'] = 0.5 * resampled_bm['S&P 500 Total Return Index'] + 0.25 * resampled_bm[
        # 	'Vanguard Total Bond Market'] + 0.25 * resampled_bm['MSCI All World Index']
        # resampled_bm['bm_lcg'] = resampled_bm['S&P 500 Total Return Index'].copy()
        # resampled_bm['bm_tfg'] = resampled_bm['NASDAQ 100 Index'].copy()
        # resampled_bm['bm_bgo'] = 0.7 * resampled_bm['NASDAQ 100 Index'] + 0.3 * resampled_bm['Vanguard Total Stock Mkt']

        # ------------------------BM Block Ends---------------------------------------------
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
        # Equal Weight  Universe Portfolio
        returns_df['eq_wt'] = resampled_px.pct_change().fillna(0).mean(axis=1)

        # Deducting annual mgmt fees monthly
        monthly_fees = (0.01 * adv_fees) / 12
        returns_df = returns_df.map(lambda x: x - monthly_fees)
        # No Fees for the starting month. Assign 0 to all returns
        returns_df.iloc[0] = 0.0
        returns_df = pd.merge(returns_df, resampled_bm.pct_change().fillna(0), left_index=True, right_index=True)
        returns_df.iloc[0] = 0.0
        portfolio_nav = 100 * returns_df.add(1).cumprod()
        portfolio_nav.to_csv(self.out_path + 'portfolio_nav.csv')
        returns_df.to_csv(self.out_path + 'backtest_returns.csv')
        maxwts = wts_df.max(axis=1)
        print(maxwts.sort_values(ascending=False))
        print(wts_df.min(axis=1).sort_values(ascending=False))
        print('*' * 100)
        print(wts_df.iloc[-5:].fillna(0).T)

    def portfolio_analytics(self):
        # read portfolio Net Values, benchmark and cash daily values
        read_navs = pd.read_csv(self.out_path + "portfolio_nav.csv", index_col=[0], parse_dates=True)
        # read_navs = 100 * read_navs.loc['2019-04-30':].pct_change().fillna(0).add(1).cumprod()

        # Benchmark prices
        # bm_px = pd.read_csv('data/rfr.csv', index_col=[0], parse_dates=True)
        bm_px = pd.read_csv(self.out_path + 'rfr.csv', index_col=[0], parse_dates=True)
        bm_px = bm_px.resample('ME', closed='right').last()
        bm_px = bm_px.loc[read_navs.index[0]: read_navs.index[-1], :]
        frame2 = read_navs.copy()

        try:
            # YTD
            stats_df = pd.DataFrame(columns=frame2.columns)
            yearly = frame2.copy()
            if pd.Series(frame2.index[0]).dt.is_year_end[0]:
                yearly = frame2.resample('YE', closed='right').last().pct_change()
            else:
                dummy_yr = frame2.index[0] + offsets.YearEnd(-1)
                yearly.loc[dummy_yr, :] = yearly.iloc[0]
                yearly.sort_index(inplace=True)
                yearly = yearly.resample('YE', closed='right').last().pct_change()

            # ------------Remove the current partial month to calculate N period statistics----------
            # portfolio_nav = frame2.copy()
            # Remove the current month performance for ignore partial month performance
            # if pd.to_datetime(self.end_date).is_month_end:
            # 	portfolio_nav = portfolio_nav
            # else:
            portfolio_nav = frame2.copy()

            # 1 Year
            N = 12
            rfr_ts = bm_px.iloc[-(N + 1):]
            rfr = rfr_ts.mean()
            # rfr = 1 + (rfr_ts.pct_change().dropna())
            # rfr = rfr.cumprod().iloc[-1] ** (12 / N) - 1
            one_yr = portfolio_nav.iloc[-(N + 1):]
            r1 = 1 + (one_yr.pct_change().dropna())
            r1 = r1.cumprod().iloc[-1] ** (12 / N) - 1
            risk1 = one_yr.iloc[1:].pct_change().std() * np.sqrt(12)
            sharpe1 = r1.apply(lambda x: (x - rfr))
            sharpe1 = sharpe1.divide(risk1, axis=0)
            f1 = False

            # 3 Year
            N = 36
            if len(portfolio_nav) < N:
                f3 = True

            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                # rfr = 1 + (rfr_ts.pct_change().dropna())
                # rfr = rfr.cumprod().iloc[-1] ** (12 / N) - 1
                three_yr = portfolio_nav.iloc[-(N + 1):]
                r3 = 1 + (three_yr.pct_change().dropna())
                r3 = r3.cumprod().iloc[-1] ** (12 / N) - 1
                risk3 = three_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe3 = r3.apply(lambda x: (x - rfr))
                sharpe3 = sharpe3.divide(risk3, axis=0)
                f3 = False

            # 5 Year
            N = 60
            if len(portfolio_nav) < N:
                f5 = True
            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                # rfr = 1 + (rfr_ts.pct_change().dropna())
                # rfr = rfr.cumprod().iloc[-1] ** (12 / N) - 1
                five_yr = portfolio_nav.iloc[-(N + 1):]
                r5 = 1 + (five_yr.pct_change().dropna())
                r5 = r5.cumprod().iloc[-1] ** (12 / N) - 1
                risk5 = five_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe5 = r5.apply(lambda x: (x - rfr))
                sharpe5 = sharpe5.divide(risk5, axis=0)
                f5 = False

            # 10 Year
            N = 120
            if len(portfolio_nav) < N:
                f10 = True
            else:
                rfr_ts = bm_px.iloc[-(N + 1):]
                rfr = rfr_ts.mean()
                # rfr = 1 + (rfr_ts.pct_change().dropna())
                # rfr = rfr.cumprod().iloc[-1] ** (12 / N) - 1
                ten_yr = portfolio_nav.iloc[-(N + 1):]
                r10 = 1 + (ten_yr.pct_change().dropna())
                r10 = r10.cumprod().iloc[-1] ** (12 / N) - 1
                risk10 = ten_yr.iloc[1:].pct_change().std() * np.sqrt(12)
                sharpe10 = r10.apply(lambda x: (x - rfr))
                sharpe10 = sharpe10.divide(risk10, axis=0)
                f10 = False

            # Inception
            # RFR Inception
            N = len(frame2) - 1
            # rfr_ts = bm_px.iloc[-(N + 1):]
            rfr = bm_px.mean()
            # rfr = 1 + (bm_px.pct_change().dropna())
            # rfr = rfr.cumprod().iloc[-1] ** (12 / N) - 1

            ri = 1 + (frame2.pct_change().dropna())
            ri = ri.cumprod().iloc[-1] ** (12 / N) - 1
            riski = frame2.pct_change().std() * np.sqrt(12)
            sharpei = ri.apply(lambda x: (x - rfr))
            sharpei = sharpei.divide(riski, axis=0)

            stats_df.loc['YTD', :] = yearly.loc[yearly.index[-1], :].values
            stats_df.loc['YTD', :] = stats_df.loc['YTD', :].apply(lambda x: round(x, 4))

            # formatting to 4 decimal places and adding to the dataframe
            if f1:
                stats_df.loc['Annualized Return (% 1Y)', :] = 0
                stats_df.loc['Annualized Risk (% 1Y)', :] = 0
                stats_df.loc['Sharpe Ratio (1Y)', :] = 0
            else:
                stats_df.loc['Annualized Return (% 1Y)', :] = r1.apply(lambda x: round(x, 4)).values
                stats_df.loc['Annualized Risk (% 1Y)', :] = risk1.apply(lambda x: round(x, 4)).values
                stats_df.loc['Sharpe Ratio (1Y)', :] = sharpe1.apply(lambda x: round(x, 4)).values.flatten()

            if f3:
                stats_df.loc['Annualized Return (% 3Y)', :] = 0
                stats_df.loc['Annualized Risk (% 3Y)', :] = 0
                stats_df.loc['Sharpe Ratio (3Y)', :] = 0
            else:
                stats_df.loc['Annualized Return (% 3Y)', :] = r3.apply(lambda x: round(x, 4)).values
                stats_df.loc['Annualized Risk (% 3Y)', :] = risk3.apply(lambda x: round(x, 4)).values
                stats_df.loc['Sharpe Ratio (3Y)', :] = sharpe3.apply(lambda x: round(x, 4)).values.flatten()

            if f5:
                stats_df.loc['Annualized Return (% 5Y)', :] = 0
                stats_df.loc['Annualized Risk (% 5Y)', :] = 0
                stats_df.loc['Sharpe Ratio (5Y)', :] = 0
            else:
                stats_df.loc['Annualized Return (% 5Y)', :] = r5.apply(lambda x: round(x, 4)).values
                stats_df.loc['Annualized Risk (% 5Y)', :] = risk5.apply(lambda x: round(x, 4)).values
                stats_df.loc['Sharpe Ratio (5Y)', :] = sharpe5.apply(lambda x: round(x, 4)).values.flatten()

            if f10:
                stats_df.loc['Annualized Return (% 10Y)', :] = 0
                stats_df.loc['Annualized Risk (% 10Y)', :] = 0
                stats_df.loc['Sharpe Ratio (10Y)', :] = 0
            else:
                stats_df.loc['Annualized Return (% 10Y)', :] = r10.apply(lambda x: round(x, 4)).values
                stats_df.loc['Annualized Risk (% 10Y)', :] = risk10.apply(lambda x: round(x, 4)).values
                stats_df.loc['Sharpe Ratio (10Y)', :] = sharpe10.apply(lambda x: round(x, 4)).values.flatten()

            stats_df.loc['Annualized Return (% inception)', :] = ri.apply(lambda x: round(x, 4)).values
            stats_df.loc['Annualized Risk (% inception)', :] = riski.apply(lambda x: round(x, 4)).values
            stats_df.loc['Sharpe Ratio (inception)', :] = sharpei.apply(lambda x: round(x, 4)).values.flatten()

            stats_df.loc['% CAGR', :] = round((frame2.iloc[-1] / frame2.iloc[0] - 1), 4)
            stats_df.loc['$ Growth', :] = round(frame2.iloc[-1], 4)

            # --------------------Drawdown Calculations----------------------------------
            # Calculate the max value of returns based on rolling 365 days of returns
            roll_max = portfolio_nav.rolling(min_periods=1, window=12).max()

            # Calculate daily draw-down for rolling max
            rolling_drawdown = portfolio_nav / roll_max - 1.0

            # max Drawdown based on daily data
            max_dd = rolling_drawdown.min()

            # -----------Determine the start and end date of the drawdowns------------------
            # ----------------------------Current Portfolio---------------------------
            asset_1 = portfolio_nav.loc[:, portfolio_nav.columns[0]]
            dd_end1 = np.argmax(np.maximum.accumulate(asset_1) - asset_1)
            dd_start1 = np.argmax(asset_1[:dd_end1])
            # sdate1 = rebased_nav.index[dd_start1]
            # endate1 = rebased_nav.index[dd_end1]

            # -------------------------------Proposed Portfolio-------------------------
            # asset_2 = frame2.loc[:, frame2.columns[1]]
            asset_2 = portfolio_nav.loc[:, portfolio_nav.columns[1]]
            dd_end2 = np.argmax(np.maximum.accumulate(asset_2) - asset_2)
            dd_start2 = np.argmax(asset_2[:dd_end2])
            # sdate2 = rebased_nav.index[dd_start2]
            # endate2 = rebased_nav.index[dd_end2]

            # Calculate the min and max yearly returns for the portfolios
            worst_ret = yearly.min().values

            # AvgDD based on rolling 3645 days return
            stats_df.loc['% AvgDD', :] = abs(round(rolling_drawdown.mean(), 2)).values
            stats_df.loc['% MaxDD', :] = abs(round(max_dd, 2)).values
            stats_df.loc['Skew', :] = round(frame2.pct_change().skew(), 2).values
            stats_df.loc['Kurtosis', :] = round(frame2.pct_change().kurtosis(), 2).values
            stats_df.loc['min_ret', :] = yearly.min().apply(lambda x: round(x, 4)).values
            stats_df.loc['max_ret', :] = yearly.max().apply(lambda x: round(x, 4)).values

            # Get test statics from Fama French
            alpha, beta, rsq_adj, p_alpha = self.ff_regression()
            stats_df.loc['alpha', :] = alpha
            stats_df.loc['beta', :] = beta
            stats_df.loc['rsq_adj', :] = rsq_adj
            stats_df.loc['p-value(alpha)', :] = p_alpha

            # parse yearly data and add inception and YTD number to it. Convert index to be yearly only.
            yearly.dropna(inplace=True)
            yearly.index = pd.to_datetime(yearly.index).strftime('%Y')
            yearly.loc['YTD', :] = stats_df.loc['YTD', :].values
            yearly.loc['since_inception', :] = stats_df.loc['Annualized Return (% inception)', :].values

            # Calcualtion of S&P 500 Sharpe ratio
            # sp500 = bbg_indices.resample('M', closed='right').last()
            # sp500_monthly = sp500.loc[monthly_nav.index[0]: monthly_nav.index[-1], ['SPXT Index']]
            # total_ret = sp500_monthly.iloc[-1] / sp500_monthly.iloc[0]
            # nperiod = 12 / len(sp500_monthly)
            # # total_ret = tot_ret
            # ann_ret = total_ret ** nperiod - 1
            # ann_risk = sp500_monthly.pct_change().std() * np.sqrt(12)
            # sp500_sharpe = (ann_ret[0] - rfr[-1]) / ann_risk[0]
            # stats_df.loc['sp_500_sharpe', :] = round(sp500_sharpe[0], 4)

            # -------------------Market Crashes----------------------
            # cnames = self.crashes.index.tolist()
            # sorted_cols = ['SP500', 'current', 'proposed']
            # crash_df = pd.DataFrame(index=cnames, columns=sorted_cols)
            # temp_sp500 = bbg_indices.loc[read_navs.index[0]: read_navs.index[-1], ['SPXT Index']]
            #
            # for d in range(len(self.crashes)):
            # 	flag1 = self.crashes.loc[self.crashes.index[d], 'start'] in read_navs.index
            # 	flag2 = self.crashes.loc[self.crashes.index[d], 'end'] in read_navs.index
            # 	if flag1 and flag2:
            # 		temp_df = read_navs.copy()
            # 		# temp_df.loc[:, 'SP500'] = temp_sp500
            # 		temp_df.ffill(inplace=True)
            # 		sd1 = self.crashes.loc[cnames[d], 'start']
            # 		ed1 = self.crashes.loc[cnames[d], 'end']
            # 		temp_df = temp_df.loc[sd1:ed1, :]
            # 		temp_df = temp_df[sorted_cols]
            # 		per_ret = temp_df.iloc[-1] / temp_df.iloc[0] - 1
            # 		crash_df.loc[cnames[d], :] = per_ret.values
            # 	else:
            # 		crash_df.loc[cnames[d], :] = np.nan
            #
            # # adding since inception returns to the dataframe
            # # crash_df.loc['Annualized Portfolio', 'SP500'] = ann_ret[0]
            # crash_df.loc['Annualized Portfolio', ['current', 'proposed']] = yearly.loc['since_inception', :].values
            # crash_df.dropna(axis=0, inplace=True)

            writer = pd.ExcelWriter(self.out_path + '{}_analysis_output.xlsx'.format(self.sname),
                                    engine='xlsxwriter')

            # Portfolio Statistics
            stats_df.to_excel(writer, sheet_name="port_statistics")

            # Daily time series
            portfolio_nav.index = portfolio_nav.index.strftime('%m/%d/%Y')
            

            # Monthly time series
            # monthly_nav.index = monthly_nav.index.strftime('%m/%d/%Y')
            # monthly_nav.to_excel(writer, sheet_name='time_series')
            # Edited to capture the user selected end - 03/09/2022
            frame2.index = frame2.index.strftime('%m/%d/%Y')
            frame2.to_excel(writer, sheet_name='time_series')

            # yearly time series
            yearly.to_excel(writer, sheet_name="yearly")
            # TODO: Rolling N Period annualized returns charts
            
            
            # generate introduction slide
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
        
            # Skip if holdings file doesn't exist
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
        # to create bbg_upload and Ycharts upload file
        # read optimized weights file
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
        # df_rounded = df.div(df.sum(axis=1), axis=0).round(2)
        # temp = df.copy()
        df.fillna(0, inplace=True)
        df = round(df.div(df.sum(axis=1), axis=0), 2)
        # df.iloc[1].apply(calibrate_data)
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
