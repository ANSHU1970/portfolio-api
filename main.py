import os
import logging
import traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from datetime import datetime
from func import get_only_cutoff_date, investmentUniverse, investmentModels, fred_data,process_custom_model_with_tickers
import uvicorn
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("investment_api.log")
    ]
)
logger = logging.getLogger("investment-api")

app = FastAPI(title="Investment Model Processing API")

PROCESSING_STATUS = {
    "is_processing": False,
    "last_run": None,
    "processed_models": [],
    "error": None
}
def get_available_models() -> Dict:
    universe_file = os.path.join(os.path.dirname(__file__), "universe.xlsx")
    if not os.path.exists(universe_file):
        raise FileNotFoundError("universe.xlsx file not found")
    univ_df = pd.read_excel(universe_file, index_col=[0], sheet_name=None)
    params = univ_df['model_params']
    return {
        "univ_df": univ_df,
        "params": params,
        "model_names": params.index.tolist()
    }


class TickerWeight(BaseModel):
    ticker: str
    weight: float

class CustomModelRequest(BaseModel):
    model: str
    tickers_and_weights: List[TickerWeight]

@app.post("/custom-model")
async def run_custom_model_with_user_tickers(
    request: CustomModelRequest
):
    try:
        model = request.model
        tickers_and_weights = request.tickers_and_weights
        tickers = [tw.ticker for tw in tickers_and_weights]
        weights = [tw.weight for tw in tickers_and_weights]

        if not np.isclose(sum(weights), 1.0):
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        all_data = get_available_models()
        model_names = all_data["model_names"]
        if model not in model_names:
            raise HTTPException(status_code=400, detail=f"Invalid model name. Choose from: {model_names}")

        cutoff_date = get_only_cutoff_date()

        result = process_custom_model_with_tickers(
            model_name=model,
            tickers=tickers,
            weights=weights,  
            all_params=all_data["params"],
            cutoff_date=cutoff_date,
            univ_df=all_data["univ_df"]
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "message": "Model processed with user tickers and weights",
            "model": model,
            "tickers": result["tickers"],
            "output_file": result["output_file"]
        }

    except Exception as e:
        logger.error(f"Custom model error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/process-models")
async def process_models(
    models: List[str] = Query(None, description="List of specific models to process. If empty, all models will be processed")
):
    global PROCESSING_STATUS
    if PROCESSING_STATUS["is_processing"]:
        raise HTTPException(
            status_code=400,
            detail="Processing already in progress"
        )
    try:
        model_data = get_available_models()
        univ_df = model_data["univ_df"]
        params = model_data["params"]
        all_model_names = model_data["model_names"]
        if models:
            invalid_models = [m for m in models if m not in all_model_names]
            if invalid_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid models requested: {invalid_models}. Available models: {all_model_names}"
                )
            model_names = models
        else:
            model_names = all_model_names
        PROCESSING_STATUS = {
            "is_processing": True,
            "start_time": datetime.now(),
            "processed_models": [],
            "error": None,
            "requested_models": model_names
        }
        logger.info(f"Starting processing for models: {', '.join(model_names)}")
        file_name = get_only_cutoff_date()
        dest_dir = os.path.join("./", str(file_name))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            logger.info(f"Created output directory: {dest_dir}")
        processed_models = []
        for i, mods in enumerate(model_names):
            try:
                logger.info(f"Processing model {i+1}/{len(model_names)}: {mods}")
                sh_mods = univ_df[mods]
                sh_mods.index.name = 'date'
                tickers = [ticker.replace('/', '-') for ticker in sh_mods.index]
                q_thres = params.loc[mods, 'thres_q']
                rolling_window = params.loc[mods, 'days']
                wt_scheme = [params.loc[mods, 'wt_scheme']]
                benchmark = params.loc[mods, 'benchmark']
                adv_fees = 1.65 if mods in ['large', 'tech', 'small', 'wealthx'] else 1.0
                univcls = investmentUniverse(mods, tickers, "./", sh_mods)
                univcls.fetch_moving_averages()
                univcls.fetch_closing_prices()
                if i == 0:
                    univcls.benchmarks_closing_prices()
                clss = investmentModels(
                    sh_mods, mods, "./", "./", q_thres, rolling_window, adv_fees, benchmark
                )
                if i == 0:
                    clss.fetch_famafrench_factors()
                    fred_data()
                clss.filter_assets_based_on_moving_averages()
                stats_ser = clss.portfolio_analytics()
                stats_ser.to_csv(os.path.join(dest_dir, f"{mods}.csv"))
                processed_models.append(mods)
                if i == len(model_names) - 1:
                    if processed_models:
                        clss.rebalance_trades(model_data["params"], dest_dir, file_name)
            except Exception as e:
                logger.error(f"Error processing model {mods}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        duration = datetime.now() - PROCESSING_STATUS["start_time"]
        PROCESSING_STATUS = {
            "is_processing": False,
            "last_run": datetime.now(),
            "processed_models": processed_models,
            "duration_seconds": duration.total_seconds(),
            "error": None,
            "requested_models": model_names
        }
        logger.info(f"Processing completed. Duration: {duration.total_seconds():.2f} seconds")
        return {
            "status": "success",
            "requested_models": model_names,
            "processed_models": processed_models,
            "skipped_models": list(set(model_names) - set(processed_models)),
            "duration_seconds": duration.total_seconds(),
            "output_directory": dest_dir
        }
    except HTTPException:
        raise
    except Exception as e:
        PROCESSING_STATUS = {
            "is_processing": False,
            "last_run": datetime.now(),
            "processed_models": [],
            "error": str(e)
        }
        logger.error(f"Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/get-analysis/{model_name}")
async def get_analysis_output(model_name: str):
    try:
        root_file = f"{model_name}_analysis_output.xlsx"
        if os.path.exists(root_file):
            return await _read_analysis_file(root_file)
        date_dirs = [d for d in os.listdir("./")
                   if os.path.isdir(d) and d.replace("-", "").isdigit()]
        for dir_name in sorted(date_dirs, reverse=True):
            file_path = os.path.join(dir_name, root_file)
            if os.path.exists(file_path):
                return await _read_analysis_file(file_path)
        raise HTTPException(
            status_code=404,
            detail=f"Analysis file not found in root or date directories for model: {model_name}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
async def _read_analysis_file(file_path: str):
    xls = pd.ExcelFile(file_path)
    sheet_data = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.replace([np.inf, -np.inf], np.nan).fillna("null")
        sheet_data[sheet_name] = df.set_index(df.columns[0]).to_dict(orient='index')
        
    return JSONResponse(content=sheet_data)


@app.get("/get-holdings/{model_name}/{holding_type}")
async def get_holdings_file(model_name: str, holding_type: str):
    """
    Retrieve holdings CSV file (e.g., ew, zscore, alpha, inverse, lower_quintile) as JSON.
    """
    try:
        filename = f"{model_name}_{holding_type}_holdings.csv"

        # Check in root first
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
        else:
            # Search in dated directories
            date_dirs = [d for d in os.listdir("./") if os.path.isdir(d) and d.replace("-", "").isdigit()]
            for dir_name in sorted(date_dirs, reverse=True):
                file_path = os.path.join(dir_name, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    break
            else:
                raise HTTPException(status_code=404, detail=f"{filename} not found")

        df = df.replace([np.inf, -np.inf], np.nan).fillna("null")
        return JSONResponse(content=df.to_dict(orient='index'))

    except Exception as e:
        logger.error(f"Error loading {model_name} {holding_type} holdings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-full-report/{model_name}")
async def get_full_report(model_name: str):
    try:
        # Define model-to-holdings mapping
        model_weight_map = {
            "us-fixed": "zscore",
            "developed": "inverse",
            "emerging": "lower_quintile",
            "gtaa": "ew",
            "us-sector": "zscore",
            "us-equities": "zscore",
            "large": "zscore",
            "tech": "zscore",
            "small": "zscore"
        }

        if model_name not in model_weight_map:
            raise HTTPException(status_code=400, detail=f"No holding type defined for model: {model_name}")

        holding_type = model_weight_map[model_name]
        analysis_filename = f"{model_name}_analysis_output.xlsx"
        holdings_filename = f"{model_name}_{holding_type}_holdings.csv"

        # Search for files in root or date-stamped directories
        date_dirs = [d for d in os.listdir("./") if os.path.isdir(d) and d.replace("-", "").isdigit()]
        search_dirs = ["./"] + [os.path.join("./", d) for d in sorted(date_dirs, reverse=True)]

        analysis_path = None
        holdings_path = None

        for directory in search_dirs:
            a_path = os.path.join(directory, analysis_filename)
            h_path = os.path.join(directory, holdings_filename)
            if os.path.exists(a_path):
                analysis_path = a_path
            if os.path.exists(h_path):
                holdings_path = h_path
            if analysis_path and holdings_path:
                break

        if not analysis_path:
            raise HTTPException(status_code=404, detail=f"Analysis file not found for {model_name}")
        if not holdings_path:
            raise HTTPException(status_code=404, detail=f"Holdings file not found for {model_name} with scheme {holding_type}")

        # Read analysis Excel
        xls = pd.ExcelFile(analysis_path)
        analysis_data = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.replace([np.inf, -np.inf], np.nan).fillna("null")
            analysis_data[sheet] = df.set_index(df.columns[0]).to_dict(orient="index")

        # Read holdings CSV
        holdings_df = pd.read_csv(holdings_path, index_col=0)
        holdings_df = holdings_df.replace([np.inf, -np.inf], np.nan).fillna("null")
        holdings_data = holdings_df.to_dict(orient="index")

        # Combine response
        return {
            "model": model_name,
            "holding_type": holding_type,
            "analysis_output": analysis_data,
            "holdings": holdings_data
        }

    except Exception as e:
        logger.error(f"Error in get_full_report for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/status")
async def get_processing_status():
    return PROCESSING_STATUS
@app.get("/list-models")
async def list_available_models():
    try:
        model_data = get_available_models()
        params = model_data["params"]
        model_names = model_data["model_names"]
        model_details = []
        for model in model_names:
            model_details.append({
                "model": model,
                "thres_q": params.loc[model, 'thres_q'],
                "days": params.loc[model, 'days'],
                "benchmark": params.loc[model, 'benchmark'],
                "wt_scheme": params.loc[model, 'wt_scheme'],
                "description": params.loc[model, 'Description'] if 'Description' in params.columns else ""
            })
        return {
            "available_models": model_names,
            "model_details": model_details
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/list-results")
async def list_processing_results():
    try:
        dirs = [d for d in os.listdir("./") if os.path.isdir(os.path.join("./", d))]
        if not dirs:
            return {"message": "No results available"}
        latest_dir = max(dirs)
        dest_dir = os.path.join("./", latest_dir)
        files = []
        for file in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, file)
            if os.path.isfile(file_path):
                files.append({
                    "name": file,
                    "size": os.path.getsize(file_path),
                    "path": file_path,
                    "download_url": f"/download/{latest_dir}/{file}"
                })
        return {
            "results_directory": dest_dir,
            "cutoff_date": latest_dir,
            "files": files
        }
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
if __name__ == "__main__":
    logger.info("Starting Investment Model Processing API")
    uvicorn.run(app, host="0.0.0.0", port=8000)