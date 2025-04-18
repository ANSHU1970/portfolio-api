import os
import logging
import traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from datetime import datetime
from func import get_only_cutoff_date, investmentUniverse, investmentModels, fred_data
import uvicorn
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
        if sheet_name == 'port_statistics':
            sheet_data[sheet_name] = df.set_index(df.columns[0]).to_dict(orient='index')
        else:
            sheet_data[sheet_name] = df.to_dict(orient='records')
    return JSONResponse(content=sheet_data)
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