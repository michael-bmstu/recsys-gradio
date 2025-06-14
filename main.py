from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.logger import setup_logger
import dvc.api
from pathlib import Path
from app import interface
import gradio as gr


params = dvc.api.params_show()
SEED = params['seed']
DATA = Path(params['data_root'])
WEIGHTS = Path(params['weight_root'])
logger = setup_logger()

app_api = FastAPI(title='Gradio Interface')
app_interface = interface.create_interface()

@app_api.get('/')
async def root():
    logger.info('visiting the home page')
    return RedirectResponse(url='/gradio')
    
app_api = gr.mount_gradio_app(app_api, app_interface, path='/gradio', pwa=True)