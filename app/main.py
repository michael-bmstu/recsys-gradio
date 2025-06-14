from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from logger import setup_logger
import dvc.api
from pathlib import Path
import utils
import gradio as gr


params = dvc.api.params_show()
SEED = params['seed']
DATA = Path(params['data_root'])
WEIGHTS = Path(params['weight_root'])
logger = setup_logger()

# загрузка файлов и весов
try:
    anime_clean, anime_genre, kmeans, scaler = utils.read_prepare()
    logger.info('Данные успешно прочитаны')
except:
    logger.info('Подготовленных данных не обнаружено')
    logger.info('Запуск предобработки данных и обучения модели')
    utils.first_run()
    anime_clean, anime_genre, kmeans, scaler = utils.read_prepare()
    logger.info('Данные успешно прочитаны')

app_api = FastAPI()

geners = list(anime_genre.columns)

def dynamic_change(selected_r1, selected_r2, selected_r3):
    selected_r1 = set(selected_r1)
    selected_r2 = set(selected_r2)
    selected_r3 = set(selected_r3)

    selected_genres = selected_r1 | selected_r2 | selected_r3
    new_r1_choices = [genre for genre in geners if genre not in selected_genres or genre in selected_r1]
    new_r2_choices = [genre for genre in geners if genre not in selected_genres or genre in selected_r2]
    new_r3_choices = [genre for genre in geners if genre not in selected_genres or genre in selected_r3]

    return gr.update(choices=new_r1_choices), gr.update(choices=new_r2_choices), gr.update(choices=new_r3_choices)

app_interface = gr.Blocks(title="Оценка жанров аниме")
with app_interface:
    r1 = gr.Dropdown(choices=geners, label='good', multiselect=True, interactive=True)
    r2 = gr.Dropdown(choices=geners, label='normal', multiselect=True, interactive=True)
    r3 = gr.Dropdown(choices=geners, label='bad', multiselect=True)

    r1.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])
    r2.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])
    r3.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])


@app_api.get('/')
async def root():
    logger.info('Посещение стартовой страницы')
    return RedirectResponse(url="/gradio")
    
app_api = gr.mount_gradio_app(app_api, app_interface, path="/gradio")

# @app_api.get('/gradio')
# async def interface():
#     app_interface.launch(share=True) # for public set True
#     return {'message': 'для открытия интерфейса перейдите на страницу http://localhost:7860'}