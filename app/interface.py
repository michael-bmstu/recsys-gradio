import gradio as gr
from .logger import setup_logger
from . import utils
import numpy as np

logger = setup_logger()

# загрузка файлов и весов
try:
    anime_clean, anime_genre, anime_train, kmeans, scaler = utils.read_prepare()
    logger.info('Данные успешно прочитаны')
except:
    logger.info('Подготовленных данных не обнаружено')
    logger.info('Запуск предобработки данных и обучения модели')
    utils.first_run()
    anime_clean, anime_genre, anime_train, kmeans, scaler = utils.read_prepare()
    logger.info('Данные успешно прочитаны')

genres = list(anime_genre.columns)
mean_rating = anime_train[genres].mean()
genres = list(mean_rating.sort_values(ascending=False).index)
anime_clean = anime_clean.sort_values(by='rating', ascending=False)

def dynamic_change(selected_r1, selected_r2, selected_r3):
    selected_r1 = set(selected_r1)
    selected_r2 = set(selected_r2)
    selected_r3 = set(selected_r3)

    selected_genres = selected_r1 | selected_r2 | selected_r3
    new_r1_choices = [genre for genre in genres if genre not in selected_genres or genre in selected_r1]
    new_r2_choices = [genre for genre in genres if genre not in selected_genres or genre in selected_r2]
    new_r3_choices = [genre for genre in genres if genre not in selected_genres or genre in selected_r3]

    return gr.update(choices=new_r1_choices), gr.update(choices=new_r2_choices), gr.update(choices=new_r3_choices)

def recomend(r1, r2, r3, hist, slider):
    good = set(r1)
    normal = set(r2)
    bad = set(r3)
    top_n = int(slider)
    sample = []
    for g, mean_r in zip(genres, mean_rating):
        if g in good:
            sample.append(10)
        elif g in normal:
            sample.append(6)
        elif g in bad:
            sample.append(2)
        else:
            sample.append(mean_r)

    sample = np.array(sample).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    label = kmeans.predict(sample_scaled)[0]
    cluster = kmeans.cluster_centers_[label]

    hist_id = anime_clean[anime_clean['name'].isin(hist)].index
    anime_id = utils.get_best_anime(
        cluster, anime_genre[~anime_genre.index.isin(hist_id)], genres
        )
    anime_top = anime_clean.loc[anime_id].iloc[:top_n]
    recs = anime_top[['name', 'rating']].to_html(index=False)
    text = "## The best anime for you"
    return text, recs

def create_interface():
    app_interface = gr.Blocks(title="Rating anime genres")
    with app_interface:
        r1 = gr.Dropdown(choices=genres, label='good', multiselect=True)
        r2 = gr.Dropdown(choices=genres, label='normal', multiselect=True)
        r3 = gr.Dropdown(choices=genres, label='bad', multiselect=True)
        r1.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])
        r2.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])
        r3.change(fn=dynamic_change, inputs=[r1, r2, r3], outputs=[r1, r2, r3])

        hist = gr.Dropdown(choices=list(anime_clean['name']), label='View history', multiselect=True)

        slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Select count of recomendations")

        btn = gr.Button('recommend')
        label = gr.Markdown()
        recs = gr.HTML()
        btn.click(recomend, inputs=[r1, r2, r3, hist, slider], outputs=[label, recs])
    return app_interface