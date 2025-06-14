from app import utils
utils.first_run()


"""
dvc stage add -n preprocess -d "data/anime.csv" -d "data/rating.csv" -p seed \
-o "data/rating_clean.csv" -o "data/anime_clean.csv" -o "data/anime_genre.csv" -o "data/anime_train.csv" \
-o "weights/kmeans.pkl" -o "weights/scaler.pkl" \
"C:/Users/Михаил/Documents/venvs/gradio/Scripts/python.exe" preprocess.py
"""