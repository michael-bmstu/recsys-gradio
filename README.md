# Gradio interface for clustring recomendation system
The project is an implementation of the gradio interface for a [recommender system](https://github.com/michael-bmstu/clustering_recomend_system).

## Project struct
* **`app`** : app directory
    * `interface.py` : creating gradio interface
    * `logger.py` : logger initialization
    * `utils.py` : auxiliary functions
* `main.py` : main application file (combines all functions)
* `preprocess.py` : script for dvc pipeline
* `log` : application logs
* **`data`** : data directory
* **`weigths`** : directory with model weights
* `dvc.lock` : hash of files tracked by dvc
* `dvc.yaml` : dvc stage configuration (pipeline, tracked files and parameters)
* `params.yaml` : parameters for dvc
* `requirements.txt` : necessary packages for the project

## Launch of the project
Here are some options to run on your machine. 

For correct operation it is necessary to install data from kaggle by the [link](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
(anime.csv, rating.csv).

### Manual launch using commands in the terminal (windows cmd)
```
python -m venv app_venv
app_venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt
dvc repro
fastapi run main.py --host localhost --port 8000
```
Open your web browser and go to `http://localhost:8000`

### Docker container
/.../