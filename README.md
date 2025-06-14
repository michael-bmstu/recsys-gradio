# Gradio interface for clustring recomendation system

## Project struct
* **`app`** : app directory
    * `interface.py` : creating gradio interface
    * `logger.py` : logger initialization
    * `utils.py` : auxiliary functions
* `log` : application logs
* `main.py` : main application file (combines all functions)
* `preprocess.py` : script for dvc pipeline
* **`data`** : data directory
* **`weigths`** : directory with model weights
* `dvc.lock` : hash of files tracked by dvc
* `dvc.yaml` : dvc stage configuration (pipeline, tracked files and parameters)
* `params.yaml` : parameters for dvc
* `requirements.txt` : necessary packages for the project