from config import params
from pathlib import Path
from app import interface

SEED = params['seed']
DATA = Path(params['data_root'])
WEIGHTS = Path(params['weight_root'])

app_interface = interface.create_interface()

if __name__ == "__main__":
    app_interface.launch(server_name='0.0.0.0')