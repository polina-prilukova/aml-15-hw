import sys
import os
import kaggle as kg


def download_data():
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path + "\\Data")
    kg.api.authenticate()
    kg.api.dataset_download_files(dataset="rajyellow46/wine-quality", path=module_path + '\\Data', unzip=True)

