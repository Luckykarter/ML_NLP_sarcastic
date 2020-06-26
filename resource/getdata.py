import os
import zipfile
import requests
import json

def get_datastore():
    # temporary change working dir to work with resource folder
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    DATA_FILE = 'sarcasm.json'
    if not os.path.exists(DATA_FILE):
        DATA_SOURCE = 'Sarcasm_Headlines_Dataset.json'
        # get Headlines dataset from Kaggle
        if not os.path.exists(DATA_SOURCE):
            try:
                URL = open('dataseturl.txt', 'r').read()
                r = requests.get(URL, allow_redirects=True)
                zip_path = 'zip_file.zip'
                open(zip_path, 'wb').write(r.content)
                zip = zipfile.ZipFile(zip_path)
                zip.extractall()
                os.remove(zip_path)
            except Exception as e:
                print(e)
                exit(1)

        # process JSON's to be Python dictionaries inside array

        with open(DATA_FILE, 'w') as new_file:
            content = '['
            with open(DATA_SOURCE, 'r') as f:
                while True:
                    next_line = f.readline()
                    if not next_line:
                        content = content[:-2] + ']'
                        break
                    content += (next_line[:-1] + ',\n')
            new_file.write(content)

    with open(DATA_FILE, 'r') as f:
        datastore = json.load(f)

    # return back working directory
    os.chdir(working_dir)
    return datastore
