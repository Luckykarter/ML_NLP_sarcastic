import os
import zipfile
import requests
import json

def get_datastore():
    DATA_SOURCE = 'Sarcasm_Headlines_Dataset.json'

    # get Headlines dataset from Kaggle
    if not os.path.exists(DATA_SOURCE):
        try:
            URL = open('dataseturl.txt', 'r').read()
            r = requests.get(URL, allow_redirects=True)
            zip_path = 'zip_file.zip'
            open(zip_path, 'wb').write(r.content)
            zip = zipfile.ZipFile(zip_path)
            zip.extractall('.')
            os.remove(zip_path)
        except Exception as e:
            print(e)
            exit(1)

    # process JSON's to be Python dictionaries inside array
    with open('sarcasm.json', 'w') as new_file:
        content = '['
        with open(DATA_SOURCE, 'r') as f:
            while True:
                next_line = f.readline()
                if not next_line:
                    content = content[:-2] + ']'
                    break
                content += (next_line[:-1] + ',\n')
        new_file.write(content)

    with open('sarcasm.json', 'r') as f:
        datastore = json.load(f)
    return datastore
