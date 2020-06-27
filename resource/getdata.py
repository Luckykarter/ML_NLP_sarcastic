import os
import zipfile
import requests
import json

def get_datastore():
    # temporary change working dir to work with resource folder
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_file = 'sarcasm.json'
    if not os.path.exists(data_file):
        data_source = 'Sarcasm_Headlines_Dataset.json'
        # get Headlines dataset from Kaggle
        if not os.path.exists(data_source):
            try:
                url = open('dataseturl.txt', 'r').read()
                r = requests.get(url, allow_redirects=True)
                zip_path = 'zip_file.zip'
                open(zip_path, 'wb').write(r.content)
                zip = zipfile.ZipFile(zip_path)
                zip.extractall()
                os.remove(zip_path)
            except Exception as e:
                print(e)
                exit(1)

        # process JSON's to be Python dictionaries inside array

        with open(data_file, 'w') as new_file:
            content = '['
            with open(data_source, 'r') as f:
                while True:
                    next_line = f.readline()
                    if not next_line:
                        content = content[:-2] + ']'
                        break
                    content += (next_line[:-1] + ',\n')
            new_file.write(content)

    with open(data_file, 'r') as f:
        datastore = json.load(f)

    # return back working directory
    os.chdir(working_dir)
    return datastore
