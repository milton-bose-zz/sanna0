import pickle

def read_file(filepath, file_type='txt'):
    if file_type == 'txt':
        with open(filepath, mode='r') as f:
            data = f.read()
    elif file_type == 'pkl':
        with open(filepath, mode='rb') as f:
            data = pickle.load(f)
    else:
        raise Exception('Type not recognized')

    return data

def load_datasets(filepath, data_processor=None):

    data = read_file(filepath, file_type='pkl')
    if data_processor is not None:
        data = data_processor(data)

    return data

