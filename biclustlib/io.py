import json

from models import Bicluster, Biclustering

def save_biclusterings(b, file_path, extension='.json'):
    with open(file_path + extension, 'w') as f:
        json.dump(b, f, default=_biclustering_to_dict)

def load_biclusterings(filename):
    with open(filename, 'r') as f:
        biclusterings = json.load(f, object_hook=_dict_to_biclustering)

    return biclusterings

def _biclustering_to_dict(bic):
    d = {'__class__' : bic.__class__.__name__, '__module__' : bic.__module__}
    d['biclusters'] =  [(list(b.rows), list(b.cols)) for b in bic.biclusters]
    return d

def _dict_to_biclustering(bic_dict):
    return Biclustering([Bicluster(rows, cols) for rows, cols in bic_dict[u'biclusters']])
