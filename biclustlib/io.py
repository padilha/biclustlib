import json

from .models import Bicluster, Biclustering

def save_biclusterings(b, file_path, extension='json'):
    """Dumps biclusterings to a file using the json module.

    Parameters
    ----------
    b : biclustlib.models.Biclustering or list
        A Biclustering instance or a list of Biclustering instances that will be saved.

    file_path : str
        The path of the file where the Biclustering instance or list of Biclustering instances will be saved.

    extension : str, default: 'json'
        The file extension to be used.
    """
    with open(file_path + '.' + extension, 'w') as f:
        json.dump(b, f, default=_biclustering_to_dict)

def load_biclusterings(file_path):
    """Load biclusterings from a json file.

    Parameters
    ----------
    file_path : str
        The path of the file where the Biclustering instance or list of Biclustering instances are stored.
    """
    with open(file_path, 'r') as f:
        biclusterings = json.load(f, object_hook=_dict_to_biclustering)
    return biclusterings

def _biclustering_to_dict(bic):
    d = {'__class__' : bic.__class__.__name__, '__module__' : bic.__module__}
    d['biclusters'] =  [(list(b.rows), list(b.cols)) for b in bic.biclusters]
    return d

def _dict_to_biclustering(bic_dict):
    return Biclustering([Bicluster(rows, cols) for rows, cols in bic_dict['biclusters']])
