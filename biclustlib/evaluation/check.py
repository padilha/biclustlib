from biclustlib.models import Biclustering

def check_biclusterings(b1, b2):
    if not isinstance(b1, Biclustering) or not isinstance(b2, Biclustering):
        raise ValueError('b1 and b2 must be Biclustering instances')

    if len(b1.biclusters) == 0 and len(b2.biclusters) == 0:
        return 1.0

    if len(b1.biclusters) == 0 or len(b2.biclusters) == 0:
        return 0.0

    return None
