from sys import getsizeof

def get_size_MB(obj): 
    ''' Returns size of an object in MB '''
    return round(getsizeof(obj) / 1024 / 1024, 2)