SENTIMENT_DATA_DIR = "data/sentiment_explanation"
plutchik = ["anger", "fear", "sadness", "disgust", "anticipation", "trust", "joy"]

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def partition(obj, num_partitions):
    """
    """
    chunks = int(len(obj) // num_partitions )

    chunk_size = 0
    chunk_list = []
    buf = []
    for i in obj:
        if chunk_size >= chunks:
            chunk_list.append(buf)
            chunk_size = 0
            buf = []

        buf.append(i)
        chunk_size += 1

    if len(buf) != 0:
        chunk_list.append(buf)
    
    return chunk_list

