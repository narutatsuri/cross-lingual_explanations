import re


SENTIMENT_DATA_DIR = "data/sentiment_explanation"
plutchik = ["anger", "fear", "sadness", "disgust", "anticipation", "trust", "joy"]
plutchik_en_to_ja = {
    "joy": "喜び",
    "trust": "信頼",
    "fear": "恐れ",
    "sadness": "悲しみ",
    "disgust": "嫌悪",
    "anger": "怒り",
    "anticipation": "期待"
}

############################
# General Helper Functions #
############################
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

####################################
# Data Generation Helper Functions #
####################################
def remove_brackets(data):
    bracket_items = []
    for index in range(len(data)):
        brackets = re.findall('\[.*?\]', data[index]["text"])
        if len(brackets) != 0:
            bracket_items += brackets
    bracket_items = set(bracket_items)

    for index in range(len(data)):
        for bracket_item in bracket_items:
            if bracket_item in data[index]["text"]:
                data[index]["text"] = data[index]["text"].replace(bracket_item, "")
        data[index]["text"] = data[index]["text"].strip()
        
    return data

def clean_generated_data(data):
    data = remove_brackets(data)

    for index in range(len(data)):
        example = dict(data[index])
        if "generated_raw" in example:
            del example["generated_raw"]        
        if "id" not in example:
            example["id"] = "generated"
        data[index] = example

    for index in range(len(data)):
        if ": " in data[index]["text"]:
            data[index]["text"] = data[index]["text"].replace(": ", "")
        data[index]["text"] = data[index]["text"].strip()

    return data

def get_emotion_counts(data):
    emotion_counts = {}
    for emotion in plutchik:
        emotion_counts[emotion] = 0
    for example in data:
        emotion_counts[example["choice"]] += 1
    
    return emotion_counts

def get_emotion_splits(data):
    example_by_emotion = {}
    for emotion in plutchik:
        example_by_emotion[emotion] = []
    for example in data:
        example_by_emotion[example["choice"]].append(example)

    return example_by_emotion

def get_duplicates(data):
    duplicates = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if data[i] == data[j]:
                duplicates.append((i,j))    

def remove_duplicates(data):
    return [dict(t) for t in {tuple(d.items()) for d in data}]