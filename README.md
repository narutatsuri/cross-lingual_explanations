# Exploring Cross-Lingual Transfer in Emotion Explanation Generation

## Description of Folders
 - ```conf```: Contains YAML files for configuring training (i.e., model checkpoint, language, save directory, etc.). 

 - ```data```: Contains the Emotion Explanation dataset. File names are formatted as ```[SPLIT]_lang=[LANGUAGE]-data=[DATASET_SIZE]-shots=[SHOTS]```. 
     - ```[DATASET_SIZE]```: Can be either ```split_en``` or ```split_ja```, which are the dataset sizes specified in the proejct writeup.
     - ```[SHOTS]```: Only if applicable. Contains ```[SHOTS]``` number of English training examples. ```full``` indicates the full English dataset (i.e., ```training_lang=en-data=split_en-shots=full.csv``` and ```training_lang=en-data=split_en.csv``` contain the same contents).
 - ```notebooks```: Contains Python Notebooks where data analysis is conducted.
 - ```prompting```: Contains templates for prompting.
     - ```analysis```: Contains prompt templates for investigating whether emotions are language-agnostic.
     - ```evaluation```: Contains prompt templates for scoring generated explanations.
     - ```data_augmentation```: Contains templates for data generation.
     - ```openai_keys.json```: File to store OpenAI key. 
 - ```results```: Scoring results for all models.
 - ```scripts```: Bash scripts for automating running model training sequentially.
 - ```src```: Main directory for code.
     - ```analysis```: Code for investigating whether emotions are language-agnostic.
     - ```data_generation```: Code for data augmentation.
     - ```utils```: Helper folder containing various functions, classes, and methods. 
     