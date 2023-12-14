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
 - ```scripts```: Bash scripts for automating running model training sequentially. (Most are removed for clarity)
 - ```src```: Main directory for code.
     - ```analysis```: Code for investigating whether emotions are language-agnostic.
     - ```data_generation```: Code for data augmentation.
     - ```utils```: Helper folder containing various functions, classes, and methods. 

## Data downloading / generation / preprocessing
```
python3 src/data_generation/generate_summaries.py --data_dir [PATH TO COVID-ET DATASET] 
                                                  --prompt_template prompting/data_augmentation/generate_data.json
python3 src/data_generation/generate_explanations.py --data_dir [PATH TO OUTPUT OF PREVIOUS RUN] 
                                                     --prompt_template prompting/data_augmentation/generate_explanations.json
python3 src/data_generation/generate_data.py --data_dir [PATH TO OUTPUT OF PREVIOUS RUN] 
                                             --prompt_template prompting/data_augmentation/generate_data.json 
                                             --example_retrieval [RETRIEVAL METHOD FOR IN-CONTEXT EXAMPLES]
                                             --num_examples [NO. OF IN-CONTEXT EXAMPLES]
                                             --num_samples_per_emotion [NO. OF INSTANCES PER EMOTION]
python3 src/translate_generated_data.py --data_dir [PATH TO OUTPUT OF PREVIOUS RUN] 
                                        --prompt_template prompting/data_augmentation/translate_generated_data.json
```

## Training Models
To run a batch of model training, refer to ```scripts/script_train.sh``` to train the same model checkpoint (e.g., ```mt0-large```) with different number of English examples. To run training individually, run:
```
python3 src/train.py --config_file [CONFIG FILE]
                     --shots [SHOTS] 
                     --source [SOURCE LANGUAGE] 
                     --target [TARGET LANGUAGE]
                     --save_strategy [SAVE STRATEGY] 
                     --epochs [EPOCHS FOR CROSS-LINGUAL TRANSFER]
                     --device [DEVICE] 
                     --huggingface_cache [PATH TO HUGGINGFACE CACHE]
```
Additional configurations can be made in ```conf/backbone.yaml```.

## Data Analysis
Refer to ```notebooks/data_analysis.ipynb``` and ```src/analysis```.

### Analyzing Language-Agnostic Property of Emtoions
```
python3 src/analysis/emotion_test.py --data_dir data/lang=en-data=emotion_test.json --prompt_template prompting/analysis/label_emotions_en.json
python3 src/analysis/emotion_test.py --data_dir data/lang=ja-data=emotion_test.json --prompt_template prompting/analysis/label_emotions_ja.json
```
These will create JSON files in ```results/emotion_check``` containing emotion labels by ```gpt-3.5-turbo```.

## Evaluating Model Outputs
```
python3 src/test_automated.py --model_dir [PATH TO MODEL CHECKPOINT]
                              --data_dir [PATH TO TEST DATA]
                              --device [DEVICE] 
                              --huggingface_cache [PATH TO HUGGINGFACE CACHE]
python3 src/test_prompting.py --model_dir [PATH TO MODEL CHECKPOINT]
                              --data_dir [PATH TO TEST DATA]
                              --prompt_template [PATH TO PROMPT TEMPLATE] 
                              --score_metric [COMPARISON, FLUENCY, OR ACCURACY]
                              --device [DEVICE] 
                              --huggingface_cache [PATH TO HUGGINGFACE CACHE]
```
These scripts will populate ```results/``` with a list of folders containing score outputs for each model.