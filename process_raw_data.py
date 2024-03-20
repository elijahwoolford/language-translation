import re
from tqdm import tqdm


def clean_text(text):
    """Cleans the input text by lowering the case and removing special characters."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s.,!?']", '', text)  # Remove all characters except words, whitespace, and punctuation
    return text


def preprocess_data(input_file, eng_output_file, esp_output_file):
    """Reads a TSV file and writes separate files for English and Spanish sentences."""
    with open(input_file, 'r', encoding='utf-8') as tsvfile, \
            open(eng_output_file, 'w', encoding='utf-8') as engfile, \
            open(esp_output_file, 'w', encoding='utf-8') as espfile:

        for line in tqdm(tsvfile):
            parts = line.strip().split('\t')
            if len(parts) == 4:  # Make sure the line is properly formatted
                eng, esp = parts[1], parts[3]  # Extract English and Spanish sentences
                eng, esp = clean_text(eng), clean_text(esp)  # Clean the text
                engfile.write(eng + '\n')  # Write the English sentence to the file
                espfile.write(esp + '\n')  # Write the Spanish sentence to the file


# Change 'your_dataset.tsv' to the path of your actual TSV file
input_tsv = '/Users/ewoolford/Downloads/en_sp_translation_data.tsv'
english_output = 'english_sentences.txt'
spanish_output = 'spanish_sentences.txt'

preprocess_data(input_tsv, english_output, spanish_output)
