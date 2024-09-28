import re
import string

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()

    cleaned_data = clean_text(data)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_data)

    print(f"Processed data saved to {output_file}")


process_file('data/processor/rawdataset.txt', 'data/dataset.txt')
