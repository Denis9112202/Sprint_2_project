

import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def clean_text(text):
    if text is None or (isinstance(text, str) and text.strip() == ''):
        return ""
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_dataset(input_path, clear_path, train_path, val_path, test_path, train_token_path, val_token_path, test_token_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts_cleaned = [clean_text(line.strip()) for line in lines]
    texts_filtered = [text for text in texts_cleaned if len(text) > 0]

    train_indices, val_test_indices = train_test_split(
        range(len(texts_filtered)),
        test_size=0.2,
        random_state=42
    )
    train_texts = [texts_filtered[i] for i in train_indices]
    val_test_texts = [texts_filtered[i] for i in val_test_indices]

    val_indices, test_indices = train_test_split(
        range(len(val_test_texts)),
        test_size=0.5,
        random_state=42
    )
    val_texts = [val_test_texts[i] for i in val_indices]
    test_texts = [val_test_texts[i] for i in test_indices]
    
    def save_to_csv(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + '\n')


    save_to_csv(texts_filtered, clear_path)
    save_to_csv(train_texts, train_path)
    save_to_csv(val_texts, val_path)
    save_to_csv(test_texts, test_path)
    
    
    train_texts_token = []
    for text in train_texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            )
        train_texts_token.append(token_ids)
   
    val_texts_token = []
    for text in val_texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            )
        val_texts_token.append(token_ids)

    test_texts_token = []
    for text in test_texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            )
        test_texts_token.append(token_ids)

    def save_to_csv_token(data, path):
        lines = [','.join(map(str, row)) for row in data]
        content = '\n'.join(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
      
    save_to_csv_token(train_texts_token, train_token_path)
    save_to_csv_token(val_texts_token, val_token_path)
    save_to_csv_token(test_texts_token, test_token_path)

    