from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score.rouge_scorer import RougeScorer
import torch
import evaluate
import random


def evaluate_transformer(val_path, model_name='distilgpt2', max_length=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    with open(val_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    predictions_texts = [] 
    targets_texts = []
    input_texts = []
    for line_idx, line in enumerate(lines):
        words = line.split()
        seq_len = len(words)
        if seq_len < 4:  
            continue
        split_point = seq_len * 3 // 4
        input_text = ' '.join(words[:split_point])
        target_text = ' '.join(words[split_point:])
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )

          
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

       
        outputs = outputs.cpu()
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_texts.append(input_text)
        predictions_texts.append(generated_text)
        targets_texts.append(target_text)
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions_texts, references=targets_texts)
    for key, value in rouge_scores.items():
        print(f"{key}: {value:.4f}")


    print("\n" + "="*80)
    print("10 ПРИМЕРОВ:")
    print("="*80)

    indices = random.sample(range(len(input_texts)), 10)
    for i in indices:
        print(f"\n--- Пример {i+1} ---")
        print(f"Входной текст: {input_texts[i]} *** {targets_texts[i]}")
        print(f"Предсказание: {predictions_texts[i]}")
    