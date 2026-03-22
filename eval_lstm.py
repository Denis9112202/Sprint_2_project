from rouge_score.rouge_scorer import RougeScorer
import torch
import evaluate

def calculate_rouge_scores(predictions, targets):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    for pred, target in zip(predictions, targets):
        score = scorer.score(' '.join(map(str, pred)), ' '.join(map(str, target)))
        scores.append({
            'rouge1': score['rouge1'].fmeasure,
            'rouge2': score['rouge2'].fmeasure,
            'rougeL': score['rougeL'].fmeasure
        })

    avg_scores = {
        'rouge1': sum(s['rouge1'] for s in scores) / len(scores),
        'rouge2': sum(s['rouge2'] for s in scores) / len(scores),
        'rougeL': sum(s['rougeL'] for s in scores) / len(scores)
    }
    return avg_scores

def evaluate_lstm(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []
    inputs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['texts'].to(device)
            masks = batch['masks'].to(device)
            batch_size, seq_len = input_ids.shape
            for i in range(batch_size):
                real_length = masks[i].sum().item()
                split_point = real_length * 3 // 4
                input_part = input_ids[i:i+1, :split_point]
                target_part = input_ids[i, split_point:real_length].tolist()
                generated = model.generate(
                    input_part,
                    max_length = real_length - split_point
                )
                generated_tokens = generated[0, split_point:].tolist()
                inputs.append(input_part)
                predictions.append(generated_tokens)
                targets.append(target_part)      
           
            rouge_scores = calculate_rouge_scores(predictions, targets)
    return rouge_scores, predictions, inputs, targets