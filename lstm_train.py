import torch
from torch.utils.data import DataLoader
from src.eval_lstm import evaluate_lstm
from src.next_token_dataset import NextTokenDataset
from src.lstm_model import LSTMTextGenerator
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertTokenizer

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    max_length = max(len(seq) for seq in texts)
    padded_sequences = []
    masks = []

    for seq in texts:
        padded_seq = seq + [0] * (max_length - len(seq))
        mask = [1] * len(seq) + [0] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
        masks.append(mask)

    input_ids = torch.tensor(padded_sequences, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.long)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[mask == 0] = -100
    labels[:, 0] = -100

    return {
        'texts': input_ids,
        'masks': mask,
        'labels': labels
    }

def train_lstm_model(train_path_token, val_path_token, test_path_token, model_save_path, vocab_size=30522, embedding_dim=128, hidden_dim=128, num_layers=1, batch_size=256, epochs=10, learning_rate=0.001):
    train_dataset = NextTokenDataset(train_path_token)
    val_dataset = NextTokenDataset(val_path_token)
    test_dataset = NextTokenDataset(test_path_token)
    dataset_size = len(test_dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model = LSTMTextGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['texts'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            output, _ = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print("-" * 50)
        print("-" * 50)
        print("-" * 50)
        print(f'Epoch [{epoch+1}/{epochs}], TrainLoss: {total_loss/len(train_loader):.4f}')
        rouge_scores, _,_,_ = evaluate_lstm(model, val_loader, device)
        for key, value in rouge_scores.items():
            print(f"{key}: {value:.4f}")

        
        random_indices = random.sample(range(dataset_size), 10)
        sampler = SubsetRandomSampler(random_indices)
        test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=collate_fn, sampler=sampler)
        _, predictions, inputs, targets = evaluate_lstm(model, test_loader, device)

        inputs_list = [t.squeeze(0).tolist() for t in inputs]
        
        decoded_inputs = tokenizer.batch_decode(
            inputs_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
       
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded_targets = tokenizer.batch_decode(
            targets,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for i, (inp, pred, targ) in enumerate(zip(decoded_inputs, decoded_predictions, decoded_targets)):
            print(f"\n--- Пример {i+1} ---")
            print(f"Вход: {inp} *** {targ}")
            print(f"Предсказание: {inp} *** {pred}")
            print("-" * 50)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")