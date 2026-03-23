import torch
import torch.nn as nn

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(LSTMTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output, (hidden, cell)

    def generate(self, start_tokens, max_length, temperature=1.0, device='cuda', eos_token_id=102):
        self.eval()
        generated = start_tokens.clone().to(device)

        with torch.no_grad():
            embedded = self.embedding(generated)
            lstm_out, (hidden, cell) = self.lstm(embedded)
            output = self.fc(lstm_out)
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            if next_token.item() == eos_token_id:
                return generated
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            for _ in range(max_length - 1):
                last_token = generated[:, -1:]
                embedded = self.embedding(last_token)
                lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
                output = self.fc(lstm_out)
                next_token_logits = output[0, 0, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                if next_token.item() == eos_token_id:
                    break
                generated = torch.cat([generated, next_token], dim=1)
        return generated