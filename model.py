import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import collections
import re
import copy
import random

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.num_merges = vocab_size
        self.vocab = []
        self.merges = {}
    def _get_stats(self, word_freqs):
        pairs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    def _merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            v_out[p.sub(''.join(pair), word)] = v_in[word]
        return v_out
    def train(self, corpus):
        base_vocab_list = sorted(list(set("".join(corpus).replace(" ", ""))))
        word_freqs = collections.defaultdict(int)
        for text in corpus:
            for word in text.strip().split():
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        for i in range(self.num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            self.merges[best_pair] = i
        final_tokens = base_vocab_list + ["".join(token) if isinstance(token, tuple) else token for token in sorted(self.merges.keys(), key=self.merges.get)]
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.vocab = self.special_tokens + list(dict.fromkeys(final_tokens))
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.unk_id = self.token_to_id["<unk>"]
        print(f"BPE токенизатор обучен. Размер словаря: {len(self.vocab)}")
    def encode(self, text):
        pre_tokenized_words = [' '.join(list(word)) + ' </w>' for word in text.strip().split()]
        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
            for i, word in enumerate(pre_tokenized_words):
                pre_tokenized_words[i] = self._merge_vocab(pair, {word: 1}).popitem()[0]
        final_tokens = ' '.join(pre_tokenized_words).split()
        return [self.token_to_id.get(token, self.unk_id) for token in final_tokens]
    def decode(self, ids):
        tokens = [self.id_to_token.get(i, '<unk>') for i in ids]
        return ''.join(tokens).replace('</w>', ' ').strip()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q = q.view(q.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(context.shape[0], -1, self.d_model)
        return self.W_o(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask)
        x = x + self.ff(self.norm3(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def make_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)
    def make_tgt_mask(self, tgt):
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.shape[1]
        seq_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        return pad_mask & seq_mask
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc_out(tgt)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def augment_data(conversations):
    augmented = []
    for q, a in conversations:
        q_clean, a_clean = clean_text(q), clean_text(a)
        augmented.append((q_clean, a_clean))
        words = q_clean.split()
        if len(words) > 1:
            for i in range(len(words)):
                new_q = " ".join(words[:i] + words[i+1:])
                if new_q:
                    augmented.append((new_q, a_clean))
    print(f"Аугментация завершена. Исходных примеров: {len(conversations)}, стало: {len(augmented)}")
    return augmented

conversations = [
    ("привет", "здравствуй"), ("добрый день", "и вам добрый"), ("здравствуй", "и тебе привет"),
    ("пока", "до скорой встречи"), ("до свидания", "всего хорошего"),
    ("кто ты", "я нейросеть текстовая модель"), ("как тебя зовут", "у меня нет имени"),
    ("что ты умеешь", "я могу отвечать на простые вопросы"),
    ("как дела", "все отлично спасибо что спросил"), ("большое спасибо", "не за что"),
    ("благодарю", "всегда пожалуйста"),
    ("меня зовут", "очень приятно познакомиться")
]
known_questions = [clean_text(q) for q, a in conversations]
known_words = set(" ".join(known_questions).split())
augmented_conversations = augment_data(conversations)
corpus = [q for q, a in augmented_conversations] + [a for q, a in augmented_conversations]
tokenizer = BPETokenizer(vocab_size=70)
tokenizer.train(corpus)
vocab_size = len(tokenizer.vocab)
PAD_ID = tokenizer.token_to_id["<pad>"]
SOS_ID = tokenizer.token_to_id["<sos>"]
EOS_ID = tokenizer.token_to_id["<eos>"]
max_seq_length = 20
def pad_sequence(tokens, max_len, pad_id):
    return (tokens + [pad_id] * max_len)[:max_len]
src_data_list, tgt_data_list, y_labels_list = [], [], []
for q, a in augmented_conversations:
    src_tokens = tokenizer.encode(q)
    tgt_tokens = tokenizer.encode(a)
    src_data_list.append(pad_sequence(src_tokens + [EOS_ID], max_seq_length, PAD_ID))
    tgt_data_list.append(pad_sequence([SOS_ID] + tgt_tokens, max_seq_length, PAD_ID))
    y_labels_list.append(pad_sequence(tgt_tokens + [EOS_ID], max_seq_length, PAD_ID))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_data = torch.LongTensor(src_data_list).to(device)
tgt_data = torch.LongTensor(tgt_data_list).to(device)
y_labels = torch.LongTensor(y_labels_list).to(device)

d_model = 128
num_heads = 4
num_layers = 3
d_ff = 512
learning_rate = 0.0001
epochs = 500
model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, PAD_ID).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
print("Начало обучения на PyTorch...")
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(src_data, tgt_data)
    loss = criterion(output.view(-1, vocab_size), y_labels.view(-1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss.item():.4f}")
print("Обучение завершено.")

alias_memory = {}
context_memory = collections.defaultdict(list)

def _generate_single_response(clean_input):
    model.eval()
    src_tokens = tokenizer.encode(clean_input)
    src_tensor = torch.LongTensor([pad_sequence(src_tokens + [EOS_ID], max_seq_length, PAD_ID)]).to(device)
    output_ids = [SOS_ID]
    for _ in range(max_seq_length - 1):
        with torch.no_grad():
            tgt_tensor = torch.LongTensor([pad_sequence(output_ids, max_seq_length, PAD_ID)]).to(device)
            output = model(src_tensor, tgt_tensor)
        
        last_logits = output[0, len(output_ids) - 1, :]
        last_word = tokenizer.decode([output_ids[-1]]).strip()
        
        if last_word in context_memory and random.random() < 0.7:
            new_word = random.choice(context_memory[last_word])
            response = tokenizer.decode(output_ids) + " " + new_word
            return response.replace("<sos>", "").strip()

        probs = torch.softmax(last_logits, dim=-1)
        next_word_id = torch.argmax(probs).item()
        
        if next_word_id == EOS_ID:
            break
        output_ids.append(next_word_id)
        
    raw_response = tokenizer.decode(output_ids)
    return raw_response.replace("<sos>", "").strip()

def chat(user_input):
    clean_input = clean_text(user_input)
    correction_threshold = 2
    
    words = clean_input.split()
    for i, word in enumerate(words):
        if word not in known_words and i > 0:
            prev_word = words[i-1]
            if prev_word in known_words:
                if word not in context_memory[prev_word]:
                     context_memory[prev_word].append(word)
                     print(f"(Контекстная память: после '{prev_word}' может идти '{word}')")

    if clean_input in alias_memory:
        known_phrase = alias_memory[clean_input]
        print(f"(Память алиасов: '{clean_input}' -> '{known_phrase}')")
        clean_input = known_phrase

    found_known_phrases = []
    remaining_input = " " + clean_input + " "
    for phrase in sorted(known_questions, key=len, reverse=True):
        search_phrase = " " + phrase + " "
        if search_phrase in remaining_input:
            found_known_phrases.append(phrase)
            remaining_input = remaining_input.replace(search_phrase, " ", 1)
    
    remaining_input = remaining_input.strip()

    if remaining_input:
        best_match = None
        min_dist = float('inf')
        all_known_items = known_questions + list(alias_memory.keys())
        for item in all_known_items:
            if remaining_input == item: continue
            dist = levenshtein_distance(remaining_input, item)
            if dist < min_dist:
                min_dist = dist
                best_match = item
        
        if min_dist <= correction_threshold:
            print(f"(Думаю, '{remaining_input}' - это опечатка в '{best_match}')")
            corrected_phrase = alias_memory.get(best_match, best_match)
            if corrected_phrase not in found_known_phrases:
                found_known_phrases.append(corrected_phrase)
        elif len(found_known_phrases) == 1:
            alias = remaining_input
            known_part = found_known_phrases[0]
            contains_known_words_in_alias = any(word in known_words for word in alias.split())
            if not contains_known_words_in_alias and alias not in known_questions and alias not in alias_memory:
                alias_memory[alias] = known_part
                print(f"(Память алиасов: запомнил '{alias}' -> '{known_part}')")

    if not found_known_phrases:
        best_match = None
        min_dist = float('inf')
        for question in known_questions:
            dist = levenshtein_distance(clean_input, question)
            if dist < min_dist:
                min_dist = dist
                best_match = question
        if min_dist <= correction_threshold:
            print(f"(Думаю, вы имели в виду: '{best_match}')")
            found_known_phrases.append(best_match)
        else:
            return _generate_single_response(clean_input)
    
    responses = [_generate_single_response(phrase) for phrase in found_known_phrases]
    unique_responses = list(dict.fromkeys(responses))
    final_response = ", ".join(unique_responses)
    return final_response.capitalize() if final_response else "Я не совсем понял, можешь перефразировать?"

print("\nУлучшенная модель 3.0. Попробуйте 'ало привет', затем 'ало', а затем 'привет алт'.")
while True:
    user_message = input("Вы: ")
    if user_message.lower() == 'выход':
        break
    response = chat(user_message)
    print(f"B1TLER-GPT: {response}")
