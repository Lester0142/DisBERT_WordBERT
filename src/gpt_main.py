import pandas as pd
import torch

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fix the seed for reproducibility
set_seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Function to sample 10% of the data
def sample_data(data, fraction=1.0):
    return data.sample(frac=fraction, random_state=42)

# Load and sample the training data
train_data_path = './data/VUA18/train.tsv'
train_data = pd.read_csv(train_data_path, delimiter='\t')
train_data = sample_data(train_data)

# Load and sample the validation data
val_data_path = './data/VUA18/dev.tsv'
val_data = pd.read_csv(val_data_path, delimiter='\t')
val_data = sample_data(val_data)

# Function to add special tokens around the target word
def add_special_tokens(sentence, w_index):
    words = sentence.split()
    words = [" "] + words if sentence.startswith(" ") else words
    words[w_index] = f"<target> {words[w_index]} </target>"
    return ' '.join(words)

# Preprocess the training data
train_sentences = [add_special_tokens(sent, idx) for sent, idx in zip(train_data['sentence'].tolist(), train_data['w_index'].tolist())]
train_labels = train_data['label'].tolist()

# Preprocess the validation data
val_sentences = [add_special_tokens(sent, idx) for sent, idx in zip(val_data['sentence'].tolist(), val_data['w_index'].tolist())]
val_labels = val_data['label'].tolist()

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'additional_special_tokens': ['<target>', '</target>']})
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to accommodate new special tokens
model.to(device)  # Move model to GPU
model.model_parallel = False  # Set model_parallel attribute

# Tokenize the input sentences
train_inputs = tokenizer(train_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
val_inputs = tokenizer(val_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Convert labels to tensor and move to GPU
train_labels = torch.tensor(train_labels).to(device)
val_labels = torch.tensor(val_labels).to(device)

# Create a dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = CustomDataset(train_inputs, train_labels)
val_dataset = CustomDataset(val_inputs, val_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',  # Evaluate every epoch
    save_steps=10000,  # Save model every 500 steps
)

# Define a compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']}")
print(f"Validation Precision: {eval_results['eval_precision']}")
print(f"Validation Recall: {eval_results['eval_recall']}")
print(f"Validation F1 Score: {eval_results['eval_f1']}")

# Load and sample the test data
test_data_path = './data/VUA18/test.tsv'
test_data = pd.read_csv(test_data_path, delimiter='\t')
test_data = sample_data(test_data)

# Preprocess the test data
test_sentences = [add_special_tokens(sent, idx) for sent, idx in zip(test_data['sentence'].tolist(), test_data['w_index'].tolist())]
test_labels = test_data['label'].tolist()

# Tokenize the test sentences
test_inputs = tokenizer(test_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Convert test labels to tensor and move to GPUs
test_labels = torch.tensor(test_labels).to(device)

# Create the test dataset
test_dataset = CustomDataset(test_inputs, test_labels)

# Evaluate the model on the test dataset
test_results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']}")
print(f"Test Precision: {test_results['eval_precision']}")
print(f"Test Recall: {test_results['eval_recall']}")
print(f"Test F1 Score: {test_results['eval_f1']}")