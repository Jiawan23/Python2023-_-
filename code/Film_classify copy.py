from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pre_data_for_review import preprocess
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel

from torch import nn
from torch.optim import Adam, SGD
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  

# model_path = r"E:/pan/movie/py/py/bert-base-cased"
model_path = r'./bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
writer = SummaryWriter('./log2')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df["rating"]
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)
        self.texts = [tokenizer(text, padding='max_length', max_length=config["max_length"], truncation=True,
                                return_tensors="pt") for text in df['Review']]
        self.numeric_data = torch.tensor(df.drop(columns=['Review', 'rating']).values, dtype=torch.float32)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return self.labels[idx]

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_numeric_data(self, idx):
        return self.numeric_data[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_numeric = self.get_batch_numeric_data(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_numeric, batch_y


# Bert+BiLSTM，用法与BertClassifier一样，可直接在train里面调用
class BertLstmClassifier(nn.Module):
    def __init__(self,  dropout=0.5):
        super(BertLstmClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.lstm = nn.LSTM(input_size=792, hidden_size=396, num_layers=6, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(792, 5)  # 双向LSTM 需要乘以2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask, numeric_data):
        outputs = self.bert(input_ids=input_id, attention_mask=mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output, _ = last_hidden_state.max(dim=1)
        pooled_output = self.dropout(pooled_output)
        combined = torch.cat((pooled_output, numeric_data), dim=1)
        logits = self.classifier(combined)
        return logits


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    # train_dataloader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # weights = torch.tensor([0.3, 0.25, 0.15, 0.1, 0.2])
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch_num in range(epochs):
        torch.cuda.empty_cache()  # 清空显存
        total_acc_train = 0
        total_loss_train = 0
        step = 0
        for train_texts, train_numeric, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_texts['attention_mask'].to(device)
            input_id = train_texts['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask, train_numeric.to(device))  # 传递数字数据到模型中
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
             # 将损失和准确率写入TensorBoard
            writer.add_scalar("Train/Loss", batch_loss.item(), step)
            writer.add_scalar("Train/Accuracy", acc / len(train_label), step)
            step += 1
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for val_texts, val_numeric, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_texts['attention_mask'].to(device)
                input_id = val_texts['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask, val_numeric.to(device))  # 传递数字数据到模型中
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
                output_list = output.cpu().numpy().tolist()
                val_label_list = val_label.long().cpu().numpy().tolist()
                path_ = r"../data/data.txt"
                with open(path_, 'a') as file:
                    file.write(f"the valid:\n")
                    for out, label in zip(output_list, val_label_list):
                        file.write(f"Output: {out}, Label: {label}\n")

        print(
            f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_data): .5f} 
          | Train Accuracy: {total_acc_train / len(train_data): .5f} 
          | Val Loss: {total_loss_val / len(val_data): .5f} 
          | Val Accuracy: {total_acc_val / len(val_data): .5f}''')

config = {
    "learning_tare": 0.0001,
    "spilt_rate": 0.15,
    "hidden_size": 396,
    "batch_size": 16,
    "epochs": 16,
    "seed": 65,
    # "seed": 42,
    "max_length": 128,
    "total_features": 1
}
if __name__ == "__main__":
    np.random.seed(config["seed"])
    ori_data = preprocess()
    config["total_features"] = ori_data.shape[1]
    ori_train, ori_val = train_test_split(ori_data, test_size=config["spilt_rate"], random_state=config["seed"])
    model = BertLstmClassifier()
    train(model, ori_train, ori_val, config["learning_tare"], config["epochs"])
