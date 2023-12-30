import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import TextBlob
from transformers import BertModel, BertTokenizer, logging
from torch import nn
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm  # 导入tqdm库，用于在循环中显示进度条
import torch


def convert_time_to_seconds(time_str):
    """处理时间转化为以秒为单位"""
    if isinstance(time_str, str):  # 判断是否为空字符串或者缺失数据
        hours = re.findall(r'(\d+)h', time_str)
        minutes = re.findall(r'(\d+)min', time_str)
        hours = int(hours[0]) if hours else 0
        minutes = int(minutes[0]) if minutes else 0
        total_seconds = hours * 3600 + minutes * 60
        return total_seconds
    else:
        return None


def convert_to_int(count_str):
    """处理人数"""
    if 'K' in count_str:
        # 如果包含"k"，则将"k"替换为空字符，然后乘以1000转为整数
        return int(float(count_str.replace('K', '')) * 1000)
    else:
        # 否则直接转为整数
        return int(count_str)


class PreProcessData:
    def __init__(self):
        self.original_data_home = '../archive/'
        self.processed_data_home = '../data/'
        self.bert_path = '../bert-base-uncased'
        self.movies = pd.read_csv(self.original_data_home+'movies.csv')
        self.ratings = pd.read_csv(self.original_data_home + 'ratings.csv')
        self.tags = pd.read_csv(self.original_data_home+'tags.csv')
        self.info = pd.read_csv(self.original_data_home+'info.csv')
        self.film_ori = pd.read_csv(self.original_data_home+'film_ori.csv')

    def process_movies(self):
        # 分割出电影名和上映时间两个数据
        pattern = r"(?P<name>.+)\((?P<year>\d+)\)"
        self.movies[["name", "year"]] = self.movies["title"].str.extract(pattern)

        # 使用MultiLabelBinarizer生成表示是否属于某个类别的多个列的数据
        mlb = MultiLabelBinarizer()
        self.movies['genres'] = self.movies['genres'].str.split('|')
        genres_dummies = pd.DataFrame(mlb.fit_transform(self.movies['genres']), columns=mlb.classes_,
                                      index=self.movies['movieId'])
        # genres_dummies = genres_dummies.astype(bool)      # 如果需要将各个genres作为bool型变量，则取消该条注释

        # 统计各个genre的个数
        genre_counts = genres_dummies.iloc[:, :].sum().sort_values(ascending=False)
        print('各genre的统计')
        print(genre_counts)

        genres_dummies = genres_dummies.reset_index()   # 重新得到movieId这一列数据

        # 将year由字符串类型转为整型
        genres_and_year = genres_dummies
        genres_and_year['year'] = pd.to_numeric(self.movies['year'], downcast='integer', errors='coerce')

        return genres_and_year

    def process_ratings(self):
        # 计算每部电影的用户平均打分
        average_ratings = self.ratings.groupby('movieId')['rating'].mean().reset_index()

        # 分桶操作，划分为10或5个类别，区间取左开右闭（没有为0的值）
        # bin_edges = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        bin_edges = [0, 1, 2, 3, 4, 5]
        average_ratings['rating_class'] = pd.cut(average_ratings['rating'], bins=bin_edges, labels=False, right=True)

        # 统计每部电影的打分人数
        average_ratings['rating_count'] = self.ratings['movieId'].value_counts()

        # 输出各分数段电影的个数及比例
        class_counts = average_ratings['rating_class'].value_counts()
        class_proportions = average_ratings['rating_class'].value_counts(normalize=True)
        print(f'class_counts:\n{class_counts}')
        print(f'class_proportions:\n{class_proportions}')

        print(f'shape of average_ratings:{average_ratings.shape}')
        return average_ratings

    def process_tags(self):
        # 统计每部电影的所有tag
        all_tags = self.tags.groupby("movieId")["tag"].apply(list).reset_index()

        # 统计tag数，即每部电影有多少人给出tag
        all_tags['tag_count'] = self.tags['movieId'].value_counts()

        print(f'shape of all_tags:{all_tags.shape}')
        return all_tags

    def process_info(self):
        # 对DataFrame中的每个时间数据应用转换函数
        self.info['time'] = self.info['time'].apply(convert_time_to_seconds)
        self.info['Users_reviews'] = self.info['Users_reviews'].apply(convert_to_int)
        self.info['Critic reviews'] = self.info['Critic reviews'].apply(convert_to_int)
        self.info['star_level'] = self.info['star_level'].astype(float)
        self.info['Director_stars'] = self.info['Director_stars'].astype(float)

    def bert_lstm_process(self, texts, tokenizer):
        texts = tokenizer(texts, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        bert = BertModel.from_pretrained(self.bert_path).to(device)
        # lstm = nn.LSTM(input_size=792, hidden_size=396, num_layers=6, batch_first=True, bidirectional=True)
        dropout = nn.Dropout(0.5).to(device)
        mask = texts['attention_mask'].to(device)
        input_id = texts['input_ids'].squeeze(1).to(device)
        outputs = bert(input_ids=input_id, attention_mask=mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output, _ = last_hidden_state.max(dim=1)
        return dropout(pooled_output).tolist()[0]

    def process_film_ori(self):
        columns_to_drop = ['Runtime', 'Release Year', "Title", "Directors",
                           "Actors", "tag", "imdb", "Review_rat", "Genres"]
        self.film_ori.drop(columns=columns_to_drop, inplace=True)

        # # 处理Review
        # self.film_ori['Review'].fillna('no review', inplace=True)
        # tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        #
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = [executor.submit(self.bert_lstm_process, texts, tokenizer) for texts in self.film_ori['Review']]
        #     # 使用tqdm显示进度
        #     for _ in tqdm(as_completed(futures), total=len(futures)):
        #         pass
        #
        # result_df = pd.DataFrame([future.result() for future in futures],
        #                          columns=['bert{}'.format(i) for i in range(768)])
        # self.film_ori = pd.concat([self.film_ori.drop(columns='Review'), result_df], axis=1)
        self.film_ori.drop(columns=['Review'], inplace=True)

    def process_data(self, mode):
        genres_and_year = self.process_movies()
        average_ratings = self.process_ratings()
        all_tags = self.process_tags()
        self.process_info()
        self.process_film_ori()

        # 合并各数据
        data = pd.merge(genres_and_year, all_tags, on='movieId', how='outer')
        data = pd.merge(data, average_ratings, on='movieId', how='outer')
        data = pd.merge(data, self.info, on='movieId', how='outer')
        data = pd.merge(data, self.film_ori, on='movieId', how='outer')

        # 根据tag获取情感计分值
        data['tag'].fillna(value=pd.Series([[]] * len(data)), inplace=True)  # 用空列表填充缺失值
        data['tag'] = data['tag'].apply(lambda tags: ','.join(tags))    # 将列表转为用逗号分割的字符串
        data['Sentiment'] = data['tag'].apply(lambda tags: TextBlob(tags).sentiment.polarity)

        # 填充各count的缺失值为0
        data['rating_count'].fillna(0, inplace=True)
        data['tag_count'].fillna(0, inplace=True)

        columns_to_fill = ['Budget', 'Gross US/Canada', 'Opening Weekend', 'Gross Worldwide']
        # # 用中位数填充票房缺失值并进行标准化
        # for column in columns_to_fill:
        #     median_value = data[column].median()  # 中位数
        #     data[column].replace([0, pd.NA, pd.NaT, np.nan], median_value, inplace=True)
        # for column in columns_to_fill:
        #     median_value = data[column].median()
        #     std_dev = data[column].std()
        #     data[column] = (data[column] - median_value) / std_dev
        # 用均值来填充缺失数据
        for column in columns_to_fill:
            mean_value = data[column].mean()  # 计算均值
            data[column].replace([0, pd.NA, pd.NaT, np.nan], mean_value, inplace=True)
        for column in columns_to_fill:
            mean_value = data[column].mean()
            std_dev = data[column].std()
            data[column] = (data[column] - mean_value) / std_dev

        if mode == 1:
            """分类模型的处理"""
            rating_class = data['rating_class']
            data.drop(['tag', 'rating', 'rating_class'], axis=1, inplace=True)
            data = pd.concat([data, rating_class], axis=1)
        else:
            """回归模型的处理"""
            rating = data['rating']
            data.drop(['tag', 'rating', 'rating_class'], axis=1, inplace=True)
            data = pd.concat([data, rating], axis=1)

        # 得到有缺失值的样本
        na_data = data[data.isna().any(axis=1)]
        # 找到有缺失值的列，保留'movieId'
        columns_to_keep = ['movieId']+na_data.columns[na_data.isna().any()].tolist()
        na_data = na_data.filter(columns_to_keep)
        # print(f'含有缺失值的行号：\n{na_data.index}')
        print(f"含有缺失值的行号和movieId:\n{na_data['movieId']}")
        print(f'共{len(na_data)}条有缺失')

        # 丢掉仍有缺失值的数据
        data.drop(['movieId'], axis=1, inplace=True)
        data.dropna(inplace=True)

        print(f'shape of data:{data.shape}')

        if mode == 1:
            data.to_csv(self.processed_data_home + 'data_for_classification_raw.csv', index=False)
            # data.to_csv(self.processed_data_home + 'data_for_classification_raw_has_review.csv', index=False)
        else:
            data.to_csv(self.processed_data_home + 'data_for_regression_raw.csv', index=False)
            # data.to_csv(self.processed_data_home + 'data_for_regression_raw_has_review.csv', index=False)


if __name__ == '__main__':
    logging.set_verbosity_error()
    s = PreProcessData()
    s.process_data(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
