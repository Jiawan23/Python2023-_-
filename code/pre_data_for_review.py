import torch
import pandas as pd
import numpy as np
from math import ceil
import random


def pre_first1():
    ori_path = r"./archive/film_ori.csv"
    ori_data = pd.read_csv(ori_path)
    # columns_to_drop = ['Runtime', 'Release Year', "Title", "Directors", "Actors", "tag", "imdb", "Review_rat"]
    columns_to_drop = ['Runtime', 'Release Year', "Title", "Directors", "Actors", "tag"]
    ori_data.drop(columns=columns_to_drop, inplace=True)
    ori_data = ori_data.dropna()
    columns_to_fill = ['Budget', 'Gross US/Canada', 'Opening Weekend', 'Gross Worldwide']
    for column in columns_to_fill:
        median_value = ori_data[column].median()
        ori_data[column].replace([0, pd.NA, pd.NaT, np.nan], median_value, inplace=True)
    for column in columns_to_fill:
        median_value = ori_data[column].median()
        std_dev = ori_data[column].std()
        ori_data[column] = (ori_data[column] - median_value) / std_dev
    genres_dummies = ori_data["Genres"].str.get_dummies(sep="|")
    ori_data = pd.concat([ori_data, genres_dummies], axis=1)
    ori_data.drop(columns=["Genres"], inplace=True)
    return ori_data


def preprocess():
    rating_path = r"./archive/ratings.csv"
    rating = pd.read_csv(rating_path)
    average_ratings = rating.groupby('movieId')['rating'].mean().reset_index()
    ori_data = pre_first1()
    ori_data = pd.merge(ori_data, average_ratings, on="movieId", how="inner")
    ori_data = ori_data.dropna()
    ori_data.drop(columns=['movieId'], inplace=True)
    text_data = ori_data["Review"]
    numeric_data = ori_data.drop(columns=['Review', 'rating'])
    target_data = ori_data['rating'].apply(lambda x: ceil(x) - 1)
    new_data = pd.concat([text_data, target_data], axis=1)
    new_data = pd.concat([numeric_data, new_data], axis=1)
    # new_data = generate_new_data(new_data)
    return new_data


# 计算评分的数量
def calculate_ratios(data):
    rating_counts = data['rating'].value_counts().sort_index()
    rating_counts_dict = rating_counts.to_dict()
    return rating_counts_dict


# 计算缩放因子
def calculate_scaling_factors(rating_counts):
    scaling_factors = {}
    for rating in range(5):
        if rating != 3:
            scaling_factors[rating] = rating_counts[3] / rating_counts[rating]
    return scaling_factors


# 删除tag中的随机词语
def remove_random_words(tag):
    words = tag.split()
    num_words = len(words)
    if num_words > 2:  # 如果词数足够多，可以随机删除
        num_words_to_remove = random.randint(1, num_words - 1)  # 随机数量的词语，保证至少删除一个词汇
        words_to_remove = random.sample(words, num_words_to_remove)
        words = [word for word in words if word not in words_to_remove]
    return ' '.join(words)


# 构建新的数据集
def generate_new_data(data):
    rating_counts = calculate_ratios(data)
    scaling_factors = calculate_scaling_factors(rating_counts)
    new_data = pd.DataFrame(columns=data.columns)

    for rating, count in rating_counts.items():
        subset = data[data['rating'] == rating].copy().reset_index(drop=True)
        subset.drop_duplicates(inplace=True)
        new_data = pd.concat([new_data, subset], ignore_index=True)
        if rating != 3:
            for _ in range(int(scaling_factors[rating])):
                temp = subset.copy()
                temp['Review'] = temp['Review'].apply(remove_random_words)
                new_data = pd.concat([new_data, temp], ignore_index=True)
    rating_counts = calculate_ratios(new_data)
    new_data.drop_duplicates(inplace=True)
    rating_counts = calculate_ratios(new_data)
    new_data['rating'] = new_data['rating'].astype('int64')
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    numeric_columns = new_data.drop(columns=['Review', 'rating']).columns.tolist()
    new_data[numeric_columns] = new_data[numeric_columns].astype('float32')
    return new_data


if __name__ == '__main__':
    ori_data = preprocess()
