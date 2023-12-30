import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from tqdm import tqdm  # 导入tqdm库，用于在循环中显示进度条
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split  # 导入PyTorch的数据处理工具
from torch.utils.tensorboard import SummaryWriter  # 导入PyTorch的TensorBoard可视化工具
import os
import math


def same_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_feat(train_data, valid_data, select_all=True):
    """选取特征和标签,默认最后一列作为标签"""
    y_train, y_valid = train_data.iloc[:, -1], valid_data.iloc[:, -1]
    raw_x_train, raw_x_valid = train_data.iloc[:, :-1], valid_data.iloc[:, :-1]
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # feat_idx = []
        feat_idx = [21, 22, 24, 25, 26]     # 选择 rating_count,tag_count,Users_reviews,Critic reviews,star_level
        # feat_idx = list(range(20, 28))  # 可以在这里修改要选取的特征,当select_all 为false时则会在这里去选取特征
    all_columns = raw_x_train.columns.tolist()
    feature_list = [all_columns[i] for i in feat_idx]
    return raw_x_train.iloc[:, feat_idx], raw_x_valid.iloc[:, feat_idx], y_train, y_valid, feature_list


def data_process(data_path, mode=1):
    # 读取数据集
    data = pd.read_csv(data_path)
    print(f'shape of data:{data.shape}')
    # 划分数据集为训练集和验证集，比例为9：1
    data_train, data_val = train_test_split(data, test_size=0.1)

    x_train, x_valid, y_train, y_valid, feature_list = select_feat(data_train, data_val, select_all=True)
    if mode == 3:
        x_train = torch.tensor(x_train.values, dtype=torch.float32)
        x_valid = torch.tensor(x_valid.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        y_valid = torch.tensor(y_valid.values, dtype=torch.float32)

    # 特征标准化 注意fit_transform和transform的区别，前者用于训练集的拟合和转换，后者用于测试集或新数据的转换
    # 以保证数据的一致性和正确的预处理操作
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    print(f'number of features: {len(feature_list)}')
    return x_train, x_valid, y_train, y_valid, feature_list


def evaluation_for_classification(model, x_valid, y_valid, mode=1):
    # target_names = ['(0,0.5]', '(0.5,1]', '(1,1.5]', '(1.5,2]', '(2,2.5]', '(2.5,3]',
    #                 '(3,3.5]', '(3.5,4]', '(4,4.5]', '(4.5,5]']
    target_names = ['(0,1]', '(1,2]', '(2,3]', '(3,4]', '(4,5]']
    if mode == 3:
        x_valid = torch.FloatTensor(x_valid)
        y_predict_array = model(x_valid).numpy()
        y_predict = np.argmax(y_predict_array, axis=1)
    else:
        y_predict = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_predict)
    report = classification_report(y_valid, y_predict, target_names=target_names, zero_division=1)
    print(f'模型准确率:{accuracy}')
    print(f'分类报告：\n{report}')


def evaluation_for_regression(model, x_valid, y_valid, mode=1):
    if mode == 3:
        x_valid = torch.FloatTensor(x_valid)
        y_predict = model(x_valid).numpy()
    else:
        y_predict = model.predict(x_valid)
    mse = mean_squared_error(y_valid, y_predict)
    print(f'mse:{mse}, rmse:{mse ** 0.5}')
    print(f'accuracy(用1-mse表示):{1-mse}')


def importance_analysis(model, feature_list):
    importance = list(model.feature_importances_)
    feature_importance = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importance)]
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    print('特征重要性分析：')
    [print('Variable: {:20} Importance:{}'.format(*pair)) for pair in feature_importance]


class RandomForestParamsForBayesian:
    def __init__(self, mode, model, x_train, x_valid, y_train, y_valid):
        self.mode = mode
        self.x_train, self.x_valid, self.y_train, self.y_valid = x_train, x_valid, y_train, y_valid
        # 随机森林的模型
        self.model = model
        # 建立树的个数
        self.n_estimators = (10, 400)
        # 最大特征的选择方式
        self.max_features = (0.1, 0.999)
        # 树的最大深度
        self.max_depth = (5, 100)
        # 节点最小分裂所需要的样本数
        self.min_samples_split = (2, 50)
        # 叶子节点最小样本数，任何分裂不能让其子节点样本数少于此值
        self.min_samples_leaf = (1, 32)
        # 搜索参数空间
        self.search_grid = {'n_estimators': self.n_estimators, 'max_features': self.max_features,
                            'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split,
                            'min_samples_leaf': self.min_samples_leaf}

    def black_box_function(self, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf):
        res = self.model(n_estimators=int(n_estimators), max_features=min(max_features, 0.999),     # float
                         max_depth=int(max_depth), min_samples_split=int(min_samples_split),
                         min_samples_leaf=int(min_samples_leaf)).fit(self.x_train, self.y_train)
        if self.mode == 1:
            y_predict = res.predict(self.x_valid)
            return accuracy_score(self.y_valid, y_predict)
        else:
            y_predict = res.predict(self.x_valid)
            return 1 - mean_squared_error(self.y_valid, y_predict)

    def find_best_params_bayesian(self):
        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=self.search_grid,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            # random_state=42,
        )
        optimizer.maximize(init_points=25, n_iter=15)    # 分别表示随机搜索的步数和执行贝叶斯优化的步数
        print(optimizer.max)



class Model:
    def __init__(self, mode):
        self.mode = mode
        self.data_home = '../data/'
        self.data_for_classification_path = self.data_home + 'data_for_classification_raw.csv'
        self.data_for_regression_path = self.data_home + 'data_for_regression_raw.csv'
        # self.data_for_classification_path = self.data_home + 'data_for_classification_raw_has_review.csv'
        # self.data_for_regression_path = self.data_home + 'data_for_regression_raw_has_review.csv'
        if mode == 1:
            data_path = self.data_for_classification_path
        else:
            data_path = self.data_for_regression_path
        self.x_train, self.x_valid, self.y_train, self.y_valid, self.feature_list = data_process(data_path)

    def train(self, model):
        model.fit(self.x_train, self.y_train)

    def eval(self, model):
        if self.mode == 1:
            evaluation_for_classification(model, self.x_valid, self.y_valid)
        else:
            evaluation_for_regression(model, self.x_valid, self.y_valid)


class MyDataset(Dataset):
    def __init__(self, x, y=None, mode=1):
        if y is None:
            self.y = y
        else:
            if mode == 1:
                self.y = torch.FloatTensor(y).type(torch.LongTensor)  # 分类模型y作为标签应该采用整型
            else:
                self.y = torch.FloatTensor(y)   # 回归模型时y采用Float表示连续值
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),  # 激活函数
            # nn.Linear(32, 16),
            # nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # 去除输出张量中的维度为 1 的维度，使输出变为一维张量
        return x


class ModelNet:
    def __init__(self, mode):
        # 配置模型训练的参数
        self.config = {
            'seed': 42,  # 随机数生成器的种子
            'n_epochs': 100,  # 训练轮数
            'batch_size': 16,  # 批次大小
            'learning_rate': 0.001,  # 学习率
            'early_stop': 30,  # 如果模型在连续这么多轮没有改进，就停止训练
            'output_dim': 10,
            'data_path': '../data/data_for_classification_raw.csv',  # 读取数据的路径
            # 'data_path': '../data/data_for_classification_raw_has_review.csv',  # 读取数据的路径
            'save_path': '../nn_save_models/ranking_predict1.ckpt'  # 模型保存的路径
        }
        self.mode = mode
        if mode == 1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.config['output_dim'] = 1
            self.criterion = nn.MSELoss()
            self.config['save_path'] = '../nn_save_models/ranking_predict2.ckpt'
            self.config['data_path'] = '../data/data_for_regression_raw.csv'
            # self.config['data_path'] = '../data/data_for_regression_raw_has_review.csv'

        # same_seed(self.config['seed'])  # 设置随机数种子
        self.x_train, self.x_valid, self.y_train, self.y_valid, self.feature_list = \
            data_process(self.config['data_path'], 3)
        self.model = MyModel(input_dim=self.x_train.shape[1], output_dim=self.config['output_dim'])

        # 创建训练集、验证集和测试集的数据集对象
        train_dataset, valid_dataset = MyDataset(self.x_train, self.y_train, mode), \
                                       MyDataset(self.x_valid, self.y_valid, mode)
        # 创建用于训练、验证和测试的数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.15, weight_decay=1e-5)  # 使用随机梯度下降优化器
        writer = SummaryWriter(log_dir='../nn_runs')  # 使用TensorBoard可视化训练过程

        if not os.path.isdir('../nn_save_models'):
            os.mkdir('../nn_save_models')

        n_epochs, best_loss, step, early_stop_count = self.config['n_epochs'], math.inf, 0, 0
        # 从config字典中获取n_epochs的值
        ''' n_epochs 用于指定训练的总轮数，best_loss 被用来保存目前遇到的最佳损失值，step 用于迭代计数，
        而 early_stop_count 表示出现连续没有改善的轮数计数器,可能会在早停算法中用于判断是否停止训练 '''
        for epoch in range(n_epochs):
            self.model.train()  # 模型切换为训练模式
            loss_record = []

            train_pbar = tqdm(self.train_loader, position=0, leave=True)

            for x, y in train_pbar:
                optimizer.zero_grad()  # 清除梯度
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                optimizer.step()  # 更新模型参数
                step += 1
                loss_record.append(loss.detach().item())

                train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})  # 设置后缀信息

            mean_train_loss = sum(loss_record) / len(loss_record)
            writer.add_scalar('Loss/train', mean_train_loss, step)

            self.model.eval()
            loss_record = []
            for x, y, in self.valid_loader:
                with torch.no_grad():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                loss_record.append(loss.item())

            mean_valid_loss = sum(loss_record) / len(loss_record)
            print(
                f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
            writer.add_scalar('Loss/valid', mean_valid_loss, step)
            if mean_train_loss < best_loss:
                best_loss = mean_train_loss
                torch.save(self.model.state_dict(), self.config['save_path'])  # 用来保存模型参数
                print('Saving model with loss {:.3f}...'.format(best_loss))
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.config['early_stop']:
                print('\nModel is not improving, so we halt the training session.')
                return

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            if self.mode == 1:
                evaluation_for_classification(self.model, self.x_valid, self.y_valid, 3)
            else:
                evaluation_for_regression(self.model, self.x_valid, self.y_valid, 3)


if __name__ == '__main__':
    # 定义分类模型
    M1 = Model(1)
    # 逻辑回归模型（实际用于分类问题）
    model1 = LogisticRegression(multi_class='multinomial', solver='newton-cg')  # 'lbfgs','newton-cg','sag','saga'
    print('逻辑回归模型：')
    M1.train(model1), M1.eval(model1)

    # 随机森林分类模型（优化前）
    model2 = RandomForestClassifier(n_estimators=50)
    print('随机森林分类模型（优化前）:')
    M1.train(model2), M1.eval(model2), importance_analysis(model2, M1.feature_list)

    # 找到最佳参数
    RandomForestParamsForBayesian(1, RandomForestClassifier, M1.x_train, M1.x_valid,
                                  M1.y_train, M1.y_valid).find_best_params_bayesian()
    # 随机森林分类模型（优化后）
    model2_optimized = RandomForestClassifier(n_estimators=131, min_samples_split=5, min_samples_leaf=11,
                                              max_features=0.66, max_depth=86)
    print('随机森林分类模型（优化后）:')
    M1.train(model2_optimized), M1.eval(model2_optimized), importance_analysis(model2_optimized, M1.feature_list)

    # 神经网络分类模型
    model3 = ModelNet(1)
    model3.train()
    # 加载最佳模型参数
    model3.model.load_state_dict(torch.load(model3.config['save_path']))
    model3.eval()

    # 定义回归模型
    M2 = Model(2)
    # 线性回归模型
    model4 = LinearRegression()
    print('线性回归模型：')
    M2.train(model4), M2.eval(model4)

    # 随机森林回归模型（优化前）
    model5 = RandomForestRegressor(n_estimators=50)
    print('随机森林回归模型（优化前）:')
    M2.train(model5), M2.eval(model5), importance_analysis(model5, M2.feature_list)

    # 找到最佳参数
    RandomForestParamsForBayesian(2, RandomForestRegressor, M2.x_train, M2.x_valid,
                                  M2.y_train, M2.y_valid).find_best_params_bayesian()
    # 随机森林回归模型（优化后）
    model5_optimized = RandomForestRegressor(n_estimators=225, min_samples_split=17, min_samples_leaf=4,
                                             max_features=0.3738, max_depth=41)
    print('随机森林分类模型（优化后）:')
    M2.train(model5_optimized), M2.eval(model5_optimized), importance_analysis(model5_optimized, M2.feature_list)

    # 神经网络回归模型
    model6 = ModelNet(2)
    model6.train()
    # 加载最佳模型参数
    model6.model.load_state_dict(torch.load(model6.config['save_path']))
    model6.eval()
