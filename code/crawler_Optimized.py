import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import logging
import csv
from tqdm import tqdm  # 导入tqdm库，用于在循环中显示进度条
import concurrent.futures   # 用于多线程优化
import pandas as pd


class Model:
    def __init__(self):
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.o (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/65.0.3325.162 Safari/537.36'
        }
        # 存放每一步电影的movieId和imdb的id
        self.movie_dct = {}
        # 存放已经处理完的movie id
        self.white_lst = []
        # 电影详情的初始url
        self.url = 'https://www.imdb.com/title/'
        self.url2 = 'https://www.imdb.com/'
        self.movie_csv_path = '../archive/links.csv'
        # 电影信息的保存文件
        self.info_save_path = '../info/info_raw.csv'
        self.process_info_save_path = '../archive/info.csv'
        # logging的配置，记录运行日志
        self.log_save_path = '../crawler_runs/'
        logging.basicConfig(filename=self.log_save_path+"run.log",
                            filemode="a+", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        # 表示当前处理的电影
        self.session = requests.session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20))
        self.session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20))

    def get_white_lst(self):
        """获取处理完的白名单"""
        with open(self.log_save_path+'white_list') as fb:
            for line in fb:
                line = line.strip()
                self.white_lst.append(line)

    def get_movie_id(self):
        """获取电影的id和imdb的id"""
        with open(self.movie_csv_path) as fb:
            fb.readline()
            for line in fb:
                line = line.strip()
                line = line.split(',')
                # 电影id 对应 imdbid
                self.movie_dct[line[0]] = line[1]

    def update_white_lst(self, movie_id):
        """更新白名单"""
        with open(self.log_save_path+'white_list', 'a+') as fb:
            fb.write(movie_id + '\n')

    def update_black_lst(self, movie_id, msg=''):
        with open(self.log_save_path+'black_list', 'a+') as fb:
            # 写入movie id 和imdb id，并且加上错误原因
            # msg=1是URL失效
            fb.write(movie_id + ' ' + self.movie_dct[movie_id] + ' ' + msg + '\n')

    def get_url_response(self, url, movie_id):
        """访问网页请求，返回response"""
        # logging.info(f'get {url}')
        i = 0
        # 超时重传，最多5次
        while i < 5:
            try:
                response = self.session.get(url, timeout=6, headers=self.headers)
                if response.status_code == 200:
                    # logging.info(f'get {url} success')
                    # 正常获取，直接返回
                    return response
                # 如果状态码不对，获取失败，返回None，不再尝试
                logging.error(f'get {url} status_code error: {response.status_code} movie_id is {movie_id}')
                return None
            except requests.RequestException:
                # 如果超时
                logging.error(f'get {url} error, try to restart {i + 1}')
                i += 1
        # 重试5次都失败，返回None
        return None

    def process_html(self, html, movie_id):
        """解析html，获取电影信息"""
        soup = BeautifulSoup(html, 'html.parser')   # 在新版的lxml中不再使用'lxml'而应该改为’html.parser'

        # 电影的基本信息   1h 21min | 4.5 | 82 | 73 | 4.1
        info = ['']*5

        try:
            # 时长时间
            info[0] = soup.find(class_='sc-69e49b85-0 jqlHBQ').find_all(
                class_='ipc-inline-list__item')[-1].get_text().strip()
        except Exception:
            # 没有则添加空字符串
            info[0] = ''

        try:
            # 电影的星级
            info[1] = soup.find(class_='sc-bde20123-1 cMEQkK').get_text().strip()
        except Exception:
            # 没有则添加空字符串
            info[1] = ''

        # 电影的users reviews数和Critic reviews数
        try:
            # users reviews数
            info[2] = soup.find(class_='score').get_text().strip()
        except Exception:
            # 没有则添加'0'
            info[2] = '0'
        try:
            # Critic reviews数
            info[3] = soup.find_all(class_='score')[1].get_text().strip()
        except Exception:
            # 没有则添加'0'
            info[3] = '0'

        # 电影导演的电影的平均星级
        try:
            director_link = self.url2 + soup.find(class_='ipc-metadata-list-item__list-content-item ipc-metadata-'
                                                         'list-item__list-content-item--link').get('href')
            response = self.get_url_response(director_link, movie_id)
            soup2 = BeautifulSoup(response.content, 'html.parser')
            director_star = soup2.find(class_='ipc-metadata-list ipc-metadata-list--dividers-between '
                                              'ipc-metadata-list--base'
                                       ).find_all(class_='ipc-rating-star ipc-rating-star--base ipc-rating-star--imdb '
                                                         'ipc-rating-star-group--imdb')
            star_list = []
            for star in director_star:
                star_list.append(float(star.get_text().strip()))
            length = len(star_list)
            average_star = 0
            if length:
                average_star = sum(star_list) / len(star_list)
            info[4] = '{:.1f}'.format(average_star)  # 保留一位小数加入到info中
        except Exception:
            # 没有则添加'0'
            info[4] = '0'

        # id，时长，星级，users reviews数，Critic reviews数，导演的电影的平均星级
        detail = [movie_id]+info
        self.save_info(detail)

    def save_info(self, detail):
        # 存储到CSV文件中
        with open(f'{self.info_save_path}', 'a+', encoding='utf-8', newline='') as fb:
            writer = csv.writer(fb)
            writer.writerow(detail)

    def run(self):
        # 开始爬取信息
        self.get_white_lst()
        self.get_movie_id()

        # 使用线程池进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(
                self.get_movie_details, movie_id, imdb_id) for movie_id, imdb_id in self.movie_dct.items()]

            # 使用tqdm显示进度
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

    def get_movie_details(self, movie_id, imdb_id):
        if movie_id in self.white_lst:
            return
        # self.cur_movie_id = movie_id
        # self.cur_imdb_id = imdb_id

        response = self.get_url_response(self.url + 'tt' + imdb_id, movie_id)
        if response is None:
            self.save_info([movie_id]+['']*5)
            self.update_white_lst(movie_id)
            self.update_black_lst(movie_id, '1')
            return

        self.process_html(response.content, movie_id)

        # 处理完成，增加movie id到白名单中
        self.update_white_lst(movie_id)
        # logging.info(f'process movie {movie_id} success')

    def process_info(self):
        df = pd.read_csv(self.info_save_path, header=None)
        df.columns = ['movieId', 'time', 'star_level', 'Users_reviews', 'Critic reviews', 'Director_stars']
        df = df.sort_values(by='movieId')
        df = df.reset_index(drop=True)
        # df = df.drop_duplicates(subset='movieId') # 如果有重复的则去掉，可能由于手动删除白名单和info_raw不统一导致重复爬取
        df.to_csv(self.process_info_save_path, index=False)


if __name__ == '__main__':
    s = Model()
    s.run()
    s.process_info()
