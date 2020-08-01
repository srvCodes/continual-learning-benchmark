import math
import pickle
import random
from random import shuffle
from operator import itemgetter
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from train.visualisations import vis_by_person
from train.visualisations.training_visualizer import plot_heatmap
from sklearn.preprocessing import LabelEncoder
# np.random.seed(42)
TEST_SIZE = 0.3
TRAIN_VAL_SPLIT = 0.9
NUM_PERMUTATIONS = 3

label_enc_dict = defaultdict(LabelEncoder)

def drop_column_by_idx(df, index=None):
    df = df.drop(df.columns[index], axis=1)
    return df

def pickle_loader(filepath, encoding='latin1'):
    pickle_file = pickle.load(open(filepath, 'rb'), encoding=encoding)
    return pickle_file

class DataHandler:
    def __init__(self, dataname, base_classes, per_batch_classes, train_percent, seed_value, vis, corr_vis, keep_val):
        self.seed_value = seed_value
        self.vis = vis
        self.corr_vis = corr_vis
        self.seed_randomness()
        self.dataname = dataname
        self.tp = train_percent # considering separate train and test directories for hapt (0.7+0.3 = 1.0)
        self.original_mapping = {}
        self.train_data, self.test_data, self.train_labels, self.test_labels = self.get_features_and_labels()
        self.nb_cl = base_classes
        self.num_classes = per_batch_classes
        self.classes_by_groups = []
        self.label_map = {}
        self.keep_val = keep_val
        self.train_groups, self.val_groups, self.test_groups = self.initialize()

    def seed_randomness(self):
        np.random.seed(self.seed_value)

    @staticmethod
    def read_config():
        with open('conf/data_paths.yaml', 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as err:
                print(err)

    def get_file_path(self):
        config_dict = self.read_config()
        if self.dataname == 'milan' or self.dataname == 'twor' or self.dataname == 'aruba':
            dir_path = config_dict['large_data'][self.dataname]
        elif self.dataname == 'hatn6' or self.dataname == 'ws':
            dir_path = config_dict['small_data'][self.dataname]
        elif self.dataname in ['pamap', 'dsads', 'hapt']:
            dir_path = config_dict['medium_data'][self.dataname]
        else:
            dir_path = config_dict[self.dataname.split('_')[1] if '_' in self.dataname else self.dataname]
        return dir_path

    def get_df_from_mat(self, filename):
        dir_path = self.get_file_path()
        data_path = dir_path + filename + '.mat'
        mat = loadmat(data_path)
        data = mat['data_' + filename]
        df = pd.DataFrame(data=data)
        return df

    def read_data(self):
        config_dict = self.read_config()
        train_data_path = config_dict[self.dataname]['train']
        test_data_path = config_dict[self.dataname]['test']
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        return train_df, test_df

    def get_features_and_labels(self):
        """
        Function to separate features and labels from the dataframe.
        @param df: is the dataframe read from the csv file.
        @return: a tuple of two lists: (features, labels)
        """
        train_df, test_df = self.generate_dataframe()
        train_values, test_values = [df.values.tolist() for df in [train_df, test_df]]
        X_train, X_test = [[np.array(item[1:]) for item in each] for each in [train_values, test_values]]
        y_train, y_test = [[item[0] for item in each ] for each in [train_values, test_values]]
        """ Oversampling of data"""
        # if self.dataname == 'hapt':
        #     counts_dict = Counter(y_train)
        #     sample_nums = {self.original_mapping[i]: math.ceil(counts_dict[self.original_mapping[i]]*2.8) if i != 8 else
        #     math.ceil(counts_dict[self.original_mapping[i]]*5) for i in range(7, 13)}
        #     smote_enn = SMOTEENN(random_state=0, sampling_strategy=sample_nums)
        #     print(f"Before resampling: {counts_dict}, mapping: {self.original_mapping}")
        #     X_train, y_train = smote_enn.fit_resample(X_train, y_train)
        #     X_train = [np.array(item) for item in X_train]
        #     print(f"After resampling: {Counter(y_train)}, mapping: {self.original_mapping}")
        """ End of oversampling"""
        return X_train, X_test, y_train, y_test


    def get_reversed_original_label_maps(self):
        return dict(map(reversed, self.original_mapping.items()))

    def replace_class_labels(self, df, train=False):
        if train:
            sorted_elements = [i for i in range(len(df.AID.unique()))]
            self.original_mapping = dict(zip(df.AID.unique(), sorted_elements))
        df.AID = df.AID.map(self.original_mapping)
        return df

    @staticmethod
    def reorder_columns(df, mnist=False):
        cols = list(df.columns.values)
        if not mnist:
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
        new_col_names = ['AID' if idx==0 else 'S'+str(each) for idx,each in enumerate(cols)]
        df.rename(columns=dict(zip(df.columns, new_col_names)), inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.sort_values(by=['AID'], ascending=True)
        return df


    def get_hapt_data(self, full_path, train=True):
        file_mode = 'train' if train else 'test'
        features_df = pd.read_csv(full_path + 'X_' + file_mode + '.txt', sep=' ', header=None)
        feature_cols = list(features_df.columns.values)
        new_col_names = ['S' + str(each) for idx, each in enumerate(feature_cols)]
        features_df.rename(columns=dict(zip(features_df.columns, new_col_names)), inplace=True)
        labels_df = pd.read_csv(full_path + 'y_' + file_mode + '.txt', sep=' ', header=None)
        features_df.insert(0, 'AID', labels_df[labels_df.columns[0]])
        persons_list = None
        path = 'train' if train else 'test'
        with open(full_path+f'subject_id_{path}.txt', 'r') as FileObj:
            persons_list = np.array([int(line[:-1]) for line in FileObj.readlines()])
            all_persons = np.unique(persons_list)
            if train:
                train_persons = all_persons[:int(len(all_persons) * self.tp)]
                index_of_train_persons = np.concatenate([np.where(persons_list == person)[0] for person in
                                                         train_persons]).ravel()
                features_df = features_df[features_df.index.isin(index_of_train_persons)]
        return (features_df, persons_list)

    def generate_dataframe(self):
        persons_list = None # used for visualizing imbalance in data for 'hapt' dataset, None for all others
        if self.dataname in ['dsads', 'pamap', 'opp']:
            df = self.get_df_from_mat(self.dataname + '_loco' if self.dataname == 'opp' else self.dataname)
            df_to_visualize = df if self.vis else None
            person_column = 407 if self.dataname == 'dsads' else 244 if self.dataname == 'pamap' else 461
            print("Count: ", df.groupby(df.columns[243]).count());
            if self.dataname == 'pamap':
                df.drop(df.loc[(df[df.columns[person_column]] == 9.0) |
                                         (df[df.columns[person_column]] == 3.0)].index,
                        inplace=True)
            persons = df[df.columns[person_column]].unique()
            train_persons = [i + 1 for i in range(math.ceil(max(persons) * (1 - TEST_SIZE)))]
            train_persons = train_persons[:math.ceil(self.tp * len(train_persons))]
            test_persons = [i for i in persons[-int(TEST_SIZE*len(persons)):]]
            # train_persons = np.random.choice(list(persons), math.ceil(max(persons) * (1 - TEST_SIZE)), replace=False).tolist()
            print(f"Persons: {persons}, train: {train_persons}, test: {test_persons}")
            train_df = drop_column_by_idx(df.loc[df[df.columns[person_column]].isin(train_persons)],
                                               index=person_column)
            test_df = drop_column_by_idx(df.loc[df[df.columns[person_column]].isin(test_persons)],
                                              index=person_column)
            if self.dataname == 'dsads':
                # From data descriptions: Column 406 is the activity sequence indicating the executing of activities (usually not used in experiments).
                train_df, test_df = [drop_column_by_idx(df, 405) for df in [train_df, test_df]]
            if self.dataname == 'opp':
                # From data descriptions: Column 461 is the activity drill.
                train_df, test_df = [drop_column_by_idx(df, 460) for df in [train_df, test_df]]
            train_df, test_df = [self.reorder_columns(each) for each in [train_df, test_df]]
        elif self.dataname == 'hapt':
            dir_path = self.get_file_path()
            train_path, test_path = dir_path + 'Train/', dir_path + 'Test/'
            (train_df, persons_list_train), (test_df, persons_list_test) = [self.get_hapt_data(full_path, train=idx==0) for idx, full_path in
                                 enumerate([train_path, test_path])]
            persons_list = np.concatenate((persons_list_train, persons_list_test))
            df_to_visualize = pd.concat((train_df, test_df)) if self.vis else None

        elif self.dataname in ['ws', 'hatn6', 'milan', 'aruba', 'twor']:
            # for ha_t_n6 and ws datasets
            dir_path = self.get_file_path()
            if self.dataname in ['ws', 'hatn6']:
                df = pd.read_csv(dir_path)
            else:
                df = pd.read_excel(dir_path)
                df = df.rename(columns={'activity_label': 'AID'})
                # Source: https://stackoverflow.com/a/31939145/5140684
                # df['AID'] = df.AID.apply(lambda x: label_enc_dict[x].fit_transform(x))
                # Source: https://pbpython.com/categorical-encoding.html
                df['AID'] = df['AID'].astype('category')
                df['AID'] = df['AID'].cat.codes
                df = self.reorder_columns(df)
            df_to_visualize = df.copy() if self.vis else None
            train_df, test_df, _, _ = train_test_split(df, df['AID'], test_size=TEST_SIZE, random_state=self.seed_value,
                                                       stratify=df['AID'])
            train_df_ = train_df.copy()
            # sample dataframe based on count of individual labels: a minimum of 1 sample per class should be present
            train_df = train_df_.groupby('AID').apply(lambda x: x.sample(max(1, int(len(x) * self.tp)))).reset_index(drop=True)

        elif self.dataname == 'cifar100':
            dir_path = self.get_file_path()
            train_set, test_set = [pickle_loader(dir_path[key]) for key in ['train', 'test']]
            train_df = pd.DataFrame(train_set['data'])
            train_df['AID'] = train_set['fine_labels']
            test_df = pd.DataFrame(test_set['data'])
            test_df['AID'] = test_set['fine_labels']
            train_df = self.reorder_columns(train_df)
            test_df = self.reorder_columns(test_df)

        elif 'mnist' in self.dataname:
            dir_path = self.get_file_path()
            train_df, test_df = [pd.read_csv(dir_path[key], header=None, index_col=None) for key in ['train', 'test']]
            train_df, test_df = [self.reorder_columns(df, mnist=True) for df in [train_df, test_df]]
            train_df_ = train_df.copy()
            train_df = train_df_.groupby('AID').apply(lambda x: x.sample(max(1, int(len(x) * self.tp)))).reset_index(drop=True)

        train_df, test_df = [self.replace_class_labels(df, train=idx == 0) for idx, df in enumerate([train_df, test_df])]
        if self.vis:
            self.visualize_by_persons(df_to_visualize, persons_list)
        if self.corr_vis:
            corr = self.compute_correlation_mat(train_df)
            plot_heatmap(corr, out_path=f'corr_vis/by_raw_features/{self.dataname}_tp_{self.tp}.pdf', original_map=self.original_mapping)
        return train_df, test_df

    def compute_correlation_mat(self, df):
        df_of_means = self.compute_mean_of_df(df)
        # df_of_means = df_of_means.rename(columns = self.label_map)
        corr = df_of_means.corr()
        return corr

    @staticmethod
    def compute_mean_of_df(df):
        df_of_means = df.groupby('AID', as_index=True).mean()
        df_of_means = df_of_means.T.iloc[1:]
        return df_of_means

    def visualize_by_persons(self, df, persons_list):
        visualizer = vis_by_person.VisualizeStatsPerPerson(self.dataname, df)
        # if self.dataname == 'pamap' or self.dataname == 'dsads':
        #     visualizer.plot_variance_by_persons()
        visualizer.visualise_imbalance_by_persons(persons_list)

    def reshape_img_dataframes(self, train=False):
        data = self.train_data if train else self.test_data
        data_r = np.array([each[:1024].reshape(32,32) for each in data])
        data_g = np.array([each[1024:2048].reshape(32,32) for each in data])
        data_b = np.array([each[2048:].reshape(32,32) for each in data])
        data = np.dstack((data_r, data_g, data_b))
        if train:
            self.train_data = data
        else:
            self.test_data = data
        return data

    @staticmethod
    def permutate_img_pixels(image, permutation):
        image = image[permutation]
        return image

    def get_data_by_groups(self, train=True):
        if self.dataname == 'cifar100':
            _ = self.reshape_img_dataframes(train=train)
        if train:
            _labels = sorted(set(self.train_labels))
            shuffled_labels = np.random.choice(_labels, len(_labels), replace=False).tolist()
            original_labels = [each for each in range(len(shuffled_labels))]
            self.label_map = dict(zip(shuffled_labels, original_labels))
            print(self.label_map)
            self.classes_by_groups.append(shuffled_labels[:self.nb_cl])
            if 'permuted' in self.dataname:
                for idx in range(NUM_PERMUTATIONS - 1):
                    self.classes_by_groups.append(shuffled_labels)
            else:
                for idx in range(self.nb_cl, len(shuffled_labels), self.num_classes):
                    temp = []
                    for i in range(self.num_classes):
                        if (idx + i) < len(shuffled_labels):
                            temp.append(shuffled_labels[idx + i])
                    self.classes_by_groups.append(temp)
            self.num_tasks = len(self.classes_by_groups)
        grouped_data = [[] for _ in range(self.num_tasks)]
        print(f"Classes in each group: {self.classes_by_groups}")
        if 'permuted' in self.dataname:
            idx = list(range(28 * 28))
            for p_i in range(NUM_PERMUTATIONS):
                random.shuffle(idx)
                X_train_new, X_test_new = [[self.permutate_img_pixels(item, permutation=idx) for item in each] for each in
                                   [self.train_data, self.test_data]]
                data_to_consider = zip(X_train_new, self.train_labels) if train else zip(X_test_new, self.test_labels)
                group_ID = p_i
                for data, label in data_to_consider:
                    grouped_data[group_ID].append((data, self.label_map[label]))
        else:
            data_to_consider = zip(self.train_data, self.train_labels) if train else zip(self.test_data, self.test_labels)
            print(self.classes_by_groups)
            for data, label in data_to_consider:
                group_ID = [idx for idx in range(self.num_tasks) if label in self.classes_by_groups[idx]][0]
                # print(f"label: {label}, id: {group_ID}")
                grouped_data[group_ID].append((data, self.label_map[label]))
            if self.dataname == 'cifar100' and train:
                for group in grouped_data:
                    assert len(group) == 10000, len(group)
        return grouped_data

    def initialize(self):
        train_groups = self.get_data_by_groups(train=True)
        test_groups = self.get_data_by_groups(train=False)
        val_groups = [[] for i in range(self.num_tasks)]
        if self.keep_val:
            for i, train_group in enumerate(train_groups):
                _, labels = zip(*train_group)
                labels = np.array(labels)
                temp_train, temp_val = [], []
                for label in set(labels):
                    indices = np.where(labels == label)[0]
                    temp_val.extend([each for idx, each in enumerate(train_group) if idx in
                                     indices[(int)(TRAIN_VAL_SPLIT * len(indices)):]])
                    temp_train.extend([each for idx, each in enumerate(train_group) if idx in
                                       indices[:(int)(TRAIN_VAL_SPLIT * len(indices))]])
                train_groups[i] = temp_train
                val_groups[i] = temp_val
        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

    def getInputDim(self):
        return 32*32 if self.dataname == 'cifar100' else self.train_data[0].shape[0]