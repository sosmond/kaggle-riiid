import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit
from datetime import datetime, timedelta
from math import ceil

from .utils import (
    read_from_pickle, write_to_pickle, read_from_pickle_dict_chunks, write_to_pickle_dict_chunks, mahalanobis, pdist_sim,
    np_array, np_nans, np_absmean, np_mean, np_std, np_max, np_min, np_slope, arraylist, arraylist2D
)
    
#region ######################### FEATURIZER & BASE ACCUMULATOR ##########################

class Featurizer: 
    
    def __init__(self, accumulators=[], train_col_mappings={}, valid_col_mappings={}, keep_index=False):
        self.keep_index = keep_index

        if type(accumulators) == dict:
            self.accumulators = accumulators
        else:
            self.accumulators = {}
            for acc in accumulators:
                if acc.getname() == "RowIndex":
                    assert self.keep_index, "keep_index must be True in order to use RowIndex"
                if acc.level not in self.accumulators:
                    self.accumulators[acc.level] = []
                self.accumulators[acc.level].append(acc)

        self.accumulator_levels = sorted(list(self.accumulators.keys()))

        self.schema = {}
        self.feature_col_mappings = {}
        self.feature_count = 0
        for level in self.accumulator_levels:
            for acc in self.accumulators[level]:
                num_feats = len(acc.columns)
                self.schema[acc] = {'level': level, 'num_feats': num_feats, 'start_idx': self.feature_count, 'end_idx': self.feature_count + num_feats}
                self.feature_col_mappings.update({col: self.feature_count + i for i, col in enumerate(acc.columns)})
                self.feature_count += num_feats

        self.train_col_mappings = train_col_mappings
        self.valid_col_mappings = valid_col_mappings
        self.prev_rows = []
        
    def featurize(self, df, dataset="valid", update_only=False, verbose=True):
        assert dataset == "train" or dataset == "valid", f"allowed dataset values are 'train' or 'valid' but received {dataset}"

        if self.keep_index:
            df_v = df.reset_index().to_numpy() # retain row id for multiprocessing use
            df_columns = ["index"] + list(df.columns)
        else: 
            df_v = df.to_numpy()
            df_columns = df.columns

        feature_rows = np.zeros([df_v.shape[0], self.feature_count]) if not update_only else None
        feature_cache = {}

        # expects training set to be under the competition format
        # i.e. ordered by user_id and timestamp
        # for each row in df (important to avoid leaks): 
        # - if accumulator does not need answered_correctly or user_response, update accumulator first, then get features
        # - otherwise, get features first then update accumulators
        if dataset == "train":
            
            # generate col_mappings using the first df passed
            if not self.train_col_mappings: 
                self.train_col_mappings = {col: i for i, col in enumerate(df_columns)}

            frow = np.zeros(self.feature_count)

            for i, row in enumerate(tqdm(df_v) if verbose else df_v):
                frow[:] = 0
                for level in self.accumulator_levels:
                    for acc in self.accumulators[level]:
                        if not acc.req_answered_correctly and not acc.req_user_answer:
                            acc.update(row, feature_cache, self.train_col_mappings)
                            frow[self.schema[acc]['start_idx']:self.schema[acc]['end_idx']] = acc.get_features(row, feature_cache, self.train_col_mappings)
                        else:
                            frow[self.schema[acc]['start_idx']:self.schema[acc]['end_idx']] = acc.get_features(row, feature_cache, self.train_col_mappings)
                            acc.update(row, feature_cache, self.train_col_mappings)
                if not update_only:
                    feature_rows[i, :] = frow
        
        # expects a valid group under the competition format 
        # i.e. first row should have populated prior_group_responses and prior_group_answers_correct
        # steps:
        # 1. parse prior_group_answers_correct and prior_group_responses
        # 2. update accumulators that need prior_group_answers_correct and/or prior_group_responses
        # 3. for each row in df, update accumulators that don't need prior_ and get features
        # 4. save df rows so next iteration can add prior_group_answers_correct and prior_group_responses
        if dataset == "valid": 
            
            # generate col_mappings using the first df passed
            if not self.valid_col_mappings: 
                self.valid_col_mappings = {col: i for i, col in enumerate(df_columns)}
            
            # update accumulators that need prior_group_answers_correct and/or prior_group_responses
            pgac = eval(df_v[0, self.valid_col_mappings['prior_group_answers_correct']])
            pgr = eval(df_v[0, self.valid_col_mappings['prior_group_responses']])
            
            assert len(pgac) == len(self.prev_rows), f"pgac has {len(pgac)} entries but prev_rows has {len(self.prev_rows)} rows! Group num {df_v[0, self.valid_col_mappings['group_num']]}"
            
            for i, row in enumerate(self.prev_rows):
                row['data']['answered_correctly'] = pgac[i]
                row['data']['user_answer'] = pgr[i]
                for level in self.accumulator_levels:
                    for acc in self.accumulators[level]:
                        if acc.req_answered_correctly or acc.req_user_answer:
                            acc.get_features(row['data'], row['features'])
                            acc.update(row['data'], row['features'])
            
            self.prev_rows = []
            for i, row in enumerate(df_v):
                for level in self.accumulator_levels:
                    for acc in self.accumulators[level]:
                        if not acc.req_answered_correctly and not acc.req_user_answer:
                            acc.update(row, feature_cache, self.valid_col_mappings)
                        feature_rows[i, self.schema[acc]['start_idx']:self.schema[acc]['end_idx']] = acc.get_features(row, feature_cache, self.valid_col_mappings)
                self.prev_rows.append({'data': {col: row[idx] for col, idx in self.valid_col_mappings.items()}, 'features': feature_cache.copy()})
        
        return feature_rows

class Accumulator:
    """
    Base accumulator class
    """

    columns = []

    @classmethod
    def getname(cls):
        return cls.__name__

    def __init__(self, base_acc_dict, updatable, suffix):
        self.acc_dict = base_acc_dict
        self.updatable = updatable
        self.columns = [f"{col}{suffix}" for col in self.columns]

    def set_updatable(self, updatable):
        self.updatable = updatable

    def read(self, filename):
        self.acct_dict = read_from_pickle(filename)

    def save(self, filename):
        write_to_pickle(self.acc_dict, filename)

#endregion
#region ######################### LEVEL 0 ACCUMULATORS ##########################

class RowIndex(Accumulator):
    """
    Provides the index/row id for parallel processing purposes. Not a real accumulator and not to be used as a real feature!

    Features:
    - index: row id 
    """
    level = 0
    req_answered_correctly = False
    req_user_answer = False
    columns = ['index']

    def __init__(self):
        super().__init__({}, False, "")
    
    def update(self, row, feature_cache, cm={}):
        pass # don't update since not a real accumulator

    def get_features(self, row, feature_cache, cm={}):
        index = row[cm['index']] if cm else row['index']
        features = np_array(index)

        return features

class MetadataFeats(Accumulator):
    """
    Provides some metadata features (e.g. question part).

    Accumulates By: NA

    Features:
    - tag1: first tag
    - part: the relevant section of the TOEIC test 
    - bundle_id_count: number of questions in the bundle
    """

    level = 0
    req_answered_correctly = False
    req_user_answer = False
    columns = ['tag1', 'part', 'bundle_id_count']

    # TODO: add lecture df and provide metadata
    def __init__(self, questions_df):
        # expects a valid questions_df with the following cm format:
        bundle_id_counts = questions_df['bundle_id'].value_counts()
        questions_df['bundle_id_count'] = [bundle_id_counts[bundle_id] for bundle_id in questions_df['bundle_id']]
        cm = {'question_id': 0, 'bundle_id': 1, 'correct_answer': 2, 'part': 3, 'tags': 4, 'bundle_id_count': 5}

        self.acc_dict = {}
        for row in questions_df.values: 
            question_id = row[cm['question_id']]
            bundle_id = row[cm['bundle_id']]
            correct_answer = row[cm['correct_answer']]
            part = row[cm['part']]

            tags = row[cm['tags']]
            if pd.isnull(tags):
                tags = '0'

            bundle_id_count = row[cm['bundle_id_count']]
            self.acc_dict[question_id] = {'bundle_id': bundle_id, 'correct_answer': correct_answer, 'part': part, 'tags': tags, 'bundle_id_count': bundle_id_count}
        
        self.updatable = False

    def update(self, row, feature_cache, cm={}):
        pass # don't update since questions.csv is the same in train and test

    def get_features(self, row, feature_cache, cm={}):
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        
        feature_cache['tag1'] = np.nan
        feature_cache['part'] = np.nan
        feature_cache['bundle_id'] = np.nan
        feature_cache['bundle_id_count'] = np.nan
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_content = self.acc_dict.get(content_id, None)

            if acc_content is not None:
                tag1 = int(acc_content['tags'].split(' ', 1)[0])
                part = acc_content['part'] - 1
                bundle_id = acc_content['bundle_id']
                bundle_id_count = acc_content['bundle_id_count']

                feature_cache['tag1'] = tag1
                feature_cache['part'] = part
                feature_cache['bundle_id'] = bundle_id
                feature_cache['bundle_id_count'] = bundle_id_count
                features = np_array(
                    tag1,
                    part,
                    bundle_id_count
                )

        return features

class QAccuracy(Accumulator): 
    """
    Provides features capturing the historical performance across every response for a given question content_id.

    Accumulates By: content_id

    Features:
    - question_accuracy: the average answered_correctly across every attempt on the given question content_id
    - question_responses: the total number of responses to the given question content_id
    """

    level = 0
    req_answered_correctly = True
    req_user_answer = False
    columns = ['question_accuracy', 'question_responses']

    def __init__(self, base_acc_dict=None, w_smooth=20, updatable=True, suffix=""):
        self.w_smooth = w_smooth
        self.all_answered_correctly = 0
        self.all_responses = 0

        if base_acc_dict is not None:
            self.all_answered_correctly = sum([v['answered_correctly'] for k, v in base_acc_dict.items()])
            self.all_responses = sum([v['total'] for k, v in base_acc_dict.items()])

        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']

        if content_type_id == 0:
            if content_id not in self.acc_dict:
                self.acc_dict[content_id] = {'answered_correctly': 0, 'total': 0}
            self.acc_dict[content_id]['answered_correctly'] += answered_correctly
            self.acc_dict[content_id]['total'] += 1
            self.all_answered_correctly += answered_correctly
            self.all_responses += 1
    
    def get_features(self, row, feature_cache, cm={}):
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']

        feature_cache['question_accuracy'] = np.nan
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_content = self.acc_dict.get(content_id, None)
            if acc_content is not None:
                total = acc_content['total']
                overall_accuracy = self.all_answered_correctly/self.all_responses if self.all_responses > 0 else np.nan
                question_accuracy = (acc_content['answered_correctly'] + self.w_smooth*overall_accuracy)/(total + self.w_smooth)

                feature_cache['question_accuracy'] = question_accuracy
                features = np_array(
                    question_accuracy,
                    total
                )

        return features

#endregion
#region ######################### LEVEL 1 ACCUMULATORS ##########################

class PebgFeatures(Accumulator):
    """
    Provides features from pretrained question embedding using PEBG.

    Accumulates By: user_id

    Features:
    - curr_pebg_{i}: pebg embedding of the current question (1 feature per element in embedding)
    - mean_pebg_{i}: mean of pebg embedding (1 feature per element in embedding)
    - max_pebg_{i}: max of pebg embedding (1 feature per element in embedding)

    Note:
    - min_pebg_{i} not useful since it's usually 0 for every feature
    """

    level = 1
    req_answered_correctly = False
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, pebg_embed_matrix, base_acc_dict=None, updatable=True, suffix=""):
        self.pebg_embed_dict = {i: q_embed for i, q_embed in enumerate(pebg_embed_matrix)}
        self.pebg_embed_size = pebg_embed_matrix.shape[1]
        self.columns = (
            [f"curr_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            [f"mean_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            [f"max_pebg_{i}" for i in range(self.pebg_embed_size)]
        )
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'sum_pebg': np.zeros(self.pebg_embed_size), 
                                          'max_pebg': np.zeros(self.pebg_embed_size), 
                                          'total': 0}
            
            pebg_embed = self.pebg_embed_dict[content_id]
            self.acc_dict[user_id]['sum_pebg'] += pebg_embed
            self.acc_dict[user_id]['max_pebg'] = np.maximum(self.acc_dict[user_id]['max_pebg'], pebg_embed)
            self.acc_dict[user_id]['total'] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            curr_pebg = self.pebg_embed_dict[content_id]

            total = self.acc_dict[user_id]['total']
            mean_pebg = self.acc_dict[user_id]['sum_pebg']/total if total > 0 else np_nans(self.pebg_embed_size)
            max_pebg = self.acc_dict[user_id]['max_pebg']
            
            features[:self.pebg_embed_size] = curr_pebg
            features[self.pebg_embed_size:self.pebg_embed_size*2] = mean_pebg
            features[self.pebg_embed_size*2:] = max_pebg

        return features

class PebgCorrect(Accumulator):
    """
    Provides PEBG features that compare the current question embedding to accumulated embeddings where the user was correct/incorrect.

    Accumulates By: user_id

    Features:
    - mean_correct_pebg_{i}: mean of pebg embedding for correct answers
    - mean_incorrect_pebg_{i}: mean of pebg embedding for incorrect answers
    - pebg_correct_distance: mahalanobis distance between curr_pebg and correct_pebg
    - pebg_incorrect_distance: mahalanobis distance between curr_pebg and incorrect_pebg 
    - pebg_correct_distance_ratio: ratio between pebg_correct_distance and pebg_incorrect_distance
    """

    level = 1
    req_answered_correctly = True
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, pebg_embed_matrix, base_acc_dict=None, updatable=True, suffix=""):
        self.pebg_embed_dict = {i: q_embed for i, q_embed in enumerate(pebg_embed_matrix)}
        self.pebg_embed_size = pebg_embed_matrix.shape[1]
        self.VI = np.linalg.pinv(np.cov(pebg_embed_matrix.T))
        self.columns = (
            [f"mean_correct_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            [f"mean_incorrect_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            ['pebg_correct_distance', 'pebg_incorrect_distance', 'pebg_correct_distance_ratio']
        )
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {
                    'sum_correct_pebg': np.zeros(self.pebg_embed_size), 
                    'sum_incorrect_pebg': np.zeros(self.pebg_embed_size), 
                    'correct_total': 0,
                    'incorrect_total': 0
                }
            
            pebg_embed = self.pebg_embed_dict[content_id]
            if answered_correctly:
                self.acc_dict[user_id]['sum_correct_pebg'] += pebg_embed
                self.acc_dict[user_id]['correct_total'] += 1
            else:   
                self.acc_dict[user_id]['sum_incorrect_pebg'] += pebg_embed
                self.acc_dict[user_id]['incorrect_total'] += 1 

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:                
                curr_pebg = self.pebg_embed_dict[content_id]

                correct_total = acc_user['correct_total']
                incorrect_total = acc_user['incorrect_total']
                mean_correct_pebg = acc_user['sum_correct_pebg']/correct_total if correct_total > 0 else np_nans(self.pebg_embed_size)
                mean_incorrect_pebg = acc_user['sum_incorrect_pebg']/incorrect_total if incorrect_total > 0 else np_nans(self.pebg_embed_size)
                pebg_correct_distance = mahalanobis(curr_pebg, mean_correct_pebg, self.VI) if correct_total > 0 else np.nan
                pebg_incorrect_distance = mahalanobis(curr_pebg, mean_incorrect_pebg, self.VI) if incorrect_total > 0 else np.nan
                
                features[:self.pebg_embed_size] = mean_correct_pebg
                features[self.pebg_embed_size:self.pebg_embed_size*2] = mean_incorrect_pebg
                features[self.pebg_embed_size*2] = pebg_correct_distance
                features[self.pebg_embed_size*2+1] = pebg_incorrect_distance
                features[self.pebg_embed_size*2+2] = pebg_correct_distance/pebg_incorrect_distance if pebg_incorrect_distance > 0 else np.nan

        return features

class RealTimestampFeats(Accumulator):
    """
    Provides features related to the real timestamp of the current question.
    Accumulates By: user_id
    Features:
    - real_timestamp_hour: real timestamp hour of the day
    - real_timestamp_day: real timestamp day of the month
    - real_timestamp_weekday: real timestamp day of the week
    - real_timestamp_weekofmonth: real timestamp week of the month
    """

    level = 1
    req_answered_correctly = False
    req_user_answer = False
    columns = ['real_timestamp_hour', 'real_timestamp_day', 'real_timestamp_weekday', 'real_timestamp_weekofmonth']

    def __init__(self, user_start_time_dict, base_acc_dict=None, suffix=""):
        self.user_start_time_dict = user_start_time_dict
        super().__init__(base_acc_dict or {}, False, suffix)
    
    def update(self, row, feature_cache, cm={}):
        pass

    @staticmethod
    def week_of_month(dt):
        first_day = dt.replace(day=1)

        dom = dt.day
        adjusted_dom = dom + first_day.weekday()

        return int(ceil(adjusted_dom / 7.0))

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            if user_id in self.user_start_time_dict:
                real_timestamp = self.user_start_time_dict[user_id] + timestamp
                real_datetime = datetime.utcfromtimestamp(real_timestamp/1000) + timedelta(hours=9)

                features = np_array(
                    real_datetime.hour,
                    real_datetime.day,
                    real_datetime.weekday(),
                    self.week_of_month(real_datetime)
                )

        return features

class TempTimestampFeats(Accumulator):
    """
    Use this to generate features for elapsed time model. Not to be used for real featurizing.

    Accumulates By: user_id
    
    Requires:
    - bundle_id_count: from MetadataFeats

    Features:
    - prior_question_elapsed_time: average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between
    - prior_question_had_explanation: Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between
    - avg_elapsed_time: average of all the user's prior_question_elapsed_time
    - prop_question_had_explanation: % of user's questions that had explanation 
    - timestamp: timestamp of the current row
    - time_since_last_action: the difference between current and previous timestamp for a given user_id
    - time_since_last_action2: the difference between the n-1 and n-2 timestamps for a given user_id, where n is the index of the current question
    - prior_question_down_time: average time in milliseconds the user does not spend time answering questions in the bundle, defined as the difference between lag and elapsed time
    - prior_down_to_elapsed: ratio of down time to elapsed time for the previous question
    - avg_prior_down_time: average of all the user's prior_question_down_time
    - avg_prior_down_to_elapsed: ratio of avg_prior_down_time to avg_elapsed_time
    """

    level = 1
    req_answered_correctly = True
    req_user_answer = False
    columns = [
        'prior_question_elapsed_time', 'prior_question_had_explanation', 'avg_elapsed_time', 'prop_question_had_explanation',
        'time_since_last_action',
        'prior_question_down_time', 'prior_elapsed_to_tsla', 'avg_prior_down_time', 'avg_prior_elapsed_to_tsla',
        'qaccuracy', 'answered_correctly'
    ]

    def __init__(self, questions_df, qaccuracy, base_acc_dict=None, updatable=True, suffix=""):
        bundle_id_counts = questions_df['bundle_id'].value_counts()
        questions_df['bundle_id_count'] = [bundle_id_counts[bundle_id] for bundle_id in questions_df['bundle_id']]
        cm = {'question_id': 0, 'bundle_id': 1, 'correct_answer': 2, 'part': 3, 'tags': 4, 'bundle_id_count': 5}

        self.ref_dict = {}
        for row in questions_df.values: 
            question_id = row[cm['question_id']]
            bundle_id = row[cm['bundle_id']]
            correct_answer = row[cm['correct_answer']]
            part = row[cm['part']]

            tags = row[cm['tags']]
            if pd.isnull(tags):
                tags = '0'

            bundle_id_count = row[cm['bundle_id_count']]
            self.ref_dict[question_id] = {'bundle_id': bundle_id, 'correct_answer': correct_answer, 'part': part, 'tags': tags, 'bundle_id_count': bundle_id_count}

        self.qaccuracy = qaccuracy
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']

        prior_elapsed_time_exists = prior_question_elapsed_time > 0 and prior_question_elapsed_time != np.inf

        if user_id not in self.acc_dict:
            self.acc_dict[user_id] = {
                'total_prior_elapsed_time': 0, 'total_prior_had_explanation': 0, 'total_prior': 0, 'max_prior_elapsed_time': 0, 'min_prior_elapsed_time': np.inf,
                'timestamp': 0, 'time_since_last_action': 0,
                'prior_question_down_time': 0, 'total_prior_down_time': 0, 'down_total': 0
            }

        acc_user = self.acc_dict[user_id]

        if content_type_id == 0:
            bundle_id_count = self.ref_dict[content_id]['bundle_id_count']

            if prior_elapsed_time_exists: # check for prior_question_elapsed_time == np.nan or np.inf
                acc_user['total_prior_elapsed_time'] += prior_question_elapsed_time 

                if prior_question_elapsed_time > acc_user['max_prior_elapsed_time']:
                    acc_user['max_prior_elapsed_time'] = prior_question_elapsed_time
                if prior_question_elapsed_time < acc_user['min_prior_elapsed_time']:
                    acc_user['min_prior_elapsed_time'] = prior_question_elapsed_time

            if prior_question_had_explanation > 0:
                acc_user['total_prior_had_explanation'] += 1
            if pd.notnull(prior_question_elapsed_time) and pd.notnull(prior_question_had_explanation):
                acc_user['total_prior'] += 1

            if timestamp > acc_user['timestamp']:
                # before they're updated, time_since_last_action and timestamp hold previous values
                prior_question_down_time = acc_user['time_since_last_action'] - prior_question_elapsed_time if prior_elapsed_time_exists else 0

                # if user switches bundles part way through, this still makes sense since they were shown all the questions at once in the previous bundle
                acc_user['time_since_last_action'] = (timestamp - acc_user['timestamp'])/bundle_id_count if bundle_id_count > 0 else 0
                
                if prior_question_down_time > 0:
                    acc_user['prior_question_down_time'] = prior_question_down_time

                    if prior_question_down_time <= 300000:
                        acc_user['total_prior_down_time'] += prior_question_down_time
                        acc_user['down_total'] += 1
                else:
                    acc_user['prior_question_down_time'] = 0
        
        acc_user['timestamp'] = timestamp

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']

        prior_elapsed_time_exists = prior_question_elapsed_time > 0 and prior_question_elapsed_time != np.inf
        
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                total_prior = acc_user['total_prior']
                down_total = acc_user['down_total']
                avg_elapsed_time = acc_user['total_prior_elapsed_time']/total_prior if total_prior > 0 else np.nan
                prior_question_down_time = acc_user['prior_question_down_time']
                avg_prior_down_time = acc_user['total_prior_down_time']/down_total if down_total > 0 else np.nan
                time_since_last_action = acc_user['time_since_last_action']
                qacc = self.qaccuracy[content_id]

                features = np_array(
                    prior_question_elapsed_time, 
                    prior_question_had_explanation, 
                    avg_elapsed_time, 
                    acc_user['total_prior_had_explanation']/total_prior if total_prior > 0 else np.nan,
                    time_since_last_action,
                    prior_question_down_time,
                    prior_question_elapsed_time/(prior_question_elapsed_time + prior_question_down_time) if prior_elapsed_time_exists else np.nan, # prior_down_to_elapsed
                    avg_prior_down_time,
                    avg_elapsed_time/(avg_elapsed_time + avg_prior_down_time) if avg_elapsed_time > 0 else np.nan,
                    qacc['answered_correctly']/qacc['total'] if qacc['total'] > 0 else np.nan,
                    answered_correctly
                )

        return features

class PriorTimestampFeats(Accumulator):
    """
    Provides some features about the previous question, capturing the breakdown of time since last action into elapsed and down times.
    TODO: check if excluding lectures is better

    Accumulates By: user_id
    
    Requires:
    - bundle_id_count: from MetadataFeats
    - question_accuracy: from QAccuracy

    Features:
    - prior_question_elapsed_time: average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between
    - prior_question_had_explanation: Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between
    - avg_elapsed_time: average of all the user's prior_question_elapsed_time
    - prop_question_had_explanation: % of user's questions that had explanation 
    - timestamp: timestamp of the current row
    - time_since_last_action: the difference between current and previous timestamp for a given user_id
    - time_since_last_action2: the difference between the n-1 and n-2 timestamps for a given user_id, where n is the index of the current question
    - prior_question_down_time: average time in milliseconds the user does not spend time answering questions in the bundle, defined as the difference between lag and elapsed time
    - prior_down_to_elapsed: ratio of down time to elapsed time for the previous question
    - avg_prior_down_time: average of all the user's prior_question_down_time
    - avg_prior_down_to_elapsed: ratio of avg_prior_down_time to avg_elapsed_time
    """

    level = 1
    req_answered_correctly = False
    req_user_answer = False
    columns = [
        'prior_question_elapsed_time', 'prior_question_had_explanation', 'avg_elapsed_time', 'prop_question_had_explanation',
        'timestamp', 'time_since_last_action', 'time_since_last_action2', 'bundle_idx',
        'prior_question_down_time', 'prior_down_to_elapsed', 'avg_prior_down_time', 'avg_prior_down_to_elapsed'
    ]

    def __init__(self, question_elapsed_time_dict, elapsed_time_model_dict, base_acc_dict=None, updatable=True, suffix=""):
        self.bundle_idx = 1
        self.question_elapsed_time_dict = question_elapsed_time_dict
        self.elapsed_time_model_dict = elapsed_time_model_dict
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']
        bundle_id_count = feature_cache['bundle_id_count']

        prior_elapsed_time_exists = prior_question_elapsed_time > 0

        if user_id not in self.acc_dict:
            self.acc_dict[user_id] = {
                'prior_question_elapsed_time': np.nan, 'prior_question_had_explanation': 0, 'prior_question_content_id': np.nan,
                'total_prior_elapsed_time': 0, 'total_prior_had_explanation': 0, 'total_prior': 0, 
                'timestamp': 0, 'time_since_last_action': 0, 'time_since_last_action2': 0, 
                'prior_question_down_time': 0, 'total_prior_down_time': 0, 'down_total': 0
            }

        acc_user = self.acc_dict[user_id]

        if content_type_id == 0:
            pqet = np.nan

            if prior_question_elapsed_time >= 65536: # predict elapsed time using model
                total_prior = acc_user['total_prior']
                prev_pqet = acc_user['prior_question_elapsed_time']
                prev_pqdt = acc_user['prior_question_down_time']
                avg_pqet = acc_user['total_prior_elapsed_time']/total_prior if total_prior > 0 else np.nan
                avg_pqdt = acc_user['total_prior_down_time']/acc_user['down_total'] if acc_user['down_total'] > 0 else np.nan
                prior_elapsed_to_tsla = prev_pqet/(prev_pqet + prev_pqdt) if prev_pqet + prev_pqdt > 0 else np.nan
                avg_prior_elapsed_to_tsla = avg_pqet/(avg_pqet + avg_pqdt) if avg_pqet + avg_pqdt > 0 else np.nan
                prop_pqhe = acc_user['total_prior_had_explanation']/total_prior if total_prior > 0 else np.nan
                question_accuracy = feature_cache['question_accuracy']
                prev_cid = acc_user['prior_question_content_id']
                content_elapsed_mean = self.question_elapsed_time_dict[int(prev_cid)] if prev_cid >= 0 else np.nan

                pqet = (
                    self.elapsed_time_model_dict['prior_question_elapsed_time']*prev_pqet + 
                    self.elapsed_time_model_dict['prior_question_had_explanation']*acc_user['prior_question_had_explanation'] + 
                    self.elapsed_time_model_dict['avg_elapsed_time']*avg_pqet + 
                    self.elapsed_time_model_dict['prop_question_had_explanation']*prop_pqhe + 
                    self.elapsed_time_model_dict['time_since_last_action']*acc_user['time_since_last_action'] + 
                    self.elapsed_time_model_dict['prior_question_down_time']*prev_pqdt + 
                    self.elapsed_time_model_dict['prior_elapsed_to_tsla']*prior_elapsed_to_tsla + 
                    self.elapsed_time_model_dict['avg_prior_down_time']*avg_pqdt + 
                    self.elapsed_time_model_dict['avg_prior_elapsed_to_tsla']*avg_prior_elapsed_to_tsla + 
                    self.elapsed_time_model_dict['qaccuracy']*question_accuracy + 
                    self.elapsed_time_model_dict['content_elapsed_mean']*content_elapsed_mean
                )

                if pqet == np.nan:
                    guess = acc_user['time_since_last_action']*prior_elapsed_to_tsla
                    pqet = min(content_elapsed_mean, guess if guess > 0 else content_elapsed_mean)
            else:
                pqet = prior_question_elapsed_time

            if prior_elapsed_time_exists: # check for prior_question_elapsed_time == np.nan or np.inf
                acc_user['total_prior_elapsed_time'] += pqet
            if prior_question_had_explanation > 0:
                acc_user['total_prior_had_explanation'] += 1
            if pd.notnull(prior_question_elapsed_time) and pd.notnull(prior_question_had_explanation):
                acc_user['total_prior'] += 1

            if timestamp > acc_user['timestamp']:
                self.bundle_idx = 1
                acc_user['prior_question_elapsed_time'] = pqet
                acc_user['prior_question_had_explanation'] = prior_question_had_explanation
                acc_user['prior_question_content_id'] = content_id
                # before they're updated, time_since_last_action and timestamp hold previous values
                prior_question_down_time = acc_user['time_since_last_action'] - pqet if prior_elapsed_time_exists else 0

                acc_user['time_since_last_action2'] = acc_user['time_since_last_action']

                # if user switches bundles part way through, this still makes sense since they were shown all the questions at once in the previous bundle
                acc_user['time_since_last_action'] = (timestamp - acc_user['timestamp'])/bundle_id_count if bundle_id_count > 0 else 0
                
                if prior_question_down_time > 0:
                    acc_user['prior_question_down_time'] = prior_question_down_time

                    if prior_question_down_time <= 300000:
                        acc_user['total_prior_down_time'] += prior_question_down_time
                        acc_user['down_total'] += 1
                else:
                    acc_user['prior_question_down_time'] = 0
            else:
                if pqet >= 0: # make sure it's not the first row which has np.nan prior elapsed time
                    self.bundle_idx += 1
                else:
                    self.bundle_idx = 1
        
        acc_user['timestamp'] = timestamp

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        
        feature_cache['bundle_idx'] = np.nan
        feature_cache['time_since_last_action'] = np.nan
        feature_cache['prior_question_elapsed_time'] = np.nan
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                prior_question_elapsed_time = acc_user['prior_question_elapsed_time']
                prior_question_had_explanation = acc_user['prior_question_had_explanation']
                prior_elapsed_time_exists = prior_question_elapsed_time > 0

                total_prior = acc_user['total_prior']
                down_total = acc_user['down_total']
                avg_elapsed_time = acc_user['total_prior_elapsed_time']/total_prior if total_prior > 0 else np.nan
                prior_question_down_time = acc_user['prior_question_down_time']
                avg_prior_down_time = acc_user['total_prior_down_time']/down_total if down_total > 0 else np.nan
                time_since_last_action = acc_user['time_since_last_action']
                bundle_idx = self.bundle_idx

                feature_cache['bundle_idx'] = bundle_idx
                feature_cache['time_since_last_action'] = time_since_last_action
                feature_cache['prior_question_elapsed_time'] = prior_question_elapsed_time
                features = np_array(
                    prior_question_elapsed_time, 
                    prior_question_had_explanation, 
                    avg_elapsed_time, 
                    acc_user['total_prior_had_explanation']/total_prior if total_prior > 0 else np.nan,
                    acc_user['timestamp'],
                    time_since_last_action,
                    acc_user['time_since_last_action2'],
                    bundle_idx,
                    prior_question_down_time,
                    prior_question_down_time/prior_question_elapsed_time if prior_elapsed_time_exists else np.nan, # prior_down_to_elapsed
                    avg_prior_down_time,
                    avg_prior_down_time/avg_elapsed_time if avg_elapsed_time > 0 else np.nan
                )

        return features

class UAccuracy(Accumulator):
    """
    Provides features capturing a given user's historical performance.

    Accumulates By: user_id

    Requires:
    - part: from MetadataFeats

    Features:
    - user_accuracy: the average answered_correctly for a given user_id, ignoring lectures
    - user_responses: the number of questions a given user_id answered (does not include current question)
    """

    level = 1
    req_answered_correctly = True
    req_user_answer = False
    columns = ['user_accuracy', 'user_correct', 'user_responses', 
               'user_part0_accuracy', 'user_part0_responses',
               'user_part1_accuracy', 'user_part1_responses',
               'user_part2_accuracy', 'user_part2_responses',
               'user_part3_accuracy', 'user_part3_responses',
               'user_part4_accuracy', 'user_part4_responses',
               'user_part5_accuracy', 'user_part5_responses',
               'user_part6_accuracy', 'user_part6_responses'
              ]

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        part = feature_cache['part']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'answered_correctly': 0, 'total': 0, 
                                          'part0_correct': 0, 'part0_total': 0,
                                          'part1_correct': 0, 'part1_total': 0,
                                          'part2_correct': 0, 'part2_total': 0,
                                          'part3_correct': 0, 'part3_total': 0,
                                          'part4_correct': 0, 'part4_total': 0,
                                          'part5_correct': 0, 'part5_total': 0,
                                          'part6_correct': 0, 'part6_total': 0,
                                          }
            self.acc_dict[user_id]['answered_correctly'] += answered_correctly
            self.acc_dict[user_id]['total'] += 1

            if part == 0:
                self.acc_dict[user_id]['part0_correct'] += answered_correctly
                self.acc_dict[user_id]['part0_total'] += 1
            elif part == 1:
                self.acc_dict[user_id]['part1_correct'] += answered_correctly
                self.acc_dict[user_id]['part1_total'] += 1
            elif part == 2:
                self.acc_dict[user_id]['part2_correct'] += answered_correctly
                self.acc_dict[user_id]['part2_total'] += 1
            elif part == 3:
                self.acc_dict[user_id]['part3_correct'] += answered_correctly
                self.acc_dict[user_id]['part3_total'] += 1
            elif part == 4:
                self.acc_dict[user_id]['part4_correct'] += answered_correctly
                self.acc_dict[user_id]['part4_total'] += 1
            elif part == 5:
                self.acc_dict[user_id]['part5_correct'] += answered_correctly
                self.acc_dict[user_id]['part5_total'] += 1
            elif part == 6:
                self.acc_dict[user_id]['part6_correct'] += answered_correctly
                self.acc_dict[user_id]['part6_total'] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        feature_cache['user_responses'] = np.nan
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                user_correct = acc_user['answered_correctly']
                total = acc_user['total']
                part0_total = acc_user['part0_total']
                part1_total = acc_user['part1_total']
                part2_total = acc_user['part2_total']
                part3_total = acc_user['part3_total']
                part4_total = acc_user['part4_total']
                part5_total = acc_user['part5_total']
                part6_total = acc_user['part6_total']

                feature_cache['user_responses'] = total
                features = np_array(
                    user_correct/total,
                    user_correct,
                    total,
                    acc_user['part0_correct']/part0_total if part0_total > 0 else np.nan,
                    part0_total,
                    acc_user['part1_correct']/part1_total if part1_total > 0 else np.nan,
                    part1_total,
                    acc_user['part2_correct']/part2_total if part2_total > 0 else np.nan,
                    part2_total,
                    acc_user['part3_correct']/part3_total if part3_total > 0 else np.nan,
                    part3_total,
                    acc_user['part4_correct']/part4_total if part4_total > 0 else np.nan,
                    part4_total,
                    acc_user['part5_correct']/part5_total if part5_total > 0 else np.nan,
                    part5_total,
                    acc_user['part6_correct']/part6_total if part6_total > 0 else np.nan,
                    part6_total
                )

        return features

class PreviousQTryAccum(Accumulator):
    """
    Provides features capturing the last answered_correctly for a given user_id on a given question content_id

    Accumulates By: user_id

    Requires:
    - bundle_id: from MetadataFeats

    Features:
    - previous_try: previous answered_correctly for a given user_id on a given question content_id
    - num_prev_tries: number of previous tries for the user on the question
    - prev_try_accuracy: accuracy of the user on previous tries for the question
    - question_had_explanation: whether the user ever saw the explanation on previous attempts of the question

    Note: 
    - large accumulator dict (3.2GB for 100M+ rows in the train set) if mapping user_id's to all of the content_id's they attempted directly
    - instead, map user_id to array of content_id's, with negative content_id's representing past incorrects and positive representing past correct attempts
    - further, encode user's number of attempts for the question using multiple of 100k and number of correct attemps using the decimal - takes no extra space 
      - use 1/10000 as decimal count for correct attempts since it's unlikely any user will attempt the same question 10k times (never happens in train set)
    """
    
    level = 1
    req_answered_correctly = True
    req_user_answer = False
    columns = ['previous_try', 'num_prev_tries', 'prev_try_accuracy', 'question_had_explanation', 'time_since_prev_try']

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        self.user_id = np.nan
        self.bundle_id = np.nan
        self.timestamp = np.nan
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def update_content_id_list(user_history_arr, user_timestamp_arr, content_id, answered_correctly, timestamp):
        for i in range(len(user_history_arr)):
            user_history_arr_i = abs(user_history_arr[i])
            cid = int(user_history_arr_i % 100000)

            if cid == content_id:
                num_responses = int(user_history_arr_i // 100000)
                num_correct = round(user_history_arr_i % 1 * 10000)
                
                if answered_correctly:
                    user_history_arr[i] = 100000*(num_responses+1) + content_id + (num_correct+1)/10000
                else:
                    user_history_arr[i] = -(100000*(num_responses+1) + content_id + num_correct/10000)
                
                user_timestamp_arr[i] = timestamp
                return True
        return False

    @staticmethod
    @njit
    def update_content_id_pqhe(user_history_arr, prior_questions, pqhe):
        len_prior_questions = len(prior_questions)
        
        if pqhe == 0.1 and len_prior_questions > 0:
            for i in range(len_prior_questions):
                content_id = prior_questions[i]
                for j in range(len(user_history_arr)):
                    user_history_arr_j = abs(user_history_arr[j])
                    cid = int(user_history_arr_j % 100000) 

                    if cid == content_id:
                        prev_pqhe = round(round(user_history_arr_j, 1) % 1 * 10)
                        if prev_pqhe == 0:
                            if user_history_arr[j] < 0:
                                user_history_arr[j] -= 0.1
                            else: 
                                user_history_arr[j] += 0.1
    @staticmethod
    @njit
    def get_prev_qtry(user_history_arr, user_timestamp_arr, content_id):
        for i in range(len(user_history_arr)):
            user_history_arr_i = abs(user_history_arr[i])
            cid = int(user_history_arr_i % 100000)

            if cid == content_id:
                num_responses = int(user_history_arr_i // 100000)
                num_correct = round(user_history_arr_i % 0.1 * 10000)
                qhe = round(round(user_history_arr_i, 1) % 1 * 10)

                if user_history_arr[i] < 0:
                    return 0, num_responses, num_correct/num_responses, qhe, user_timestamp_arr[i]
                else: 
                    return 1, num_responses, num_correct/num_responses, qhe, user_timestamp_arr[i]
        return np.nan, np.nan, np.nan, np.nan, np.nan

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        bundle_id = feature_cache['bundle_id']
        
        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'user_history_arr': arraylist(), 'user_timestamp_arr': arraylist(), 'prior_questions': []}

            pqhe = 0.1 if prior_question_had_explanation > 0 else 0.0
            acc_user = self.acc_dict[user_id]
            user_history_arr = acc_user['user_history_arr']
            user_timestamp_arr = acc_user['user_timestamp_arr']
            prior_questions = acc_user['prior_questions']
            if not self.update_content_id_list(user_history_arr.data, user_timestamp_arr.data, content_id, answered_correctly, timestamp):
                user_history_arr.append(100000.0 + content_id + 0.0001 if answered_correctly else -(100000.0 + content_id))
                user_timestamp_arr.append(timestamp)

            if user_id == self.user_id and bundle_id == self.bundle_id and timestamp == self.timestamp:
                prior_questions.append(content_id)
            else:
                self.update_content_id_pqhe(user_history_arr.data, tuple(prior_questions), pqhe)
                prior_questions.clear()
                prior_questions.append(content_id)

            self.user_id = user_id
            self.bundle_id = bundle_id
            self.timestamp = timestamp
        
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']
        
        feature_cache['num_prior_questions'] = np.nan # can't provide prior_questions since it's passed by reference, and when update is called it changes
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            if acc_user is not None:
                feature_cache['num_prior_questions'] = len(acc_user['prior_questions'])

                previous_try, num_prev_tries, prev_try_accuracy, question_had_explanation, prev_timestamp = self.get_prev_qtry(acc_user['user_history_arr'].data, acc_user['user_timestamp_arr'].data, content_id)
                features = np_array(
                    previous_try, 
                    num_prev_tries, 
                    prev_try_accuracy,
                    question_had_explanation,
                    timestamp - prev_timestamp
                )

        return features

    def compress(self):
        for _, acc_user in tqdm(self.acc_dict.items()):
            v = acc_user['user_history_arr']
            v.data = v.data[~np.isnan(v.data)]

            new_size = len(v.data)
            v.size = new_size
            v.capacity = new_size

    # override read and save for this accumulator to save memory overhead since the acc_dict is large
    def read(self, filename):
        self.acct_dict = read_from_pickle_dict_chunks(filename)

    def save(self, filename):
        write_to_pickle_dict_chunks(self.acc_dict, filename)

class LectureFeats(Accumulator):
    """
    Provides features relating to lectures the user has watched

    Accumulates By: user_id

    Features:
    #- concept_pebg_{i}: sum of pebg skill embedding for the user for lectures with type "concept" 
    #- intention_pebg_{i}: sum of pebg skill embedding for the user for lectures with type "intention" 
    #- solving_question_pebg_{i}: sum of pebg skill embedding for the user for lectures with type "solving question"
    - part_{j}: number of lectures the user watched under part j = {1,2,3,4,5,6,7}
    - concept_total: number of lectures the user watched with type "concept"
    - intention_total: number of lectures the user watched with type "intention"
    - solving_question_total: number of lectures the user watched with type "solving question"

    Note:
    - Starter lectures not included since only 3 samples in all of train
    """
    
    level = 1
    req_answered_correctly = False
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, lectures_df, pebg_skill_repre, skill_id_dict, base_acc_dict=None, updatable=True, suffix=""):
        self.pebg_skill_repre = pebg_skill_repre
        self.pebg_skill_size = pebg_skill_repre.shape[1]
        self.skill_id_dict = skill_id_dict
        self.lectures_dict = lectures_df.set_index('lecture_id').to_dict('index')
        self.columns = ([f"concept_pebg_{i}" for i in range(self.pebg_skill_size)] + 
                        [f"intention_pebg_{i}" for i in range(self.pebg_skill_size)] + 
                        [f"solving_question_pebg_{i}" for i in range(self.pebg_skill_size)] + 
                        ['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7'] + 
                        ['concept_total', 'intention_total', 'solving_question_total']
                       )
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return
        
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        if content_type_id == 1:
            user_id = row[cm['user_id']] if cm else row['user_id']        
            content_id = row[cm['content_id']] if cm else row['content_id']
            lecture_info = self.lectures_dict[content_id]

            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'concept': np.zeros(self.pebg_skill_size), 
                                          'intention': np.zeros(self.pebg_skill_size),
                                          'solving question': np.zeros(self.pebg_skill_size),
                                          'part_1': 0,
                                          'part_2': 0,
                                          'part_3': 0,
                                          'part_4': 0,
                                          'part_5': 0,
                                          'part_6': 0,
                                          'part_7': 0,
                                          'concept_total': 0,
                                          'intention_total': 0,
                                          'solving question_total': 0
                                         }
            
            self.acc_dict[user_id][f"{lecture_info['type_of']}"] += self.pebg_skill_repre[self.skill_id_dict[str(lecture_info['tag'])]]
            self.acc_dict[user_id][f"part_{lecture_info['part']}"] += 1
            self.acc_dict[user_id][f"{lecture_info['type_of']}_total"] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np.zeros(len(self.columns))
        acc_user = self.acc_dict.get(user_id, None)
        if content_type_id == 0 and acc_user is not None:
            concept_total = acc_user['concept_total']
            intention_total = acc_user['intention_total']
            solving_question_total = acc_user['solving question_total']

            if concept_total > 0:
                features[:self.pebg_skill_size] = acc_user['concept']#/concept_total
            if intention_total > 0:
                features[self.pebg_skill_size:self.pebg_skill_size*2] = acc_user['intention']#/intention_total
            if solving_question_total > 0:
                features[self.pebg_skill_size*2:self.pebg_skill_size*3] = acc_user['solving question']#/solving_question_total
            
            features[self.pebg_skill_size*3] = acc_user['part_1']
            features[self.pebg_skill_size*3+1] = acc_user['part_2']
            features[self.pebg_skill_size*3+2] = acc_user['part_3']
            features[self.pebg_skill_size*3+3] = acc_user['part_4']
            features[self.pebg_skill_size*3+4] = acc_user['part_5']
            features[self.pebg_skill_size*3+5] = acc_user['part_6']
            features[self.pebg_skill_size*3+6] = acc_user['part_7']
            features[self.pebg_skill_size*3+7] = concept_total
            features[self.pebg_skill_size*3+8] = intention_total
            features[self.pebg_skill_size*3+9] = solving_question_total

        return features

class LectureFeats2(Accumulator):
    """
    Provides features relating to lectures the user has watched

    Accumulates By: user_id

    Requires:
    - tag1: from MetadataFeats

    Features:
    - lec_timeprop: user's time spent on all lectures compared to lecture duration proxy
    - user_total_lec_duration: user's time spent on all lectures
    - mean_lec_timeprop: mean of user's past lec_timeprop records
    - max_lec_timeprop: max of user's past lec_timeprop records
    - num_lectures: number of lectures the user watched
    - tag1_timestamp_diff: timestamp difference between current question and most recent lecture with the same tag
    - tag1_latest_lec_type_of: type of the most recent lecture with the same tag as the current question
    - tag1_mean_lec_timeprop: mean of user's past lec_timeprop records for lectures with the same tag as the current question
    - tag1_max_lec_timeprop: max of user's past lec_timeprop records for lectures with the same tag as the current question
    - tag1_num_lec: number of user's past lec_timeprop records for lectures with the same tag as the current question
    """

    level = 1
    req_answered_correctly = False
    req_user_answer = False
    columns = [
        'lec_timeprop', 'user_total_lec_duration', 'mean_lec_timeprop', 'max_lec_timeprop', 'num_lectures',
        'tag1_timestamp_diff', 'tag1_latest_lec_type_of', 'tag1_mean_lec_timeprop', 'tag1_max_lec_timeprop', 'tag1_num_lec'
    ]

    def __init__(self, lectures_df, lec_duration_dict, base_acc_dict=None, updatable=True, suffix=""):
        self.lectures_dict = lectures_df.set_index('lecture_id').to_dict('index')
        self.lec_duration_dict = lec_duration_dict
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    #@njit
    def get_prev_lec_features(user_history_arr, arr_size, timestamp, tag1):
        total_lec_timeprop = 0
        max_lec_timeprop = 0
        latest_timestamp = 0
        latest_lec_cid = np.nan
        tag1_num_lec = 0

        for i in range(arr_size):
            user_history_arr_i = user_history_arr[i]
            tag = round(user_history_arr_i[1] % 1 * 1000)

            if tag == tag1:
                latest_timestamp = user_history_arr_i[0] # later array entries are guaranteed to be later in time since append is used
                latest_lec_cid = int(user_history_arr_i[1])
                total_lec_timeprop += user_history_arr_i[2]
                max_lec_timeprop = max(max_lec_timeprop, user_history_arr_i[2])
                tag1_num_lec += 1

        if tag1_num_lec > 0:
            return timestamp - latest_timestamp, latest_lec_cid, total_lec_timeprop/tag1_num_lec, max_lec_timeprop, tag1_num_lec
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']

        if user_id not in self.acc_dict:
            self.acc_dict[user_id] = {
                'timestamp': 0, 'user_total_lec_duration': 0, 'total_lec_duration': 0, 'total_mean_lec_duration': 0, 'max_lec_timeprop': 0, 'total': 0, 'lecture_history': arraylist2D(space=(10,3))
            }

        acc_user = self.acc_dict[user_id]

        if content_type_id == 1:
            content_id = row[cm['content_id']] if cm else row['content_id']
            lec_duration = self.lec_duration_dict[content_id]
            user_lec_duration = min(lec_duration, timestamp - acc_user['timestamp'])
            user_lec_timeprop = user_lec_duration/lec_duration if lec_duration > 0 else 0
            user_history_arr = acc_user['lecture_history']
            lec_tag = self.lectures_dict[content_id]['tag']

            acc_user['user_total_lec_duration'] += user_lec_duration
            acc_user['total_lec_duration'] += lec_duration
            acc_user['total_mean_lec_duration'] += user_lec_timeprop
            if user_lec_timeprop > acc_user['max_lec_timeprop']:
                acc_user['max_lec_timeprop'] = user_lec_timeprop
            acc_user['total'] += 1

            #if not self.update_user_history_arr(user_history_arr.data, user_history_arr.size, timestamp, content_id, lec_tag, user_lec_timeprop):
            user_history_arr.append(np.array((timestamp, content_id + lec_tag/1000, user_lec_timeprop)))

        acc_user['timestamp'] = timestamp

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                timestamp = row[cm['timestamp']] if cm else row['timestamp']
                tag1 = feature_cache['tag1']
                user_history_arr = acc_user['lecture_history']

                tag1_timestamp_diff, tag1_latest_lec_cid, tag1_mean_lec_timeprop, tag1_max_lec_timeprop, tag1_num_lec = self.get_prev_lec_features(user_history_arr.data, user_history_arr.size, timestamp, tag1)
                llto = self.lectures_dict[tag1_latest_lec_cid]['type_of'] if tag1_latest_lec_cid >= 0 else np.nan
                tag1_latest_lec_type_of = 0 if llto == "concept" else 1 if llto == "intention" else 2 if llto == "solving_question" else 3 if llto == "starter" else np.nan

                features = np_array(
                    acc_user['user_total_lec_duration']/acc_user['total_lec_duration'] if acc_user['total_lec_duration'] > 0 else 0,
                    acc_user['user_total_lec_duration'],
                    acc_user['total_mean_lec_duration']/acc_user['total'] if acc_user['total'] > 0 else 0,
                    acc_user['max_lec_timeprop'],
                    acc_user['total'],
                    tag1_timestamp_diff,
                    tag1_latest_lec_type_of,
                    tag1_mean_lec_timeprop,
                    tag1_max_lec_timeprop,
                    tag1_num_lec
                )

        return features

#endregion
#region ######################### LEVEL 2 ACCUMULATORS ##########################

class TimestampHistory(Accumulator):
    """
    Provides timestamp history for user's last N question records. 

    Accumulates By: user_id

    Requires:
    - timestamp: from PriorTimestampFeats

    Features:
    """
    level = 2
    req_answered_correctly = True # doesn't use answered_correctly but needs to update after it's available to align with other last_n accumulators
    req_user_answer = False
    columns = []

    def __init__(self, last_n_max=200, base_acc_dict=None, updatable=True, suffix=""):
        self.last_n_max = last_n_max
        self.user_id = np.nan
        self.bundle_id = np.nan
        self.timestamp = np.nan
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    #@njit
    def np_push(arr, user_responses, last_n, curr_idx, v):
        if user_responses >= last_n:
            arr[:curr_idx] = arr[1:]
            arr[curr_idx:] = 0

        arr[curr_idx] = v

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return
        
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time'] # TODO: get this from PriorTimestampFeats
        bundle_id = feature_cache['bundle_id']
        num_prior_questions = feature_cache['num_prior_questions']
        user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'timestamp_history': np.zeros(self.last_n_max), 'elapsed_time_history': np.zeros(self.last_n_max-1)}

            acc_user = self.acc_dict[user_id]
            curr_idx = int(min(self.last_n_max-1, user_responses) if user_responses > 0 else 0)

            self.np_push(acc_user['timestamp_history'], user_responses, self.last_n_max, curr_idx, timestamp)

            if (user_id != self.user_id or bundle_id != self.bundle_id or timestamp != self.timestamp) and user_responses > 0:
                curr_idx_pqet = int(min(self.last_n_max-2, user_responses-num_prior_questions) if user_responses > 1 else 0)
                for _ in range(num_prior_questions):
                    self.np_push(acc_user['elapsed_time_history'], user_responses-num_prior_questions, self.last_n_max-1, curr_idx_pqet, prior_question_elapsed_time)
                    curr_idx_pqet = int(min(self.last_n_max-2, curr_idx_pqet + 1))    
                    num_prior_questions -= 1

            self.user_id = user_id
            self.bundle_id = bundle_id
            self.timestamp = timestamp

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        if 'timestamp_history' not in feature_cache:
            feature_cache['timestamp_history'] = np.zeros(self.last_n_max)
        if 'elapsed_time_history' not in feature_cache:
            feature_cache['elapsed_time_history'] = np.zeros(self.last_n_max-1)
        #features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                feature_cache['timestamp_history'][:] = acc_user['timestamp_history']
                feature_cache['elapsed_time_history'][:] = acc_user['elapsed_time_history']

        return

class GuessFeats(Accumulator):
    """
    Provides features about the user's guessing behaviour.

    Accumulates By: user_id

    Requires:
    - user_responses: from UAccuracy

    Features:
    - ua_0_rate: proportion of incorrect responses where the user chose answer 0
    - ua_1_rate: proportion of incorrect responses where the user chose answer 1
    - ua_2_rate: proportion of incorrect responses where the user chose answer 2
    - ua_3_rate: proportion of incorrect responses where the user chose answer 3
    """

    level = 2
    req_answered_correctly = True
    req_user_answer = False
    columns = ['ua_0_rate', 'ua_1_rate', 'ua_2_rate', 'ua_3_rate', 'total_incorrect']

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly'] 
        user_answer = row[cm['user_answer']] if cm else row['user_answer'] 

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'ua_0_count': 0, 'ua_1_count': 0, 'ua_2_count': 0, 'ua_3_count': 0}
                
            if not answered_correctly:
                self.acc_dict[user_id][f"ua_{int(user_answer)}_count"] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                ua_0_count = acc_user['ua_0_count']
                ua_1_count = acc_user['ua_1_count']
                ua_2_count = acc_user['ua_2_count']
                ua_3_count = acc_user['ua_3_count']
                total_incorrect = ua_0_count + ua_1_count + ua_2_count + ua_3_count

                features = np_array(
                    ua_0_count/total_incorrect if total_incorrect > 0 else np.nan,
                    ua_1_count/total_incorrect if total_incorrect > 0 else np.nan,
                    ua_2_count/total_incorrect if total_incorrect > 0 else np.nan,
                    ua_3_count/total_incorrect if total_incorrect > 0 else np.nan,
                    total_incorrect
                )

        return features

class QNormUAccuracy(Accumulator):
    """
    Provides user accuracy features normalized by question difficulty

    Accumulates By: user_id

    Requires:
    - question_accuracy: from QAccuracy

    Features:
    - prev_resid: user's residual (answered_correctly - question_accuracy) for their previous question
    - total_resid: user's cumulative residual
    - resid_mean: user's average residual (cumulative / number of responses)
    - resid_correct_mean: user's average residual for correct responses
    - resid_incorrect_mean: user's average residual for incorrect responses
    """

    level = 2 
    req_answered_correctly = True
    req_user_answer = False
    columns = ['prev_resid', 'total_resid', 'resid_mean', 'resid_correct_mean', 'resid_incorrect_mean']

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return
        
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        question_accuracy = feature_cache['question_accuracy']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'prev_resid': 0, 'total_resid': 0, 'total_resid_correct': 0, 'total_resid_incorrect': 0, 'total_correct': 0, 'total_incorrect': 0}
            resid = answered_correctly - question_accuracy
            self.acc_dict[user_id]['prev_resid'] = resid
            self.acc_dict[user_id]['total_resid'] += resid

            if answered_correctly:
                self.acc_dict[user_id]['total_resid_correct'] += resid
                self.acc_dict[user_id]['total_correct'] += 1
            else:
                self.acc_dict[user_id]['total_resid_incorrect'] += resid
                self.acc_dict[user_id]['total_incorrect'] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                user_total_resid = acc_user['total_resid']
                user_total_correct = acc_user['total_correct']
                user_total_incorrect = acc_user['total_incorrect']
                user_total_responses = user_total_correct + user_total_incorrect

                features = np_array(
                    acc_user['prev_resid'],
                    user_total_resid,
                    user_total_resid/user_total_responses if user_total_responses > 0 else np.nan,
                    acc_user['total_resid_correct']/user_total_correct if user_total_correct > 0 else np.nan,
                    acc_user['total_resid_incorrect']/user_total_incorrect if user_total_incorrect > 0 else np.nan
                )

        return features

class QNormUAccuracyLastN(Accumulator):
    """
    Provides user accuracy features normalized by question difficulty for the last N responses by the user

    Accumulates By: user_id

    Requires:
    - question_accuracy: from QAccuracy
    - user_responses: from UAccuracy

    Features:
    - resid_mean_last{i}: user's average residual (cumulative / number of responses) over their last i records
    - resid_std_last{i}: standard deviation of user's residuals over their last i records
    """

    level = 2 
    req_answered_correctly = True
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, last_n=[3, 10, 30, 50, 100, 200], base_acc_dict=None, updatable=True, suffix=""):
        self.last_n = tuple(sorted(last_n))
        self.last_n_max = max(self.last_n)

        base_columns = ['resid_mean', 'resid_std']
        self.base_columns_size = len(base_columns)

        self.columns = [f"{col}_last{suffix}" for suffix in last_n for col in base_columns] 
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def get_features_njit(features, last_n, user_history_arr, user_history_size, base_columns_size):
        len_last_n = len(last_n)

        for i in range(len_last_n):
            last_n_size = last_n[i]
            user_history_start_idx = max(0, user_history_size - last_n_size)
            user_resid_history = user_history_arr[user_history_start_idx:user_history_start_idx+min(last_n_size, user_history_size)]

            features[i*base_columns_size:(i+1)*base_columns_size] = np_array(
                np_mean(user_resid_history),
                np_std(user_resid_history)
            )

    # store residuals and answered_correctly in one array. Edge case: question accuracy = 0 and answered incorrectly. Store -0.0 to retain sign
    # to retrieve answered_correctly array: ac = (~np.signbit(arr)).astype(np.int8)
    @staticmethod
    @njit
    def np_push(arr, user_responses, last_n, curr_idx, answered_correctly, question_accuracy):
        resid = answered_correctly - question_accuracy
        if answered_correctly:
            v = resid
        else:
            v = resid if resid < 0 else -0.0

        if user_responses >= last_n:
            arr[:curr_idx] = arr[1:]
            arr[curr_idx:] = 0

        arr[curr_idx] = v

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return
        
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        question_accuracy = feature_cache['question_accuracy']
        user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = np.zeros(self.last_n_max)

            acc_user = self.acc_dict[user_id]
            curr_idx = int(min(self.last_n_max-1, user_responses) if user_responses > 0 else 0)

            self.np_push(acc_user, user_responses, self.last_n_max, curr_idx, answered_correctly, question_accuracy)

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        user_responses = feature_cache['user_responses']

        if 'qnorm_uaccuracy_history' not in feature_cache:
            feature_cache['qnorm_uaccuracy_history'] = np.zeros(self.last_n_max)
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                feature_cache['qnorm_uaccuracy_history'][:] = acc_user # need to pass in a copy, otherwise it will be modified by self.update
                user_history_size = int(min(self.last_n_max, user_responses) if user_responses > 0 else 0)
                self.get_features_njit(features, self.last_n, acc_user, user_history_size, self.base_columns_size)

        return features

class TimeCorrect(Accumulator):
    """
    Provides features capturing the average time spent on correct answers.
    TODO: check if excluding lectures is better

    Accumulates By: user_id, content_id

    Requires:
    - time_since_last_action: from PriorTimestampFeats

    Features:
    - avg_user_time_correct: average time_since_last_action for the user's correct responses
    - avg_content_time_correct: average time_since_last_action for the question's correct responses across all users
    - ratio_tsla_to_autc: ratio between time_since_last_action and avg_user_time_correct
    - ratio_tsla_to_actc: ratio between time_since_last_action and avg_content_time_correct
    """

    level = 2 # needs to be level 2 for 2 reasons: 1. need to use features from TimeSinceLastAction accum 2. req_answered_correctly=True here but False for TimeSinceLastAction
    req_answered_correctly = True
    req_user_answer = False
    columns = [
        'avg_user_time_correct', 'avg_user_time_incorrect', 'avg_content_time_correct', 'avg_content_time_incorrect', 
        'ratio_tsla_to_autc', 'ratio_tsla_to_auti', 'ratio_tsla_to_actc', 'ratio_tsla_to_acti'
    ]

    def __init__(self, base_acc_dict={'user_acc_dict': {}, 'content_acc_dict': {}}, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        time_since_last_action = feature_cache['time_since_last_action']

        if content_type_id == 0:
            if user_id not in self.acc_dict['user_acc_dict']:
                    self.acc_dict['user_acc_dict'][user_id] = {'time_correct': 0, 'total_correct': 0, 'time_incorrect': 0, 'total_incorrect': 0}
            if content_id not in self.acc_dict['content_acc_dict']:
                self.acc_dict['content_acc_dict'][content_id] = {'time_correct': 0, 'total_correct': 0, 'time_incorrect': 0, 'total_incorrect': 0}


            if time_since_last_action > 0 and time_since_last_action <= 300000: # handle 0 times and np.nan, and ignore cases where time_since_last_action > 5 minutes
                if answered_correctly:
                    self.acc_dict['user_acc_dict'][user_id]['time_correct'] += time_since_last_action
                    self.acc_dict['content_acc_dict'][content_id]['time_correct'] += time_since_last_action
                    self.acc_dict['user_acc_dict'][user_id]['total_correct'] += 1
                    self.acc_dict['content_acc_dict'][content_id]['total_correct'] += 1
                else:
                    self.acc_dict['user_acc_dict'][user_id]['time_incorrect'] += time_since_last_action
                    self.acc_dict['content_acc_dict'][content_id]['time_incorrect'] += time_since_last_action
                    self.acc_dict['user_acc_dict'][user_id]['total_incorrect'] += 1
                    self.acc_dict['content_acc_dict'][content_id]['total_incorrect'] += 1
        
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        time_since_last_action = feature_cache['time_since_last_action']

        features = np_nans(len(self.columns))
        autc = np.nan
        auti = np.nan
        ratio_tsla_to_autc = np.nan
        ratio_tsla_to_auti = np.nan
        actc = np.nan
        acti = np.nan
        ratio_tsla_to_actc = np.nan
        ratio_tsla_to_acti = np.nan

        if self.acc_dict['user_acc_dict'].get(user_id, None):
            user_time_correct = self.acc_dict['user_acc_dict'][user_id]['time_correct']
            user_time_incorrect = self.acc_dict['user_acc_dict'][user_id]['time_incorrect']

            user_total_correct = self.acc_dict['user_acc_dict'][user_id]['total_correct']
            user_total_incorrect = self.acc_dict['user_acc_dict'][user_id]['total_incorrect']

            autc = user_time_correct/user_total_correct if user_total_correct > 0 else np.nan
            auti = user_time_incorrect/user_total_incorrect if user_total_incorrect > 0 else np.nan

            ratio_tsla_to_autc = time_since_last_action/autc
            ratio_tsla_to_auti = time_since_last_action/auti

        if self.acc_dict['content_acc_dict'].get(content_id, None):
            content_time_correct = self.acc_dict['content_acc_dict'][content_id]['time_correct']
            content_time_incorrect = self.acc_dict['content_acc_dict'][content_id]['time_incorrect']

            content_total_correct = self.acc_dict['content_acc_dict'][content_id]['total_correct']
            content_total_incorrect = self.acc_dict['content_acc_dict'][content_id]['total_incorrect']

            actc = content_time_correct/content_total_correct if content_total_correct > 0 else np.nan
            acti = content_time_incorrect/content_total_incorrect if content_total_incorrect > 0 else np.nan

            ratio_tsla_to_actc = time_since_last_action/actc
            ratio_tsla_to_acti = time_since_last_action/acti

        features = np_array(
            autc, 
            auti,
            actc,
            acti, 
            ratio_tsla_to_autc, 
            ratio_tsla_to_auti,
            ratio_tsla_to_actc,
            ratio_tsla_to_acti
        )

        return features

class SessionFeats(Accumulator):
    """
    Provides features relating to the session the user is in. If the time since last action is greater than 5 min, we initialize a new session.

    Accumulates By: user_id

    Requires:
    - time_since_last_action: from PriorTimestampFeats

    Features:
    - user_session_accuracy: user's accuracy during the session
    - user_session_responses: number of questions the user attempted during the session
    - session_duration: session duration in milliseconds
    - user_session_accuracy_std: standard deviation of user accuracy for the session

    Note:
    - mean and standard deviation calculated using Welford's algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    level = 2
    req_answered_correctly = True
    req_user_answer = False
    columns = ['user_session_accuracy', 'user_session_responses', 'session_duration', 'user_session_accuracy_std']

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        time_since_last_action = feature_cache['time_since_last_action']

        if user_id not in self.acc_dict:
            self.acc_dict[user_id] = {'user_session_correct': 0, 'user_session_responses': 0, 'session_duration': 0, 'mean_accuracy': 0, 'm2_accuracy': 0}

        if time_since_last_action > 300000:
            self.acc_dict[user_id]['user_session_correct'] = 0
            self.acc_dict[user_id]['user_session_responses'] = 0
            self.acc_dict[user_id]['session_duration'] = 0
            self.acc_dict[user_id]['mean_accuracy'] = 0
            self.acc_dict[user_id]['m2_accuracy'] = 0
        
        if content_type_id == 0:
            prev_user_session_responses = self.acc_dict[user_id]['user_session_responses']
            prev_mean_accuracy = self.acc_dict[user_id]['mean_accuracy']

            self.acc_dict[user_id]['user_session_correct'] += answered_correctly
            self.acc_dict[user_id]['user_session_responses'] += 1
            self.acc_dict[user_id]['session_duration'] += time_since_last_action

            new_accuracy = self.acc_dict[user_id]['user_session_correct']/self.acc_dict[user_id]['user_session_responses']
            new_mean_accuracy = prev_mean_accuracy + (new_accuracy - prev_mean_accuracy)/(prev_user_session_responses+1)
            self.acc_dict[user_id]['mean_accuracy'] = new_mean_accuracy
            self.acc_dict[user_id]['m2_accuracy'] += (new_accuracy - prev_mean_accuracy) * (new_accuracy - new_mean_accuracy)

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        time_since_last_action = feature_cache['time_since_last_action']

        features = np_nans(len(self.columns))
        if content_type_id == 0 and time_since_last_action <= 300000 and self.acc_dict.get(user_id, None):
            user_session_responses = self.acc_dict[user_id]['user_session_responses']
            
            features = np_array(
                self.acc_dict[user_id]['user_session_correct']/user_session_responses if user_session_responses > 0 else np.nan,
                user_session_responses,
                self.acc_dict[user_id]['session_duration'],
                np.sqrt(self.acc_dict[user_id]['m2_accuracy']/user_session_responses) if user_session_responses > 0 else np.nan
            )
        
        return features

class PreviousTag1Accum(Accumulator):
    """
    Provides features capturing the last answered_correctly for a given user_id on a given tag1

    Accumulates By: user_id

    Requires:
    - tag1: from MetadataFeats
    - question_accuracy: from QAccuracy

    Features:
    - tag1_previous_try: previous answered_correctly for a given user_id on a given tag1
    - tag1_num_prev_tries: number of previous tries for the user on the tag1
    - tag1_prev_try_accuracy: accuracy of the user on previous tries for the tag1
    - tag1_mean_prev_try_resid: mean residual of the user on previous tries for the tag1

    Note: 
    - large accumulator dict if mapping user_id's to all of the tag1's they attempted directly
    - instead, map user_id to array of tag1's, with negative tag1's representing past incorrects and positive representing past correct attempts
    - further, encode user's number of attempts for the question using multiple of 100k and number of correct attemps using the decimal - takes no extra space 
      - use 1/10000 as decimal count for correct attempts since it's unlikely any user will attempt the same tag1 10k times (never happens in train set)
    """
    
    level = 2
    req_answered_correctly = True
    req_user_answer = False
    columns = ['tag1_previous_try', 'tag1_num_prev_tries', 'tag1_prev_try_accuracy', 'tag1_mean_prev_try_resid']

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def update_tag1_list(tag1_arrlist, tag1_resid_arrlist, tag1, answered_correctly, question_accuracy):
        for i in range(len(tag1_arrlist)):
            tag1_arrlist_i = abs(tag1_arrlist[i])
            cid = int(tag1_arrlist_i % 100000)

            if cid == tag1:
                num_responses = int(tag1_arrlist_i // 100000)
                num_correct = round(tag1_arrlist_i % 1 * 10000)
                
                if answered_correctly:
                    tag1_arrlist[i] = 100000*(num_responses+1) + tag1 + (num_correct+1)/10000
                else:
                    tag1_arrlist[i] = -(100000*(num_responses+1) + tag1 + num_correct/10000)

                tag1_resid_arrlist[i] += answered_correctly - question_accuracy

                return True
        return False

    @staticmethod
    @njit
    def get_prev_tag1(tag1_arrlist, tag1_resid_arrlist, tag1):
        for i in range(len(tag1_arrlist)):
            tag1_arrlist_i = abs(tag1_arrlist[i])
            t1 = int(tag1_arrlist_i % 100000)

            if t1 == tag1:
                num_responses = int(tag1_arrlist_i // 100000)
                num_correct = round(tag1_arrlist_i % 1 * 10000)
                prev_try_accuracy = num_correct/num_responses if num_responses > 0 else np.nan
                mean_prev_try_resid = tag1_resid_arrlist[i]/num_responses if num_responses > 0 else np.nan

                if tag1_arrlist[i] < 0:
                    return 0, num_responses, prev_try_accuracy, mean_prev_try_resid
                else: 
                    return 1, num_responses, prev_try_accuracy, mean_prev_try_resid
        return np.nan, np.nan, np.nan, np.nan

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        tag1 = feature_cache['tag1']
        question_accuracy = feature_cache['question_accuracy']
        
        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'tag_arrlist': arraylist(), 'tag_resid_arrlist': arraylist()}

            acc_user = self.acc_dict[user_id]

            if not self.update_tag1_list(acc_user['tag_arrlist'].data, acc_user['tag_resid_arrlist'].data, tag1, answered_correctly, question_accuracy):
                acc_user['tag_arrlist'].append(100000.0 + tag1 + 0.0001 if answered_correctly else -(100000.0 + tag1))
                acc_user['tag_resid_arrlist'].append(answered_correctly - question_accuracy)
        
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        tag1 = feature_cache['tag1']
        
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                previous_try, num_prev_tries, prev_try_accuracy, mean_prev_try_resid = self.get_prev_tag1(acc_user['tag_arrlist'].data, acc_user['tag_resid_arrlist'].data, tag1)
                features = np_array(
                    previous_try, 
                    num_prev_tries, 
                    prev_try_accuracy,
                    mean_prev_try_resid
                )

        return features

    def compress(self):
        def compress_arrlist(al):
            al.data = al.data[~np.isnan(al.data)]
            new_size = len(al.data)
            al.size = new_size
            al.capacity = new_size

        for _, acc_user in tqdm(self.acc_dict.items()):
            compress_arrlist(acc_user['tag_arrlist'])
            compress_arrlist(acc_user['tag_resid_arrlist'])

    # override read and save for this accumulator to save memory overhead since the acc_dict is large
    def read(self, filename):
        self.acct_dict = read_from_pickle_dict_chunks(filename)

    def save(self, filename):
        write_to_pickle_dict_chunks(self.acc_dict, filename)

class PreviousTag1Accum2(Accumulator):
    """
    Provides features capturing the last answered_correctly for a given user_id on all tags

    Accumulates By: user_id

    Requires:
    - tag1: from MetadataFeats
    - question_accuracy: from QAccuracy

    Features:
    - tag1_previous_try_{i}: previous answered_correctly for a given user_id on tag i
    - tag1_num_prev_tries_{i}: number of previous tries for the user on tag i
    - tag1_prev_try_accuracy_{i}: accuracy of the user on previous tries for tag i
    - tag1_mean_prev_try_resid_{i}: mean residual of the user on previous tries for tag i

    Note: 
    - large accumulator dict if mapping user_id's to all of the tag1's they attempted directly
    - instead, map user_id to array of tag1's, with negative tag1's representing past incorrects and positive representing past correct attempts
    - further, encode user's number of attempts for the question using multiple of 100k and number of correct attemps using the decimal - takes no extra space 
      - use 1/10000 as decimal count for correct attempts since it's unlikely any user will attempt the same tag1 10k times (never happens in train set)
    """
    
    level = 2
    req_answered_correctly = True
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, questions_df, base_acc_dict=None, updatable=True, suffix=""):
        tag_schema = {tag: i for i, tag in enumerate(np.sort(questions_df['tags'].str.split().str[0].fillna(0).astype(int).unique()))}
        self.tag_schema_tuple = tuple(tag_schema[i] if i in tag_schema else -1 for i in range(max(tag_schema)+1))
        self.columns = (
            [f"tag1_previous_try_{i}" for i in tag_schema] + 
            [f"tag1_num_prev_tries_{i}" for i in tag_schema] + 
            [f"tag1_prev_try_accuracy_{i}" for i in tag_schema] + 
            [f"tag1_num_prev_try_correct_{i}" for i in tag_schema] +
            [f"tag1_num_prev_try_incorrect_{i}" for i in tag_schema] + 
            [f"tag1_mean_prev_try_resid_{i}" for i in tag_schema]
        )
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def update_tag1_list(tag1_arrlist, tag1_resid_arrlist, tag1, answered_correctly, question_accuracy):
        for i in range(len(tag1_arrlist)):
            tag1_arrlist_i = abs(tag1_arrlist[i])
            tag = int(tag1_arrlist_i % 100000)

            if tag == tag1:
                num_responses = int(tag1_arrlist_i // 100000)
                num_correct = round(tag1_arrlist_i % 1 * 10000)
                
                if answered_correctly:
                    tag1_arrlist[i] = 100000*(num_responses+1) + tag1 + (num_correct+1)/10000
                else:
                    tag1_arrlist[i] = -(100000*(num_responses+1) + tag1 + num_correct/10000)

                tag1_resid_arrlist[i] += answered_correctly - question_accuracy

                return True
        return False

    @staticmethod
    @njit
    def get_prev_tag1(tag1_arrlist, tag1_resid_arrlist, tag_schema_tuple):
        num_tags = max(tag_schema_tuple) + 1
        tag_features = np_nans(6*num_tags)

        for i in range(len(tag1_arrlist)):
            tag1_arrlist_i = abs(tag1_arrlist[i])
            if not np.isnan(tag1_arrlist_i):
                t1 = int(tag1_arrlist_i % 100000)
                idx = tag_schema_tuple[t1]

                num_responses = int(tag1_arrlist_i // 100000)
                num_correct = round(tag1_arrlist_i % 1 * 10000)
                prev_try_accuracy = num_correct/num_responses if num_responses > 0 else np.nan
                mean_prev_try_resid = tag1_resid_arrlist[i]/num_responses if num_responses > 0 else np.nan

                if tag1_arrlist[i] < 0:
                    tag_features[idx] = 0
                else: 
                    tag_features[idx] = 1

                tag_features[num_tags + idx] = num_responses
                tag_features[num_tags*2 + idx] = prev_try_accuracy
                tag_features[num_tags*3 + idx] = num_correct
                tag_features[num_tags*4 + idx] = num_responses - num_correct
                tag_features[num_tags*5 + idx] = mean_prev_try_resid
            
        return tag_features

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        tag1 = feature_cache['tag1']
        question_accuracy = feature_cache['question_accuracy']
        
        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'tag_arrlist': arraylist(), 'tag_resid_arrlist': arraylist()}

            acc_user = self.acc_dict[user_id]

            if not self.update_tag1_list(acc_user['tag_arrlist'].data, acc_user['tag_resid_arrlist'].data, tag1, answered_correctly, question_accuracy):
                acc_user['tag_arrlist'].append(100000.0 + tag1 + 0.0001 if answered_correctly else -(100000.0 + tag1))
                acc_user['tag_resid_arrlist'].append(answered_correctly - question_accuracy)
        
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                features = self.get_prev_tag1(acc_user['tag_arrlist'].data, acc_user['tag_resid_arrlist'].data, self.tag_schema_tuple)

        return features

    def compress(self):
        def compress_arrlist(al):
            al.data = al.data[~np.isnan(al.data)]
            new_size = len(al.data)
            al.size = new_size
            al.capacity = new_size

        for _, acc_user in tqdm(self.acc_dict.items()):
            compress_arrlist(acc_user['tag_arrlist'])
            compress_arrlist(acc_user['tag_resid_arrlist'])

    # override read and save for this accumulator to save memory overhead since the acc_dict is large
    def read(self, filename):
        self.acct_dict = read_from_pickle_dict_chunks(filename)

    def save(self, filename):
        write_to_pickle_dict_chunks(self.acc_dict, filename)

class QQMFeats(Accumulator):
    """
    Provides question-question matrix features

    Accumulates by: user_id
    
    Requires: 
    - bundle_idx: from PriorTimestampFeats
    - bundle_id_count: from MetadataFeats
    - user_responses: from UAccuracy
    - question_accuracy: from QAccuracy

    Features:
    - qqm_mean: mean of qqm predictions
    - qqm_max: max of qqm predictions
    - qqm_min: min of qqm predictions
    - qqm_mean_to_qaccuracy: ratio of qqm_mean to overall question accuracy
    - qqm_max_to_qaccuracy: ratio of qqm_max to overall question accuracy
    - qqm_min_to_qaccuracy: ratio of qqm_min to overall question accuracy

    Note:
    - self.last_n should be large enough to accommodate bundles
    """

    level = 2
    req_answered_correctly = False
    req_user_answer = True
    columns = ['qqm_mean', 'qqm_max', 'qqm_min', 'qqm_mean_to_qaccuracy', 'qqm_max_to_qaccuracy', 'qqm_min_to_qaccuracy']

    def __init__(self, qqm, base_acc_dict=None, last_n=10, updatable=True, suffix=""):
        self.qqm = qqm
        self.last_n = last_n
        self.user_id = np.nan
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def get_qqm_preds(qqm, content_id, last_n_arr, last_n_size, user_history, question_accuracy, samples_gt=0):
        qqm_preds = np.full(last_n_size, question_accuracy)
        start_idx = last_n_size - user_history
        for i in range(user_history):
            uh_cid = int(last_n_arr[i][0]) # content id for i'th row in user history array
            uh_ua = int(last_n_arr[i][1]) # user answer for i'th row in user history array
            qqm_i = qqm[uh_cid][int(content_id)]
            qqm_preds[start_idx+i] = qqm_i[uh_ua]/qqm_i[4+uh_ua] if qqm_i[4+uh_ua] > samples_gt else np.nan
        
        return qqm_preds

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        user_answer = row[cm['user_answer']] if cm else row['user_answer'] 
        user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {'last_n': np.zeros((self.last_n, 2)), 'buffer': np.zeros((self.last_n, 2)), 'buffer_size': 0}

            bundle_idx = int(feature_cache['bundle_idx']) # init here since bundle_idx is nan for lectures and can't cast to int
            bundle_id_count = int(feature_cache['bundle_id_count'])

            acc_user = self.acc_dict[user_id]['last_n']
            buffer_user = self.acc_dict[user_id]['buffer']
            buffer_size = self.acc_dict[user_id]['buffer_size']
            user_history = int(min(self.last_n, user_responses-buffer_size) if user_responses > 0 else 0)
            keep_space = user_history

            # update accum if finished current bundle or user exited previous bundle into a new one
            if (bundle_idx == bundle_id_count) or (bundle_idx == 1 and buffer_size > 0):
                # if there's no room, shift everything up
                if user_responses >= self.last_n:
                    keep_space = self.last_n-buffer_size-1
                    acc_user[:keep_space, :] = acc_user[user_history-keep_space:user_history, :]
                    acc_user[keep_space:, :] = 0

                if buffer_size > 0:
                    # empty buffer into accum
                    if acc_user[keep_space:keep_space+buffer_size].shape[0] == 0:
                        print("before empty:\n", bundle_idx, bundle_id_count, user_id, content_id, self.last_n, user_responses, user_history, buffer_size, keep_space, "\n", acc_user, flush=True)
                    acc_user[keep_space:keep_space+buffer_size] = buffer_user[:buffer_size]
                    buffer_user[:] = 0
                    keep_space += buffer_size
                    self.acc_dict[user_id]['buffer_size'] = 0

                acc_user[keep_space, 0] = content_id
                acc_user[keep_space, 1] = user_answer
            else:
                # store into buffer
                buffer_user[buffer_size, 0] = content_id
                buffer_user[buffer_size, 1] = user_answer
                self.acc_dict[user_id]['buffer_size'] += 1

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']

        question_accuracy = feature_cache['question_accuracy']
        user_responses = feature_cache['user_responses']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                buffer_size = self.acc_dict[user_id]['buffer_size']
                user_history = int(min(self.last_n, user_responses-buffer_size) if user_responses > 0 else 0)
                qqm_preds = self.get_qqm_preds(self.qqm, content_id, acc_user['last_n'], self.last_n, user_history, question_accuracy)
                qqm_mean = np_mean(qqm_preds)
                qqm_max = np_max(qqm_preds)
                qqm_min = np_min(qqm_preds)

                features = np_array(
                    qqm_mean,
                    qqm_max,
                    qqm_min,
                    qqm_mean/question_accuracy if question_accuracy > 0 else np.nan,
                    qqm_max/question_accuracy if question_accuracy > 0 else np.nan,
                    qqm_min/question_accuracy if question_accuracy > 0 else np.nan
                )
        
        return features

class QQMFeats2(Accumulator):
    """
    Provides question-question matrix features. This version ignores bundle id (performs slightly better).

    Accumulates by: user_id
    
    Requires: 
    - user_responses: from UAccuracy
    - question_accuracy: from QAccuracy

    Features:
    - qqm_mean: mean of qqm predictions
    - qqm_max: max of qqm predictions
    - qqm_min: min of qqm predictions
    - qqm_mean_to_qaccuracy: ratio of qqm_mean to overall question accuracy
    - qqm_max_to_qaccuracy: ratio of qqm_max to overall question accuracy
    - qqm_min_to_qaccuracy: ratio of qqm_min to overall question accuracy
    """

    level = 2
    req_answered_correctly = False
    req_user_answer = True
    # columns defined in __init__

    def __init__(self, sim_matrix, qqm=None, qqm_overflow=None, last_n=[3, 10, 30, 50, 100, 200], w_smooth=20, samples_gt=0, base_acc_dict=None, updatable=True, suffix=""):
        self.qqm = qqm if qqm is not None else np.zeros((13523,13523,8), dtype=np.uint16)
        self.qqm_overflow = qqm_overflow if qqm_overflow is not None else {'data': np.zeros((1000,5), dtype=np.uint16), 'size': 0}
        self.sim_matrix = sim_matrix
        self.last_n = tuple(sorted(last_n))
        self.last_n_max = max(self.last_n)
        self.w_smooth = w_smooth
        self.samples_gt = samples_gt
        
        base_columns = [
            'qqm_mean', 'weighted_qqm_mean', 'qqm_max', 'qqm_min', 
            'qqm_mean_to_qaccuracy', 'weighted_qqm_mean_to_qaccuracy', 'qqm_max_to_qaccuracy', 'qqm_min_to_qaccuracy'
        ]
        self.base_columns_size = len(base_columns)

        # columns: ['qqm_mean_3', 'weighted_qqm_mean_3', ..., 'qqm_max_to_qaccuracy_200', 'qqm_min_to_qaccuracy_200']
        self.columns = [f"{col}_last{suffix}" for suffix in last_n for col in base_columns] 
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def get_features_njit(features, qqm, qqm_overflow, qqm_overflow_size, content_id, last_n, user_history_arr, user_history_size, question_accuracy, sim_matrix, w_smooth, base_columns_size, samples_gt):
        len_last_n = len(last_n)

        for i in range(len_last_n):
            last_n_size = last_n[i]
            qqm_preds = np.full(last_n_size, question_accuracy)
            user_history_start_idx = max(0, user_history_size - last_n_size)
            preds_start_idx = max(0, last_n_size - user_history_size)
            num_preds = min(last_n_size, user_history_size)
            weighted_qqm = 0
            weighted_qqm_total = 0
            
            for j in range(num_preds):
                uh_v = user_history_arr[user_history_start_idx + j] # stored value for i'th row in user history array
                uh_cid = int(uh_v) # content id for i'th row in user history array
                uh_ua = round(uh_v % 1 * 10) # user answer for i'th row in user history array
                content_id_int = int(content_id)
                qqm_j = qqm[uh_cid][content_id_int]
                uh_ua_shift = 0
                uh_uat_shift = 0

                for k in range(qqm_overflow_size):
                    if qqm_overflow[k, 0] == uh_cid and qqm_overflow[k, 1] == content_id_int and qqm_overflow[k, 2] == uh_ua:
                        uh_ua_shift = qqm_overflow[k, 3]
                        uh_uat_shift = qqm_overflow[k, 4]
                        break
                
                qqm_j_uh_ua = qqm_j[uh_ua] + 65536*uh_ua_shift
                qqm_j_uh_uat = qqm_j[4+uh_ua]+ 65536*uh_uat_shift

                qqm_preds[preds_start_idx+j] = (qqm_j_uh_ua + w_smooth*question_accuracy)/(qqm_j_uh_uat + w_smooth) if qqm_j_uh_uat > samples_gt else np.nan

                if qqm_j_uh_uat > samples_gt:
                    sim = pdist_sim(sim_matrix, uh_cid, content_id_int, len(qqm))
                    weighted_qqm += (qqm_j_uh_ua + w_smooth*question_accuracy)/(qqm_j_uh_uat + w_smooth)*sim
                    weighted_qqm_total += sim
                
            weighted_qqm_mean = weighted_qqm/weighted_qqm_total if weighted_qqm_total > 0 else np.nan
            qqm_mean = np_mean(qqm_preds)
            qqm_max = np_max(qqm_preds)
            qqm_min = np_min(qqm_preds)

            #print(features.shape, len_last_n, i, base_columns_size, i*base_columns_size, (i+1)*base_columns_size)
            features[i*base_columns_size:(i+1)*base_columns_size] = np_array(
                qqm_mean,
                weighted_qqm_mean,
                qqm_max,
                qqm_min,
                qqm_mean/question_accuracy if question_accuracy > 0 else np.nan,
                weighted_qqm_mean/question_accuracy if question_accuracy > 0 else np.nan,
                qqm_max/question_accuracy if question_accuracy > 0 else np.nan,
                qqm_min/question_accuracy if question_accuracy > 0 else np.nan
            )

    @staticmethod
    @njit
    def np_push(arr, user_responses, last_n, curr_idx, v):
        if user_responses >= last_n:
            arr[:curr_idx] = arr[1:]
            arr[curr_idx:] = 0

        arr[curr_idx] = v

    @staticmethod
    @njit
    def update_qqm(qqm, qqm_overflow, qqm_overflow_size, user_history_arr, curr_idx, content_id, answered_correctly):
        content_id_int = int(content_id)
        for i in range(curr_idx):
            uh_v = user_history_arr[i]
            cid = int(uh_v)
            uh_ua = round(uh_v % 1 * 10)
            qqm_i = qqm[cid][content_id_int]

            ua_overflow = qqm_i[uh_ua] + answered_correctly >= 65536
            uat_overflow = qqm_i[4+uh_ua] + 1 >= 65536
            if ua_overflow or uat_overflow:
                overflow_updated = False

                for j in range(qqm_overflow_size):
                    if qqm_overflow[j, 0] == cid and qqm_overflow[j, 1] == content_id_int and qqm_overflow[j, 2] == uh_ua:
                        if ua_overflow:
                            qqm_overflow[j, 3] += 1
                            qqm_i[uh_ua] = 0

                        if uat_overflow: # possible for both ua and uat to overflow
                            qqm_overflow[j, 4] += 1
                            qqm_i[4+uh_ua] = 0

                        overflow_updated = True
                        break
                
                if not overflow_updated:
                    qqm_overflow[qqm_overflow_size, 0] = cid
                    qqm_overflow[qqm_overflow_size, 1] = content_id_int
                    qqm_overflow[qqm_overflow_size, 2] = uh_ua

                    if ua_overflow:
                        qqm_overflow[qqm_overflow_size, 3] += 1
                        qqm_i[uh_ua] = 0

                    if uat_overflow:
                        qqm_overflow[qqm_overflow_size, 4] += 1
                        qqm_i[4+uh_ua] = 0
                    
                    qqm_overflow_size += 1
                
                if not ua_overflow:
                    qqm_i[uh_ua] += answered_correctly
                
                if not uat_overflow:
                    qqm_i[4+uh_ua] += 1

            else:
                qqm_i[uh_ua] += answered_correctly
                qqm_i[4+uh_ua] += 1
        
        return qqm_overflow_size

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly'] 
        user_answer = row[cm['user_answer']] if cm else row['user_answer'] 
        user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = np.zeros(self.last_n_max)

            acc_user = self.acc_dict[user_id]
            curr_idx = int(min(self.last_n_max-1, user_responses) if user_responses > 0 else 0)
            
            self.qqm_overflow['size'] = self.update_qqm(self.qqm, self.qqm_overflow['data'], self.qqm_overflow['size'], acc_user, curr_idx, content_id, answered_correctly)
            self.np_push(acc_user, user_responses, self.last_n_max, curr_idx, content_id+user_answer/10)

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']

        question_accuracy = feature_cache['question_accuracy']
        user_responses = feature_cache['user_responses']

        features = np_nans(len(self.columns))
        if 'qqm_history' not in feature_cache:
            feature_cache['qqm_history'] = np.zeros(self.last_n_max)
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                feature_cache['qqm_history'][:] = acc_user
                user_history_size = int(min(self.last_n_max, user_responses) if user_responses > 0 else 0)
                self.get_features_njit(features, self.qqm, self.qqm_overflow['data'], self.qqm_overflow['size'], content_id, self.last_n, acc_user, user_history_size, question_accuracy, self.sim_matrix, self.w_smooth, self.base_columns_size, self.samples_gt)
        
        return features

class GBMMeta(Accumulator):
    """
    Provides features generated by a meta model predicting the current answered_correctly using past history.

    Accumulates By: user_id

    Requires:
    - tag1: from MetadataFeats
    - user_responses: from UAccuracy
    - question_accuracy: from QAccuracy

    Features:
    """

    level = 2
    req_answered_correctly = True
    req_user_answer = True
    # columns defined in __init__

    def __init__(self, meta_model, pebg_embed_matrix, qqm, sim_matrix, base_acc_dict=None, last_n=10, w_smooth=20, updatable=True, suffix=""):
        self.meta_model = meta_model

        self.pebg_embed_dict = {i: q_embed for i, q_embed in tqdm(enumerate(pebg_embed_matrix))}
        self.pebg_embed_size = pebg_embed_matrix.shape[1]

        self.qqm = qqm
        self.sim_matrix = sim_matrix
        self.last_n = last_n
        self.w_smooth = w_smooth

        self.schema_columns = (
            [f"x_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            ['content_id', 'timestamp', 'tag1', 'answered_correctly', 'elapsed_time', 'down_time', 'question_had_explanation', 'user_answer']
        )
        self.schema = {v: i for i, v in enumerate(self.schema_columns)}
        self.meta_columns = (
            ['timestamp_diff', 'qqm_pred'] +
            [f"x_pebg_{i}" for i in range(self.pebg_embed_size)] + 
            ['prev_tag1', 'answered_correctly', 'elapsed_time', 'down_time', 'question_had_explanation'] +
            [f"y_pebg_{i}" for i in range(self.pebg_embed_size)] +
            ['tag1']
        )
        self.meta_schema = {v: i for i, v in enumerate(self.meta_columns)}
        #self.columns = [f'GBMMeta_{i}' for i in range(self.last_n)]
        self.columns = ['GBMMeta_mean', 'GBMMeta_sim_mean', 'GBMMeta_max', 'GBMMeta_min', 'GBMMeta_prev']
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def get_qqm_preds(qqm, cid_idx, ua_idx, content_id, last_n_arr, last_n_size, user_history, question_accuracy, sim_matrix, w_smooth, samples_gt=0):
        qqm_preds = np.full(last_n_size, question_accuracy)
        sim_weights = np.zeros(last_n_size)
        for i in range(user_history):
            uh_cid = int(last_n_arr[i][cid_idx]) # content id for i'th row in user history array
            uh_ua = int(last_n_arr[i][ua_idx]) # user answer for i'th row in user history array
            qqm_i = qqm[uh_cid][int(content_id)]
            qqm_preds[i] = (qqm_i[uh_ua] + w_smooth*question_accuracy)/(qqm_i[4+uh_ua] + w_smooth) if qqm_i[4+uh_ua] > samples_gt else np.nan
            sim_weights[i] = pdist_sim(sim_matrix, uh_cid, int(content_id), len(qqm))

        return qqm_preds, sim_weights

    @staticmethod
    @njit
    def weighted_meta_mean(meta_preds, sim_weights):
        weighted_meta = 0
        weighted_meta_total = 0
        
        for i in range(len(meta_preds)):
            sim = sim_weights[i]
            weighted_meta += meta_preds[i]*sim
            weighted_meta_total += sim
            
        weighted_meta_mean = weighted_meta/weighted_meta_total if weighted_meta_total > 0 else np.nan

        return weighted_meta_mean

    # TODO: need to keep a separate buffer for bundles. Once all questions in bundle have been asked and answered_correctly is available, update all at once
    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']

        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        user_answer = row[cm['user_answer']] if cm else row['user_answer']

        tag1 = feature_cache['tag1']
        user_responses = feature_cache['user_responses']

        # TODO: jit compile this whole section
        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = np.zeros((self.last_n, len(self.schema_columns)))

            curr_idx = int(min(self.last_n-1, user_responses) if user_responses > 0 else 0)
            
            pebg_embed = self.pebg_embed_dict[content_id]
            acc_user = self.acc_dict[user_id]
            
            # TODO: prior question logic fails for bundles
            if curr_idx > 0:
                acc_user[curr_idx-1, self.schema['elapsed_time']] = prior_question_elapsed_time
                acc_user[curr_idx-1, self.schema['question_had_explanation']] = prior_question_had_explanation

            # if there's no room, shift everything up
            if user_responses >= self.last_n:
                acc_user[:curr_idx :] = acc_user[1:, :]
                acc_user[curr_idx, :] = 0
            
            acc_user[curr_idx, :self.pebg_embed_size] = pebg_embed
            acc_user[curr_idx, self.schema['content_id']] = content_id
            acc_user[curr_idx, self.schema['timestamp']] = timestamp
            acc_user[curr_idx, self.schema['tag1']] = tag1
            acc_user[curr_idx, self.schema['answered_correctly']] = answered_correctly
            acc_user[curr_idx, self.schema['user_answer']] = user_answer

    # TODO: implement get_features. Remember to input the elapsed_time and question_had_explanation into a copy of acc_dict[user_id] (it's faster not to jit the copy) 
    # TODO: implement weighting schemes. Options are: simple average, weight decay (find the best weights using LR separately), cosine similarity, don't weight and just take raw last_n (i.e. 5 features if stored last 5)
    # TODO: add x (past) lag/down time, y (current) lag time to meta model
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        prior_question_elapsed_time = row[cm['prior_question_elapsed_time']] if cm else row['prior_question_elapsed_time']
        prior_question_had_explanation = row[cm['prior_question_had_explanation']] if cm else row['prior_question_had_explanation']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']

        tag1 = feature_cache['tag1']
        question_accuracy = feature_cache['question_accuracy']
        user_history = int(min(self.last_n, feature_cache['user_responses']))

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:
                acc_user_history = acc_user[:user_history]
                qqm_preds, sim_weights = self.get_qqm_preds(
                    self.qqm, self.schema['content_id'], self.schema['user_answer'],
                    content_id, acc_user_history, self.last_n, user_history, question_accuracy, self.sim_matrix, self.w_smooth
                )

                base_features = np.zeros((user_history, len(self.meta_columns)))
                base_features[:, self.meta_schema['timestamp_diff']] = timestamp - acc_user_history[:, self.schema['timestamp']]
                base_features[:, self.meta_schema['qqm_pred']] = qqm_preds[:user_history]
                base_features[:, self.meta_schema['x_pebg_0']:self.meta_schema[f'x_pebg_{self.pebg_embed_size-1}']+1] = acc_user_history[:, self.pebg_embed_size:]
                base_features[:, self.meta_schema['prev_tag1']] = acc_user_history[:, self.schema['tag1']]
                base_features[:, self.meta_schema['answered_correctly']] = acc_user_history[:, self.schema['answered_correctly']]
                base_features[:, self.meta_schema['elapsed_time']] = acc_user_history[:, self.schema['elapsed_time']]
                base_features[:, self.meta_schema['down_time']] = acc_user_history[:, self.schema['down_time']]
                base_features[:, self.meta_schema['question_had_explanation']] = acc_user_history[:, self.schema['question_had_explanation']]
                base_features[:, self.meta_schema['y_pebg_0']:self.meta_schema[f'y_pebg_{self.pebg_embed_size-1}']+1] = np.repeat(np.array([self.pebg_embed_dict[content_id]]), user_history, axis=0)
                base_features[:, self.meta_schema['tag1']] = tag1
                
                if user_history > 1:
                    base_features[user_history-2, self.meta_schema['elapsed_time']] = prior_question_elapsed_time
                    base_features[user_history-2, self.meta_schema['question_had_explanation']] = prior_question_had_explanation
                
                meta_preds = self.meta_model.predict(base_features)
                #if user_history < self.last_n:
                #    features = np.pad(meta_preds, (self.last_n - user_history, 0), 'constant', constant_values=np.nan)
                #else:
                #    features = meta_preds
                features = np_array(
                    np_mean(meta_preds),
                    self.weighted_meta_mean(meta_preds, sim_weights),
                    np_max(meta_preds),
                    np_min(meta_preds),
                    meta_preds[-1] if len(meta_preds) > 0 else np.nan
                )
        
        return features

class DiagnosticFeats(Accumulator):
    """
    Provides features from user's performance on diagnostic questions (first 30)
    """

    level = 2
    req_answered_correctly = True
    req_user_answer = True
    # columns defined in __init__

    def __init__(self, base_acc_dict=None, updatable=True, suffix=""):
        diagnostic_questions = [7900, 7876, 175, 1278, 2063, 2064, 2065, 3363, 3364, 3365, 2946, 2947, 2948, 2593, 2594, 2595, 4492, 4120, 4696, 6116, 6173, 6370, 6908, 6909, 6910, 6911, 7216, 7217, 7218, 7219]
        self.schema = {q: i for i, q in enumerate(diagnostic_questions)}
        self.schema_len = len(self.schema)
        self.columns = [f"diagnostic_ua_{suffix}" for suffix in self.schema]
        super().__init__(base_acc_dict or {}, updatable, suffix)

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_id = row[cm['content_id']] if cm else row['content_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        user_answer = row[cm['user_answer']] if cm else row['user_answer'] 
        #user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = np_nans(self.schema_len)
            
            #if user_responses < self.schema_len and content_id in self.schema:
            if content_id in self.schema:
                acc_user = self.acc_dict[user_id]
                acc_user[self.schema[content_id]] = user_answer

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']

        features = np_nans(len(self.columns))
        if content_type_id == 0 and user_id in self.acc_dict:
            features = self.acc_dict[user_id]

        return features

#endregion
#region ######################### LEVEL 3 ACCUMULATORS ##########################

class QNormUAccuracyLastN2(Accumulator):
    """
    Provides user accuracy features normalized by question difficulty for the last N responses by the user

    Accumulates By: user_id

    Requires:
    - question_accuracy: from QAccuracy
    - user_responses: from UAccuracy
    - timestamp_history: from TimestampHistory

    Features:
    - resid_mean_last{i}: user's average residual (cumulative / number of responses) over their last i records
    - resid_std_last{i}: standard deviation of user's residuals over their last i records
    """

    level = 2 
    req_answered_correctly = True
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, last_n=[3, 10, 30, 50, 100, 200], base_acc_dict=None, updatable=True, suffix=""):
        self.last_n = tuple(sorted(last_n))
        self.last_n_max = max(self.last_n)

        base_columns = ['resid_mean', 'resid_std']
        self.base_columns_size = len(base_columns)

        self.columns = [f"{col}_last{suffix}" for suffix in last_n for col in base_columns] #+ ['resid_mean_last_hour', 'resid_mean_last_day', 'resid_mean_last_week', 'resid_mean_last_month']
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def get_features_njit(features, last_n, user_history_arr, user_history_size, base_columns_size, user_timestamp_arr, timestamp):
        len_last_n = len(last_n)

        for i in range(len_last_n):
            last_n_size = last_n[i]
            user_history_start_idx = max(0, user_history_size - last_n_size)
            user_resid_history = user_history_arr[user_history_start_idx:user_history_start_idx+min(last_n_size, user_history_size)]

            features[i*base_columns_size:(i+1)*base_columns_size] = np_array(
                np_mean(user_resid_history),
                np_std(user_resid_history)
            )
        
        #timestamp_weights = 0.95**((timestamp - user_timestamp_arr[:user_history_size])/(1000*3600*24)) # 0.95**(days diff)
        #features[len_last_n*base_columns_size] = np.dot(user_resid_history[:user_history_size], timestamp_weights)/np.sum(timestamp_weights)

        # timestamp_diff = timestamp - user_timestamp_arr[:user_history_size]
        # month_idx = -1
        # week_idx = -1
        # day_idx = -1
        # hour_idx = -1
        # for i in range(len(timestamp_diff)):
        #     if month_idx == -1 and timestamp_diff[i] <= 1000*3600*24*30:
        #         month_idx = i
        #     if week_idx == -1 and timestamp_diff[i] <= 1000*3600*24*7:
        #         week_idx = i
        #     if day_idx == -1 and timestamp_diff[i] <= 1000*3600*24:
        #         day_idx = i
        #     if hour_idx == -1 and timestamp_diff[i] <= 1000*3600:
        #         hour_idx = i
        
        # features[len_last_n*base_columns_size] = np_mean(user_resid_history[max(hour_idx, 0):])
        # features[len_last_n*base_columns_size+1] = np_mean(user_resid_history[max(day_idx, 0):])
        # features[len_last_n*base_columns_size+2] = np_mean(user_resid_history[max(week_idx, 0):])
        # features[len_last_n*base_columns_size+3] = np_mean(user_resid_history[max(month_idx, 0):])

    # store residuals and answered_correctly in one array. Edge case: question accuracy = 0 and answered incorrectly. Store -0.0 to retain sign
    # to retrieve answered_correctly array: ac = (~np.signbit(arr)).astype(np.int8)
    @staticmethod
    @njit
    def np_push(arr, user_responses, last_n, curr_idx, answered_correctly, question_accuracy):
        resid = answered_correctly - question_accuracy
        if answered_correctly:
            v = resid
        else:
            v = resid if resid < 0 else -0.0

        if user_responses >= last_n:
            arr[:curr_idx] = arr[1:]
            arr[curr_idx:] = 0

        arr[curr_idx] = v

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return
        
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        answered_correctly = row[cm['answered_correctly']] if cm else row['answered_correctly']
        question_accuracy = feature_cache['question_accuracy']
        user_responses = feature_cache['user_responses']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = np.zeros(self.last_n_max)

            acc_user = self.acc_dict[user_id]
            curr_idx = int(min(self.last_n_max-1, user_responses) if user_responses > 0 else 0)

            self.np_push(acc_user, user_responses, self.last_n_max, curr_idx, answered_correctly, question_accuracy)

    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        timestamp = row[cm['timestamp']] if cm else row['timestamp']
        user_responses = feature_cache['user_responses']
        timestamp_history = feature_cache['timestamp_history']

        if 'qnorm_uaccuracy_history' not in feature_cache:
            feature_cache['qnorm_uaccuracy_history'] = np.zeros(self.last_n_max)
        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)

            if acc_user is not None:
                feature_cache['qnorm_uaccuracy_history'][:] = acc_user # need to pass in a copy, otherwise it will be modified by self.update
                user_history_size = int(min(self.last_n_max, user_responses) if user_responses > 0 else 0)
                self.get_features_njit(features, self.last_n, acc_user, user_history_size, self.base_columns_size, timestamp_history, timestamp)

        return features

class UAccuracyTrend(Accumulator):
    """
    Provides trend features for user accuracy

    Accumulates by: user_id

    Requires:
    - user_responses: from UAccuracy
    - qnorm_uaccuracy_history: from QNormUAccuracyLastN

    Features:
    - user_accuracy_trend_idx: user accuracy trend for the self.last_n records against index (not timestamp)
    - last_n_user_accuracy: user accuracy for the last self.last_n records
    - last_n_user_accuracy_std: standard deviation of user accuracy for the last self.last_n records
    """

    level = 3
    req_answered_correctly = True
    req_user_answer = False
    # columns defined in __init__

    def __init__(self, last_n=[4, 12, 30, 50, 100, 200], base_acc_dict=None, updatable=True, suffix=""):
        self.last_n = tuple(sorted(last_n))
        self.last_n_max = max(self.last_n) # must be <= last_n_max of QNormUAccuracyLastN
        self.acc_user_template = {last_n_i: np.zeros(last_n_i) for last_n_i in self.last_n}

        base_columns = ['user_accuracy_trend_idx', 'last_n_user_accuracy', 'last_n_user_accuracy_std']
        self.base_columns_size = len(base_columns)

        self.columns = [f"{col}_last{suffix}" for suffix in last_n for col in base_columns]
        super().__init__(base_acc_dict or {}, updatable, suffix)

    @staticmethod
    @njit
    def np_push(arr, user_responses, last_n, curr_idx, v):
        if user_responses >= last_n:
            arr[:curr_idx] = arr[1:]
            arr[curr_idx:] = 0

        arr[curr_idx] = v

    @staticmethod
    @njit
    def get_last_n_uaccuracy_mean(arr, last_n, user_responses, curr_idx):
        len_arr = len(arr)
        user_history_size = int(min(len_arr, user_responses) if user_responses > 0 else 0)
        user_history_start_idx = max(0, user_history_size - last_n)
        user_acc_history = arr[user_history_start_idx:user_history_start_idx+min(last_n, user_history_size)]
        
        return np_mean((~np.signbit(user_acc_history)).astype(np.int8))

    def update(self, row, feature_cache, cm={}):
        if not self.updatable:
            return

        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        user_responses = feature_cache['user_responses']
        qnorm_uaccuracy_history = feature_cache['qnorm_uaccuracy_history']

        if content_type_id == 0:
            if user_id not in self.acc_dict:
                self.acc_dict[user_id] = {k: v.copy() for k, v in self.acc_user_template.items()}
            
            acc_user = self.acc_dict[user_id]

            for last_n, user_accuracy_history in acc_user.items():
                curr_idx = int(min(last_n-1, user_responses) if user_responses > 0 else 0)
                last_n_uaccuracy_mean = self.get_last_n_uaccuracy_mean(qnorm_uaccuracy_history, last_n, user_responses, curr_idx) if type(qnorm_uaccuracy_history) == np.ndarray else np.nan
                self.np_push(user_accuracy_history, user_responses, last_n, curr_idx, last_n_uaccuracy_mean)
            
    def get_features(self, row, feature_cache, cm={}):
        user_id = row[cm['user_id']] if cm else row['user_id']
        content_type_id = row[cm['content_type_id']] if cm else row['content_type_id']
        user_responses = feature_cache['user_responses']

        features = np_nans(len(self.columns))
        if content_type_id == 0:
            acc_user = self.acc_dict.get(user_id, None)
            
            if acc_user is not None:                
                for i, (last_n, user_accuracy_history) in enumerate(acc_user.items()):
                    user_history_size = int(min(last_n, user_responses) if user_responses > 0 else 0)

                    #print(last_n, user_responses, user_history_size, "\n", user_accuracy_history)
                    
                    features[i*self.base_columns_size:(i+1)*self.base_columns_size] = np_array(
                        np_slope(np.arange(user_history_size)/100, user_accuracy_history[:user_history_size]) if user_history_size > 1 else 0,
                        user_accuracy_history[user_history_size-1] if user_history_size > 0 else np.nan,
                        np_std(user_accuracy_history[:user_history_size])
                    )
        return features

#endregion