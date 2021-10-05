import numpy as np
import pandas as pd
import copy
from time import sleep

from tqdm import tqdm

import random
from .utils import RandomDict

class CVSubsetter:
    """
    Manages splitting dataset into CV folds
    """

    def __init__(self, df):
        self.df = df

    # TODO: add params for random fn and seed for the dict.random_item() calls
    def create_cv_schema(self, folds=5, sample_rows=None, valid_size_approx=2500000, valid_exclusive_pct=0.25, frac_valid_range=[0.1, 0.9], frac_fn=np.random.uniform, seed=1337):
        def sample_ucd(uid_counts_dict):
            count = 0
            new_uid_counts_dict = RandomDict()

            while count < sample_rows:
                sample = uid_counts_dict.random_item()
                new_uid_counts_dict[sample[0]] = sample[1]
                uid_counts_dict.pop(sample[0], None)
                count += sample[1]

            return new_uid_counts_dict

        def generate_fold(train_ids, uid_counts_dict):
            valid_exclusive_count = 0
            share_valid_count = 0
            cv_fold_schema = {'uid_train_exclusive': {}, 'uid_shared': {}, 'uid_valid_exclusive': {}}

            for id in train_ids:
                cv_fold_schema['uid_train_exclusive'][id] = uid_counts_dict[id]
                uid_counts_dict.pop(id, None)

            if sum([val[1] for val in uid_counts_dict.values]) < valid_size_approx:
                return None

            # pick some user_ids to only appear in valid:
            valid_exclusive_threshold = int(valid_exclusive_pct*valid_size_approx)
            while valid_exclusive_count < valid_exclusive_threshold and len(uid_counts_dict) > 0:
                sample = uid_counts_dict.random_item() #format: (user_id, count)
                cv_fold_schema['uid_valid_exclusive'][sample[0]] = {'count': sample[1]}
                valid_exclusive_count += sample[1]
                uid_counts_dict.pop(sample[0], None)

            while share_valid_count < valid_size_approx - valid_exclusive_count - len(train_ids) and len(uid_counts_dict) > 0:
                sample = uid_counts_dict.random_item() #format: (user_id, count)
                valid_frac = frac_fn(frac_valid_range[0], frac_valid_range[1])
                valid_count = round(sample[1]*valid_frac)
                cv_fold_schema['uid_shared'][sample[0]] = {'count': sample[1],
                                                            'valid_frac': valid_frac,
                                                            'valid_count': valid_count,
                                                            'train_count': sample[1] - valid_count
                                                        }
                share_valid_count += valid_count
                uid_counts_dict.pop(sample[0], None)
            
            cv_fold_schema['uid_train_exclusive'].update({val[0]: {'count': val[1]} for val in uid_counts_dict.values})
            return cv_fold_schema

        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        uid_counts = pd.DataFrame(self.df.groupby('user_id').size().rename('count'))

        ucd = RandomDict()
        for user in uid_counts.itertuples():
            ucd[user.Index] = user.count

        if sample_rows and sample_rows > 0 and sample_rows < uid_counts['count'].sum():
            ucd = sample_ucd(copy.deepcopy(ucd))

        # users can only be in 1 validation set, so every time fold is generated append shared and validation exclusive ids to this list 
        # so in every subsequent fold, those ids end up in train exclusive
        train_ids = [] 
        cv_fold_schemas = []
        
        for _ in range(folds):
            if len(train_ids) < len(ucd):
                fold = generate_fold(train_ids, copy.deepcopy(ucd))
                if fold:
                    cv_fold_schemas.append(fold)
                    train_ids = train_ids + list(fold['uid_shared'].keys()) + list(fold['uid_valid_exclusive'].keys())
                else:
                    break
            # TODO: implement else using logging to log a warning that cv fold not created due to running out of user ids

        return cv_fold_schemas

    def generate_train_valid_from_schema(self, cv_fold_schema):
        train_fold = self.df[self.df['user_id'].isin(cv_fold_schema['uid_train_exclusive'].keys())]
        valid_fold = self.df[self.df['user_id'].isin(cv_fold_schema['uid_valid_exclusive'].keys())]
        train_valid_shared_df = self.df[self.df['user_id'].isin(cv_fold_schema['uid_shared'].keys())]

        train_fold_shared = (
            train_valid_shared_df.groupby('user_id')
            .progress_apply(lambda x: x.head(cv_fold_schema['uid_shared'][x['user_id'].values[0]]['train_count']))
            .droplevel(0)
        )

        valid_fold_shared = (
            train_valid_shared_df.groupby('user_id')
            .progress_apply(lambda x: x.tail(cv_fold_schema['uid_shared'][x['user_id'].values[0]]['valid_count']))
            .droplevel(0)
        )
            
        train_fold = train_fold.append(train_fold_shared)
        valid_fold = valid_fold.append(valid_fold_shared)

        return train_fold, valid_fold

class VaaS:
    """
    VaaS: Validation (set) as a Service

    Simulates riiideducation's serving the test set in groups
    """

    def __init__(self, valid_set, sleep=0, rgs_fn = random.randint, rgs_fn_params = {'a': 20, 'b': 64}):
        # TODO: right now assuming every user starts at the same time. 
        # Consider perturbing timestamp by random amount per user before sorting by batch_id and timestamp
        self.valid_set = valid_set
        self.sleep = sleep
        self.rgs_fn = rgs_fn
        self.rgs_fn_params = rgs_fn_params
        self.valid_set['batch_id'] = (self.valid_set
                                      .groupby('user_id')['timestamp']
                                      .progress_transform(lambda x: pd.factorize(x, sort=True)[0])                               
                                      .values
                                     )
        self.valid_set['prior_group_answers_correct'] = np.nan
        self.valid_set['prior_group_responses'] = np.nan
        self.valid_set['prior_group_answers_correct'] = self.valid_set['prior_group_answers_correct'].astype(object)
        self.valid_set['prior_group_responses'] = self.valid_set['prior_group_responses'].astype(object)

        self.valid_set = self.valid_set.sort_values(by=['batch_id', 'timestamp'])
        self.regen_groups()
    
    def __iter__(self):
        self.iterator = iter(self.valid_set_groups)

        return self
        
    def __next__(self):
        if self.sleep:
            sleep(self.sleep)
            
        return next(self.iterator)
    
    def regen_groups(self):
        """
        Adds group_num, prior_group_answers_correct, prior_group_responses
        Runs in <10s on 2.5m rows due to fast construction using numpy
        """
        prior_answered_correctly = str([])      
        prior_responses = str([])
        answered_correctly_values = self.valid_set['answered_correctly'].values
        user_answer_values = self.valid_set['user_answer'].values

        group_nums, group_nums_schema = self._create_group_nums(self.valid_set.groupby('batch_id').size(), self.rgs_fn, self.rgs_fn_params)
        self.valid_set['group_num'] = group_nums
        
        df_prior_answered_correctly = []
        df_prior_responses = []
        idx = 0
        for group_num in group_nums_schema.keys():
            group_size = group_nums_schema[group_num]
            pac_list = [np.nan]*group_size
            pr_list = [np.nan]*group_size

            pac_list[0] = prior_answered_correctly
            pr_list[0] = prior_responses
            df_prior_answered_correctly += pac_list
            df_prior_responses += pr_list

            prior_answered_correctly = str(list(answered_correctly_values[idx:idx+group_size]))
            prior_responses = str(list(user_answer_values[idx:idx+group_size]))
            idx += group_size

        assert len(df_prior_answered_correctly) == self.valid_set.shape[0]

        self.valid_set['prior_group_answers_correct'] = df_prior_answered_correctly
        self.valid_set['prior_group_responses'] = df_prior_responses
        self.valid_set_groups = self.valid_set.groupby('group_num')
    
    def _create_group_nums(self, batch_id_counts, rgs_fn, rgs_fn_params):
        total_rows = np.sum(batch_id_counts.values)
        group_nums = []
        group_nums_schema = {}
        current_group_num = 0
        group_count = 0
        count = 0
        current_batch = 0

        pbar = tqdm(total=total_rows)
        while count < total_rows:
            random_group_size = rgs_fn(**rgs_fn_params)

            # never go past the batch_id per batch; otherwise, chance of batch containing 2 records from same user_id
            if group_count + random_group_size < batch_id_counts.values[current_batch]: 
                group_nums += [current_group_num]*random_group_size
                group_count += random_group_size
            else:
                random_group_size = batch_id_counts.values[current_batch] - group_count
                group_nums += [current_group_num]*random_group_size
                current_batch += 1
                group_count = 0

            group_nums_schema[current_group_num] = random_group_size
            current_group_num += 1
            count += random_group_size
            pbar.update(random_group_size)

        pbar.close()
        return np.array(group_nums), group_nums_schema