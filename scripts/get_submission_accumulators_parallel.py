import numpy as np
import pandas as pd
import os 

import multiprocessing as mp
num_processes = mp.cpu_count()

# set up paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT_DIR, "data")

import sys
sys.path.append(ROOT_DIR)

from tqdm import tqdm
tqdm.pandas()

import kaggle_riiid.CVMaster as cvm
from kaggle_riiid.generate_features import *
from kaggle_riiid.utils import *

import click

def run_featurizer(chunk_id, input_folder, output_folder, update_only):
    train_df = pd.read_pickle(f"{input_folder}/train_{chunk_id}.pkl")
    questions = pd.read_csv("../data/questions.csv")
    lectures = pd.read_csv("../data/lectures.csv")
    q_accuracy_dict = read_from_pickle("../sandbox/accumulators/lgbm_starter_qaccuracy.pkl")
    pebg_embed = np.load(f"../sandbox/pebg/train_all_npz/embedding_4_8_ep1000.npz")
    #pebg_embed = np.load(f"../sandbox/pebg/train_all_npz/embedding_8_16_ep1000.npz")
    train_lec_median = read_from_pickle("../sandbox/train_lec_median.pkl")
    
    fr = Featurizer([
            RowIndex(),
            MetadataFeats(questions),
            PebgFeatures(pebg_embed["pro_final_repre"]),
            PebgCorrect(pebg_embed["pro_final_repre"]),
            PriorTimestampFeats(),
            UAccuracy(),
            UAccuracyTrend(last_n=[4,12,30,50,100,200]),
            QAccuracy(q_accuracy_dict, updatable=False),
            PreviousQTryAccum(),
            PreviousTag1Accum(),
            LectureFeats2(lectures, train_lec_median),
            #TimeCorrect(),
            QNormUAccuracy(),
            QNormUAccuracyLastN(last_n=[3,10,30,50,100,200]),
            SessionFeats(),
            #QQMFeats2(sim_matrix, qqm=qqm, last_n=[3,10,30,50,100,200]),
            DiagnosticFeats()
        ], keep_index=True
    )
    
    #pd.DataFrame(fr.featurize(train_df, "train"), columns=list(fr.feature_col_mappings)).to_pickle(f"{output_folder}/train_features_{chunk_id}.pkl")
    train_features = pd.DataFrame(fr.featurize(train_df, "train", update_only=update_only), columns=list(fr.feature_col_mappings))   
    
    if not update_only:            
        train_features.to_pickle(f"{output_folder}/train_features_{chunk_id}.pkl")
    
    write_to_pickle(fr.accumulators, f"{output_folder}/accum_parallel_{chunk_id}.pkl")
    
    #return train_features

# python get_submission_accumulators_parallel.py --num_chunks 12 --update_only
# python get_submission_accumulators_parallel.py --num_chunks 48 --update_only
# python get_submission_accumulators_parallel.py --num_chunks 12 --input_folder ../data/train_splits/fold0 --output_folder ../sandbox/submission/fold0 --create_features
# python get_submission_accumulators_parallel.py --num_chunks 48 --input_folder ../data/train_splits/fold1 --output_folder ../sandbox/submission/fold1 --create_features
@click.command()
@click.option("--num_chunks", type=int, default=None, help="Number of Chunks")
@click.option("--input_folder", default=f"{DATA_PATH}/train_splits/full_data", help="Input Folder")
@click.option("--output_folder", default=f"{ROOT_DIR}/sandbox/submission/full_data", help="Output Folder")
@click.option("--update_only/--create_features", default=False, help="Update only (T) or Create Features (F)")
def run_parallel(num_chunks, input_folder, output_folder, update_only):
    # def write_features(train_features):
    #     key = "single_key"
    #     train_features.to_hdf(f"{output_folder}/train_features_parallel.h5", key, mode="a", append=True, index=False)
    
    pool = mp.Pool(num_processes)
    for i in range(num_chunks):
        #pool.apply_async(run_featurizer, (str(i), input_folder, output_folder, update_only), callback=write_features)
        pool.apply_async(run_featurizer, (str(i), input_folder, output_folder, update_only))
    pool.close()
    pool.join()

def test_one_chunk():
    run_featurizer(0, "../data/train_splits/fold0", "../sandbox/submission/fold0", False)

if __name__ == "__main__":
    run_parallel()
    #test_one_chunk()
    
    
