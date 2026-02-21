import numpy as np
import pandas as pd
import argparse
import os
from TimeCast import TimeCast
import dill
import time

import sys


sys.path.append("_src")
import utils

EPS=5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output Data
    parser.add_argument("--input_tag", type=str)
    parser.add_argument("--out_dir", type=str, default="out/tmp/")
    parser.add_argument("--import_type", type=str, default="clean_data_rul")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--feature_cols", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--use_dict_inputs", action="store_true")
    parser.add_argument("--use_sequence_inputs", action="store_true")
    parser.add_argument("--split_ind", type=int, default=0)  # from 0 to 5 or 10
    parser.add_argument("--valid_ratio", type=float, default=0.1)  # from 0 to 5 or 10

    # model
    parser.add_argument(
        "--alpha", type=float, default=0.1
    )  # sparsity for graphical lasso
    parser.add_argument("--beta_w", type=float, default=0.9)  # loss weight
    parser.add_argument("--init_k", type=int, default=6)
    parser.add_argument("--niter", type=int, default=10)
 
    args = parser.parse_args()
    # make output dir
    outputdir = args.out_dir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Original data
    train_dataset, test_dataset = utils.import_dataset(args)

    # Z-normalization
    train_dataset, test_dataset = utils.znorm_train_test_datasets(train_dataset, test_dataset, args)

    # validation split
    train_dataset, validation_dataset = utils.data_split(train_dataset,args.valid_ratio)

    # Transform dataset for input
    print("Train samples")
    train_data, train_result_df = utils.make_samples(
        train_dataset,
        args,
    )  # data: dict object

    print("Validation samples")
    validation_data, valid_result_df = utils.make_samples(
        validation_dataset,
        args,
    )  # data: dict object

    print("\nTest samples")
    test_data, test_result_df = utils.make_samples(
        test_dataset,
        args,
    )  # data: dict object

    print(len(train_data["features"]))
    print(train_data["features"][0].shape)
    print(len(train_data["label"]))
    print(train_data["label"][0].shape)

    params = vars(args)

    # train
    TimeCast=TimeCast(
        params["alpha"],
        params["beta_w"],
        params["init_k"],
        max_iter=params["niter"],
    )

    st = time.perf_counter()
    train_time = TimeCast.tc_fit(train_data)
    train_time_perf=time.perf_counter()-st
    # predict
    pred_train, cluster_ids_train, inc_stage_ids_train, fin_stage_ids_train,pred_time_train,_ = TimeCast.tc_scan(
        train_data,
        update=False,
    )

    # predict validation
    pred_valid, cluster_ids_valid, inc_stage_ids_valid, fin_stage_ids_valid, pred_time_valid,_ = TimeCast.tc_scan(
        validation_data,
        update=False,
    )

    print("TEST START")
    st = time.perf_counter()
    pred_test, cluster_ids_test, inc_stage_ids_test, fin_stage_ids_test, pred_time_test, n_stage_hist = TimeCast.tc_scan(
        test_data,
        update=True)
    pred_time_test_perf=time.perf_counter()-st


    # write result
    train_result_df["pred"] = pred_train
    train_result_df["test"] = 0
    train_result_df["cluster_id"] = cluster_ids_train
    train_result_df["inc_stage_id"] = inc_stage_ids_train
    train_result_df["fin_stage_id"] = fin_stage_ids_train

    valid_result_df["pred"] = pred_valid
    valid_result_df["test"] = -1
    valid_result_df["cluster_id"] = cluster_ids_valid
    valid_result_df["inc_stage_id"] = inc_stage_ids_valid
    valid_result_df["fin_stage_id"] = fin_stage_ids_valid

    test_result_df["pred"] = pred_test
    test_result_df["test"] = 1
    test_result_df["cluster_id"] = cluster_ids_test
    test_result_df["inc_stage_id"] = inc_stage_ids_test
    test_result_df["fin_stage_id"] = fin_stage_ids_test


    result_df = pd.concat([train_result_df, test_result_df], axis=0)
    result_df["method"] = "TimeCast"

    # Time-to-Event prediction
    mape_v = utils.mean_absolute_percentage_error(valid_result_df.loc[valid_result_df["actual"]>EPS,"actual"], valid_result_df.loc[valid_result_df["actual"]>EPS,"pred"])
    rmspe_v = utils.root_mean_squared_percentage_error(valid_result_df.loc[valid_result_df["actual"]>EPS,"actual"], valid_result_df.loc[valid_result_df["actual"]>EPS,"pred"])


    mape_t = utils.mean_absolute_percentage_error(test_result_df.loc[test_result_df["actual"]>EPS,"actual"],test_result_df.loc[test_result_df["actual"]>EPS,"pred"])
    rmspe_t = utils.root_mean_squared_percentage_error(test_result_df.loc[test_result_df["actual"]>EPS,"actual"], test_result_df.loc[test_result_df["actual"]>EPS,"pred"])

    
    summary_dict = {
        "train_time":train_time,
        "pred_time_test": pred_time_test,
        "train_time_perf":train_time_perf,
        "pred_time_test_perf": pred_time_test_perf,
        "MAPE-valid": mape_v,
        "RMSPE-valid": rmspe_v,
        "MAPE-test": mape_t,
        "RMSPE-test": rmspe_t,
        "n_stage_hist":n_stage_hist,
    }

    # save result
    result_df.to_csv(params["out_dir"] + "/result.csv.gz", index=False)
    utils.save_as_json(summary_dict, params["out_dir"] + "/summary.json")
    dill.dump(TimeCast, open(params["out_dir"] + "/TimeCast.dill", "wb"))

    print("Summary:")
    print(f"MAPE:{mape_t}")
    print(f"RMSPE:{rmspe_t}")
    print(params["out_dir"])
