import numpy as np
import pandas as pd
import copy

from tqdm import tqdm
import time

import wiener as wiener_module
import graphical_lasso as graphical_lasso_module
import scipy
import sys

sys.path.append("_src")
import utils

ZERO = 1e-100
EPS = ZERO
SEED = 317
COV_CENTERED = False
SMOOTH_RATIO = 0.1
ITER_MRF = 500

STAGE_VERBOSE = False
ALTERNATE_VERBOSE = False

class TimeCast:
    def __init__(self, alpha: float, beta_w: float, k: int, max_iter: int):
        self.alpha = alpha
        self.l_wiener = beta_w
        self.l_mrf = 1.0 - beta_w
        self.ini_wiener_ll = 1.0
        self.ini_mrf_ll = 1.0
        self.k = k
        self.max_iter = max_iter
        self.opt_models = {}
        self.n_dim = None
        self.n_samples = None
        np.random.seed(SEED)

    def estimate_mrf(self, seq, verbose=False):
        # seq: (n_dim, duration)
        gl = graphical_lasso_module.graphical_lasso(
            alpha=self.alpha,
            beta=0,
            max_iter=ITER_MRF,
            psi="laplacian",
            assume_centered=COV_CENTERED,
            verbose=verbose,
            return_history=True,
        )
        gl.fit(seq.T)
        mrf_objective = (-1) * gl.history_[-1][0] 

        if verbose:
            print("mrf_objective:")
            print(mrf_objective)
        
        gl.mean_=np.mean(seq,axis=1) 
        gl.logdet_cov_ = np.linalg.det(gl.covariance_[0]) # det(A^{-1}) = 1/det(A), logdet_cov_ = -logdet_precision_ 
        s, _ = scipy.linalg.eigh(gl.covariance_[0], lower=True, check_finite=True)
        eps = _eigvalsh_to_eps(s, None, None)
        d = s[s > eps]
        gl.log_pdet = np.sum(np.log(d))

        return gl, gl.precision_, gl.covariance_, mrf_objective

    def estimate_stages_tensor(
        self,
        data_features,
        data_labels,
        mrf_models,
        wiener_models,
        verbose=False,
    ):
        """
        Estimate latent stages

        Parameters
        -----------
        data : ndarray
            [# windows , duration, sensor dimenstion], float
        mrf_models : object
            mrf_models[i].precision_[0] means a MRF precision matrix [dim x dim]
        Returns
        -------
        estimated stage assignments [duration]
        """

        n_stages = len(mrf_models)
        path_len = data_features.shape[0]

        path_prob = np.zeros(
            (3, n_stages)
        )  # likelihoods for the optimal path to stage s by time t

        path_prob_hist = np.zeros(
            (3, n_stages, path_len)
        )  # likelihoods for the optimal path to stage s by time t

        optimal_path = np.full((n_stages, path_len), -1)  # the optimal path to stage s
        wiener_cur_state = np.zeros(n_stages)

        label = data_labels[0, 0]
        feature = data_features[0].reshape(-1, 1)

        for s in range(n_stages):
            obj_mrf = self.l_mrf * self.multivariate_normal_logll(mrf_models[s],data_features[0]) /self.ini_mrf_ll
            obj_wiener = (
                    self.l_wiener
                    * (-1)
                    * (wiener_models[s].w_obj(feature, label, cur_state=0) + ZERO)
                    / self.ini_wiener_ll
                )

            path_prob[0, s] = obj_mrf + obj_wiener
            path_prob[1, s] = obj_mrf
            path_prob[2, s] = obj_wiener
            optimal_path[s, 0] = s
            wiener_cur_state[s] += wiener_models[s].drift
            path_prob_hist[:, s, 0] = [obj_mrf + obj_wiener, obj_mrf, obj_wiener]

        for t in range(1, path_len):
            feature = data_features[t].reshape(-1, 1)
            label = data_labels[t, 0]

            if verbose:
                print(f"time:{t}")

            path_prob_cur = np.zeros(
                (3, n_stages)
            )  # likelihoods for the stage s at the time t

            opt_path_cur = np.zeros(
                (n_stages, t + 1)
            )  # the optimal path to the current stage s by the time t

            cur_state_updated = copy.deepcopy(wiener_cur_state)
            for s in range(n_stages):
                # mi = 0 if s - 1 < 0 else s - 1 # only one hop
                mi = 0  # all stages whose upper id
                ma = s + 1 
                # ma = n_stages #TODO

                opt_stage = (
                    np.argmax(path_prob[0, mi:ma]) + mi
                )  # an optimal past stage for stage s

                opt_path_cur[s, :t] = optimal_path[opt_stage, :t]
                opt_path_cur[s, t] = s
                if opt_stage == s:
                    cur_state = wiener_cur_state[opt_stage]
                else:
                    cur_state = 0

                cur_state_updated[s] = cur_state + wiener_models[s].drift

                ob_mrf = self.l_mrf * self.multivariate_normal_logll(mrf_models[s],data_features[t]) /self.ini_mrf_ll
                ob_wiener = (
                        self.l_wiener
                        * (-1)
                        * (wiener_models[s].w_obj(feature, label, cur_state=0) + ZERO)
                        / self.ini_wiener_ll
                    )

                path_prob_cur[0, s] = ob_mrf + ob_wiener + path_prob[0, opt_stage]
                path_prob_hist[0, s, t] = ob_mrf + ob_wiener

                path_prob_cur[1, s] = ob_mrf + path_prob[1, opt_stage]
                path_prob_hist[1, s, t] = ob_mrf

                path_prob_cur[2, s] = ob_wiener + path_prob[2, opt_stage]
                path_prob_hist[2, s, t] = ob_wiener

                if verbose:
                    print(f"label:{label}")
                    print(
                        f"{s}: \nall:{ob_mrf+ob_wiener}\nmrf:{ob_mrf}, winer:{ob_wiener},"
                    )

            path_prob = path_prob_cur
            optimal_path[:, : t + 1] = opt_path_cur
            wiener_cur_state = cur_state_updated

            if verbose:
                print(f"optimam: {path_prob}")
                print(optimal_path)

        if verbose:
            print("result")
            print(path_prob)
            print(optimal_path)

        opt = np.argmax(path_prob[0, :])
        opt_val_cum = path_prob[0, opt]
        opt_val_mrf = path_prob[1, opt]
        opt_val_wiener = path_prob[2, opt]
        opt_val_hist = path_prob_hist[:, opt, :]

        final_optimal_path = optimal_path[opt]

        return (
            final_optimal_path,
            opt_val_hist,
            opt_val_cum,
            opt_val_mrf,
            opt_val_wiener,
        )

    def tc_fit(self, data):
        
        tic = time.process_time()
        # model_initialization
        candidate_models, cost_table, cost = self.model_initialization(
            data, initial_n_stage=self.k, verbose=True,
        )
        #
        norm_mrf = (
            cost_table["mrf_ll"].sum() / self.l_mrf if self.l_mrf != 0 else 1
        )
        norm_wiener = (
            (cost_table["wiener_ll"].sum() / self.l_wiener)
            if self.l_wiener != 0
            else 1
        )
    
        sparse_norm = np.sum([graphical_lasso_module.l1_od_norm(candidate_models["mrf"][key_].precision_[0]) for key_ in candidate_models["mrf"]])
        self.ini_mrf_ll = abs(norm_mrf - sparse_norm*self.alpha)
        self.ini_wiener_ll = abs(norm_wiener)
        print("Initial ll")
        print(f"Initial mrf:{self.ini_mrf_ll}")
        print(f"Inital Wiener{self.ini_wiener_ll}")
        self.opt_models = copy.deepcopy(candidate_models)
        (
            updated_models,
            cost_table,
            cost,
            objective_values,
        ) = self.alternate_optimization(
            data, self.opt_models, verbose=ALTERNATE_VERBOSE
        )
        self.opt_models = copy.deepcopy(updated_models)
        self.objective_values = objective_values
        train_time = time.process_time() - tic

        return train_time


    def build_single_model(self, data):
        opt_models = {}
        feature_mat = []
        diff_seq_single = []

        data_features = data["features"]
        data_label = data["label"]
        for dat, label in zip(data_features, data_label):
            for da in dat:
                feature_mat.append(da.reshape(-1))
            diff_seq_single.extend(label)

        feature_mat = np.array(feature_mat).T
        diff_seq_single = np.array(diff_seq_single).reshape(1, -1)

        single_wiener = wiener_module.wiener()
        single_wiener.projection_fit(feature_mat, diff_seq_single, verbose=True)
        opt_models[0] = single_wiener

        return opt_models

    def build_wiener_model(self, data_fs, data_ls, verbose=True):
        feature_mat = []
        diff_seq = []

        for dat, label in zip(data_fs, data_ls):
            feature = dat.reshape(-1)
            feature_mat.append(feature)
            diff_seq.extend(label)
        feature_mat = np.array(feature_mat).T
        diff_seq = np.array(diff_seq).reshape(1, -1)
        if verbose:
            print(f"feature_mat:{feature_mat.shape}")
            print(f"diff_seq:{diff_seq.shape}")

        wiener = wiener_module.wiener()
        wiener.projection_fit(feature_mat, diff_seq, verbose=verbose)
        if verbose:
            print(wiener.mean_drift)
            print(wiener.b_std)

        return wiener

    def model_initialization(self, data, initial_n_stage=1, verbose=False):
        data_features = data["features"]
        data_label = data["label"]
        initial_models = {}
        cost_tables = []
        n_dim = data_features[0].shape[2]
        self.n_dim = n_dim
        if initial_n_stage == 1:
            print("++INITIALIZE MRF MODELS++")
            initial_models["mrf"] = {}
            initial_mrf_models_all = {}
            data_mrf_all_list = [data_f.reshape(-1, self.n_dim) for data_f in data_features]
            data_mrf_all = np.concatenate(data_mrf_all_list, axis=0)
            mrf_model, _, _, _ = self.estimate_mrf(data_mrf_all.T, verbose=False)
            initial_models["mrf"][0] = mrf_model

            print("++INITIALIZE WIENER MODELS++")
            initial_models["wiener"] = {}
            wiener_model = self.build_single_model(data)[0]
            initial_models["wiener"][0] = wiener_model
            print("wiener pramaeters")
            print(wiener_model.mean_drift)
            print(wiener_model.b_std)

        else:
            print("++INITIALIZE MODELS++")
            initial_models = {}

            duration = data_features[0].shape[1]
            all_data = []
            all_ind = []
            all_label = []
            all_len = 0
            for data_f, data_l in zip(data_features, data_label):
                n_window = data_f.shape[0]
                all_len += n_window
                all_data.append(data_f)
                all_ind.append(np.arange(n_window))
                all_label.append(data_l)
            all_data = np.concatenate(all_data, axis=0)            
            all_ind = np.concatenate(all_ind, axis=0)
            all_label = np.concatenate(all_label, axis=0)

            # choose by lsh
            ini_data = []    
            ini_label=[]    
            ini_order = []
            set_binary = []
            set_binary.append(np.ones(initial_n_stage,dtype=int))
            # generate hyperplane
            # np.zeros(dim x initial_n_stage)
            plane_norms = np.random.rand(initial_n_stage, self.n_dim**2) - .5
            print(plane_norms)
            random_ind_arr = np.arange(all_data.shape[0])
            np.random.shuffle(random_ind_arr)
            for r_ind in random_ind_arr:
                dat = all_data[r_ind]
                la =  all_label[r_ind]
                hash_val = graphical_lasso_module.empirical_covariance(dat, assume_centered=COV_CENTERED).reshape(-1)
                hash_val = np.dot(plane_norms,hash_val)
                print(hash_val)
                hash_val = np.array(hash_val>0,dtype=int)
                print(f"{r_ind}:{hash_val}")
                check_collision = 0
                for bina in set_binary:
                    if np.sum(hash_val^bina) == 0:
                        print("==collision==")
                        check_collision+=1
                if check_collision == 0:                
                    set_binary.append(hash_val)
                    ini_data.append(dat)
                    ini_label.append(la)
                    ini_order.append(int(la))
                
                if len(set_binary) > initial_n_stage:
                    print(set_binary)
                    print(ini_order)
                    break
            initial_models["mrf"] = {}
            initial_models["wiener"] = {}
            for ini_id, sorted_id in enumerate(np.argsort(ini_order)[::-1]):
                initial_models["mrf"][ini_id] = self.estimate_mrf(ini_data[sorted_id].T)[0]
                initial_models["wiener"][ini_id] = self.build_wiener_model(
                    np.array([ini_data[sorted_id]]),
                    np.array([ini_label[sorted_id]]),
                    verbose=False,
                )
            

        print("++CALCURATION COST")
        overall_ll = 0
        overall_mrf = 0
        overall_wiener = 0
        for s_id, (seq_tensor, label) in tqdm(
            enumerate(zip(data_features, data_label))
        ):
            # for m_id in optimized_mrf_models:
            (
                optimal_path,
                opt_ll_hist,
                opt_ll,
                opt_mrf,
                opt_wiener,
            ) = self.estimate_stages_tensor(
                seq_tensor,
                label,
                initial_models["mrf"],
                initial_models["wiener"],
                verbose=False,
            )
            _cost_table = np.zeros((5, optimal_path.shape[0]))
            _cost_table[0, :] = s_id
            _cost_table[1] = optimal_path
            _cost_table[2:] = opt_ll_hist
            cost_tables.append(_cost_table)
            overall_ll += opt_ll
            overall_mrf += opt_mrf
            overall_wiener += opt_wiener
            if verbose:
                print(f"DataID:{s_id}")
                print(f"Path score:{opt_ll}")
                print("opt_ll_hist last4")
                print(opt_ll_hist[:, -4:])
                print(optimal_path)
        initial_cost_table = pd.DataFrame(
            data=np.concatenate(cost_tables, axis=1).T,
            columns=[
                "seq_id",
                "stage_id",
                "all_ll",
                "mrf_ll",
                "wiener_ll",
            ],
        )

        return initial_models, initial_cost_table, overall_ll

    def alternate_optimization(self, data, models, verbose=False):
        # data_features: n_seq(n_window x duration x n_dim)
        # estimate candidate_stages & refine model
        data_features = data["features"]
        data_label = data["label"]
        objective_values = []

        optimized_models = copy.deepcopy(models)
        n_stage= len(optimized_models["mrf"])

        old_optimal_path_set = []

        old_ll = 0
        old_w_ll = 0
        old_mrf_ll = 0
        old_sparsity=np.zeros(n_stage)
        for it in range(self.max_iter):
            print(f"START ITER #{it}")
            overall_ll = 0
            overall_mrf = 0
            overall_wiener = 0
            optimal_path_set = []
            assigned_data = {
                stage_: [] for stage_ in optimized_models["mrf"]
            }
            assigned_diff_data = {
                stage_: [] for stage_ in optimized_models["mrf"]
            }
            assigned_feature = {
                stage_: [] for stage_ in optimized_models["mrf"]
            }

            cost_tables = []
            self.n_samples=np.zeros(n_stage,dtype=int)
            sparsity = np.zeros(n_stage)
            for stage_id in range(n_stage):
                m_model = optimized_models["mrf"][stage_id]
                sparsity[stage_id] = graphical_lasso_module.l1_od_norm(m_model.precision_[0])/ self.ini_mrf_ll


            # 1. estimate candidate_stages
            for s_id, (seq_tensor, label) in tqdm(
                enumerate(zip(data_features, data_label)),
                desc="Estimate Candidate Stages",
            ):
                c_opt_ll = -np.inf
                (
                    optimal_path,
                    opt_ll_hist,
                    opt_ll,
                    opt_mrf,
                    opt_wiener,
                ) = self.estimate_stages_tensor(
                    seq_tensor,
                    label,
                    optimized_models["mrf"],
                    optimized_models["wiener"],
                    verbose=STAGE_VERBOSE,
                )
                c_opt_ll = opt_ll
                c_opt_mrf = opt_mrf
                c_opt_wiener = opt_wiener
                c_optimal_path = optimal_path

                optimal_path_set.append(c_optimal_path)
                overall_ll += c_opt_ll
                overall_mrf += c_opt_mrf
                overall_wiener += c_opt_wiener

                _cost_table = np.zeros((5, optimal_path.shape[0]))
                _cost_table[0, :] = s_id
                _cost_table[1] = optimal_path
                _cost_table[2:] = opt_ll_hist
                cost_tables.append(_cost_table)
                for stage_key in optimized_models["mrf"]:
                    ns = np.sum(c_optimal_path == stage_key)
                    self.n_samples[stage_key] += ns
                    if ns > 0:
                        assigned_data[stage_key].extend(
                            seq_tensor[c_optimal_path == stage_key]
                        )
                        for da in seq_tensor[c_optimal_path == stage_key]:
                            assigned_feature[stage_key].append(da.reshape(-1))
                        drift_array = self.get_drift_array(
                            label, c_optimal_path, stage_key
                        )
                        assigned_diff_data[stage_key].extend(drift_array)

                if verbose:
                    print(f"DataID:{s_id}, Path score:{c_opt_ll}")
            
            all_sparsity = np.sum(sparsity)*self.alpha
            gap = overall_ll - old_ll

            print(f"OVERALL Objective at ITER #{it}: {overall_ll}")
            print(f"gap(improve+):{gap}")
            print(f"pre Objective:{old_ll}")
            print(
                f"MRF:{overall_mrf} <- {old_mrf_ll}, gap(improve+):{overall_mrf-old_mrf_ll}"
            )
            print(
                f"WIENER:{overall_wiener} <- {old_w_ll}, gap(improve+):{overall_wiener-old_w_ll}"
            )
            if verbose:
                print(f"Refined Objective at ITER #{it}: {overall_ll-all_sparsity}")
                print(f"gap(improve+):{gap}")
                print(f"pre Objective:{old_ll-np.sum(old_sparsity)*self.alpha}")
                print(
                    f"Sparsity (L1-norm):{all_sparsity} <- {np.sum(old_sparsity)}, gap(improve-):{all_sparsity-np.sum(old_sparsity)}"
                )
                print(f"{sparsity}\n{old_sparsity}")


            objective_values.append([overall_ll, overall_mrf, overall_wiener])
            old_ll = overall_ll
            old_mrf_ll = overall_mrf
            old_w_ll = overall_wiener
            old_sparsity = sparsity

            if abs(gap) < EPS:
                print("Converged (below threshold).")
                break

            flattened_optimal_path_set = np.concatenate(optimal_path_set, axis=0)
            if np.array_equal(old_optimal_path_set, flattened_optimal_path_set):
                print("Converged (paths unchanged).")
                break

            old_optimal_path_set = copy.deepcopy(flattened_optimal_path_set)

            optimized_mrf_models_new = {}
            optimized_wiener_models_new = {}

            for stage_key in optimized_models["mrf"]:
                if self.n_samples[stage_key] == 0:
                    print(f"stage #{stage_key} is empty")
                    for i in range(1, stage_key + 1):
                        pre_stage_key = stage_key - i
                        pre_n_samp = self.n_samples[pre_stage_key]
                        if pre_n_samp > 0:
                            sampling_ind = np.random.choice(pre_n_samp, int(pre_n_samp * SMOOTH_RATIO), replace=False)
                            for s_ind in sampling_ind:
                                assigned_data[stage_key].append(assigned_data[pre_stage_key][s_ind])
                                assigned_diff_data[stage_key].extend(np.array([assigned_diff_data[pre_stage_key][s_ind]]))
                                assigned_feature[stage_key].append(assigned_feature[pre_stage_key][s_ind])
                            self.n_samples[stage_key] += len(sampling_ind)
                            break
                    for i in range(1, n_stage - stage_key):
                        next_stage_key = stage_key + i
                        next_n_samp = self.n_samples[next_stage_key]
                        if next_n_samp > 0:
                            sampling_ind = np.random.choice(next_n_samp, int(next_n_samp * SMOOTH_RATIO), replace=False)
                            for s_ind in sampling_ind:
                                assigned_data[stage_key].append(assigned_data[next_stage_key][s_ind])
                                assigned_diff_data[stage_key].extend(np.array([assigned_diff_data[next_stage_key][s_ind]]))
                                assigned_feature[stage_key].append(assigned_feature[next_stage_key][s_ind])
                            self.n_samples[stage_key] += len(sampling_ind)
                            break
                
                agg_data_flattened = np.array(assigned_data[stage_key]).reshape(-1, self.n_dim)
                diff_data = np.array(assigned_diff_data[stage_key]).reshape(1, -1)
                feature_mat = np.array(assigned_feature[stage_key]).T
                (
                    optimized_mrf_models_new[stage_key],
                    _,
                    _,
                    _,
                ) = self.estimate_mrf(agg_data_flattened.T)
                wiener = wiener_module.wiener()
                wiener.projection_fit(feature_mat, diff_data)
                optimized_wiener_models_new[stage_key] = copy.deepcopy(wiener)
                    

            optimized_models["mrf"] = copy.deepcopy(optimized_mrf_models_new)
            optimized_models["wiener"] = copy.deepcopy(optimized_wiener_models_new)

        cost_table = pd.DataFrame(
            data=np.concatenate(cost_tables, axis=1).T,
            columns=["seq_id", "stage_id", "all_ll", "mrf_ll", "wiener_ll"],
        )

        return optimized_models, cost_table, overall_ll, objective_values

    def tc_scan(self, data, update=False):
        pred = []
        cluster_ids = []
        inc_stage_ids = []
        fin_stage_ids = []
        n_stage_hist = []

        mrf_models = self.opt_models["mrf"]
        wiener_models = self.opt_models["wiener"]
        tic = time.process_time()

        for data_id, (seq_tensor, label) in tqdm(
            enumerate(zip(data["features"], data["label"])),
            desc="Forecasting Event Progression",
        ):
            pred_seq, incremental_path, final_path = self.staging_and_forecasting(
                    seq_tensor, mrf_models, wiener_models, verbose=False
            )

            pred.append(pred_seq)
            cluster_ids.append(np.full(len(incremental_path), -1))
            inc_stage_ids.append(incremental_path)
            fin_stage_ids.append(final_path)
            
            if update:
                stage_pe_summary, pe_df = self.check_improvable_stage(pred_seq, label, incremental_path)
                add_mrf_models, add_wiener_models = self.add_alternate_optimization(seq_tensor, label, stage_pe_summary, pe_df, mrf_models, wiener_models,verbose=ALTERNATE_VERBOSE) 


                add_pred_seq, add_incremental_path, add_final_path = self.staging_and_forecasting(
                        seq_tensor, add_mrf_models, add_wiener_models, verbose=False
                )
                _, add_pe_df = self.check_improvable_stage(add_pred_seq, label, add_incremental_path)
                opt_acc = np.mean(pe_df["pe"])
                add_opt_acc = np.mean(add_pe_df["pe"])
                if opt_acc > add_opt_acc:
                    print(f"UPDATE MODELS K++, K={len(add_mrf_models)}")
                    mrf_models = copy.deepcopy(add_mrf_models)
                    wiener_models = copy.deepcopy(add_wiener_models)
                else:
                    print(f"MAINTAIN K={len(mrf_models)}")
            n_stage_hist.append(len(mrf_models))

        pred_time = time.process_time() - tic

        pred = np.concatenate(pred, axis=0)[:, np.newaxis]

        cluster_ids = np.concatenate(cluster_ids, axis=0)
        inc_stage_ids = np.concatenate(inc_stage_ids, axis=0)
        fin_stage_ids = np.concatenate(fin_stage_ids, axis=0)

        self.opt_models["mrf"] = copy.deepcopy(mrf_models)
        self.opt_models["wiener"] = copy.deepcopy(wiener_models)

        return pred, cluster_ids, inc_stage_ids, fin_stage_ids, pred_time, n_stage_hist
    
    def check_improvable_stage(self, pred_seq, label, incremental_path):
        erreps = 5
        df = pd.DataFrame({"pred": np.asarray(pred_seq).ravel(), "label": np.asarray(label).ravel(), "stage": np.asarray(incremental_path).ravel()})
        tdf = df.loc[df["label"] > erreps].copy()
        tdf.loc[:, "pe"] = (tdf["pred"] - tdf["label"]).abs() / tdf["label"]
        tdf_summary = tdf.groupby("stage").agg(pe=("pe", "mean")).reset_index()
        return tdf_summary, tdf


    def add_alternate_optimization(self,seq_tensor, label, stage_pe_summary, pe_df, mrf_models, wiener_models,verbose=False):
        improvable_stage_id = stage_pe_summary.loc[stage_pe_summary["pe"] == stage_pe_summary['pe'].max(),"stage"].values[0]
        add_n_stages = len(mrf_models) + 1
        t_pes = pe_df.loc[pe_df["stage"]==improvable_stage_id,"pe"].values
        t_index = pe_df.loc[pe_df["stage"]==improvable_stage_id,:].index.to_numpy()
        
        seq_tensor_partial = seq_tensor[t_index]
        label_partial  = label[t_index]
        
        if len(t_pes) < 10:
            print("insufficient data samples")
            return mrf_models, wiener_models           
        sep_index=int(len(t_pes)/2)
        add_mrf_models={}
        add_wiener_models={}

        n_samples_old = copy.deepcopy(self.n_samples)
        self.n_samples = np.zeros(add_n_stages)
        if np.mean(t_pes[:sep_index]) > np.mean(t_pes[sep_index:]):
            ini_data = seq_tensor_partial[:sep_index]
            ini_label = label_partial[:sep_index]
            bias=1
        
        else:
            ini_data = seq_tensor_partial[sep_index:]
            ini_label = label_partial[sep_index:]
            bias=0        

        print(f"add stage:{improvable_stage_id+bias} \n target/total {label_partial.shape}/{label.shape}")
        # shift models
        for s in range(add_n_stages):
            if s < improvable_stage_id:
                add_mrf_models[s] = copy.deepcopy(mrf_models[s])
                add_wiener_models[s] = copy.deepcopy(wiener_models[s])
                self.n_samples[s] = n_samples_old[s]
            elif s==improvable_stage_id:
                # insert initialized models as stage s
                mrf_ini_data = [data_f.reshape(-1, self.n_dim) for data_f in ini_data]
                mrf_ini_data = np.concatenate(mrf_ini_data, axis=0)
                add_mrf_models[improvable_stage_id] = self.estimate_mrf(mrf_ini_data.T)[0]
                add_wiener_models[improvable_stage_id] = self.build_wiener_model(
                    ini_data,
                    ini_label,
                    verbose=False,
                )
            elif s > improvable_stage_id:
                add_mrf_models[s] = copy.deepcopy(mrf_models[s-1])
                add_wiener_models[s] = copy.deepcopy(wiener_models[s-1])
                self.n_samples[s] = n_samples_old[s-1]

        base_mrf_models = copy.deepcopy(add_mrf_models)
        base_wiener_models = copy.deepcopy(add_wiener_models)

        objective_values=[]
        optimal_path = []
        opt_ll = -np.inf
        opt_mrf = -np.inf
        opt_wiener = -np.inf
        
        for it in range(self.max_iter):
            print(f"START UPDATE-ITER #{it}")
            (
                c_optimal_path,
                c_opt_ll_hist,
                c_opt_ll,
                c_opt_mrf,
                c_opt_wiener,
            ) = self.estimate_stages_tensor(
                seq_tensor_partial,
                label_partial,
                add_mrf_models,
                add_wiener_models,
                verbose=False,
            )
            
            assigned_data = {
                stage_: [] for stage_ in add_mrf_models
            }
            assigned_diff_data = {
                stage_: [] for stage_ in add_mrf_models
            }
            assigned_feature = {
                stage_: [] for stage_ in add_mrf_models
            }

            for stage_key in add_mrf_models:
                ns = np.sum(c_optimal_path == stage_key)
                if verbose:
                    print(f"k:{stage_key}, ns:{ns}") 
                if ns > 0:
                    assigned_data[stage_key].extend(
                        seq_tensor_partial[c_optimal_path == stage_key]
                    )
                    for da in seq_tensor_partial[c_optimal_path == stage_key]:
                        assigned_feature[stage_key].append(da.reshape(-1))
                    drift_array = self.get_drift_array(
                        label_partial, c_optimal_path, stage_key
                    )
                    assigned_diff_data[stage_key].extend(drift_array)

                    agg_data_flattened = np.array(assigned_data[stage_key]).reshape(-1, self.n_dim)
                    diff_data = np.array(assigned_diff_data[stage_key]).reshape(1, -1)
                    feature_mat = np.array(assigned_feature[stage_key]).T

                    if stage_key == improvable_stage_id:
                        (
                            add_mrf_models[stage_key],
                            _,
                            _,
                            _,
                        ) = self.estimate_mrf(agg_data_flattened.T)
                        wiener = wiener_module.wiener()
                        wiener.projection_fit(feature_mat, diff_data)
                        add_wiener_models[stage_key] = copy.deepcopy(wiener)
                    else:
                        add_mrf_models[stage_key] = self.online_update_mrf(base_mrf_models[stage_key], agg_data_flattened.T, self.n_samples[stage_key])
                        add_wiener_models[stage_key] = self.online_update_wiener(base_wiener_models[stage_key], feature_mat, diff_data, self.n_samples[stage_key])
            
            gap = c_opt_ll - opt_ll
            if abs(gap) < EPS:
                print("Converged (below threshold).")
                break

            if np.array_equal(optimal_path, c_optimal_path):
                print("Converged (paths unchanged).")
                break
            objective_values.append([c_opt_ll, c_opt_mrf, c_opt_wiener])
            optimal_path = copy.deepcopy(c_optimal_path)
            opt_ll = c_opt_ll
            opt_mrf = c_opt_mrf
            opt_wiener = c_opt_wiener

        if np.all(optimal_path == optimal_path[0]):
            print("Non-separable: returning original models.")
            return mrf_models, wiener_models

        return add_mrf_models, add_wiener_models
    
    def online_update_mrf(self, mrf_model, seq, past_ns):
        # seq (n_dim, duration)
        upd_mrf_model = copy.deepcopy(mrf_model)
        upd_mrf_model = upd_mrf_model.fit_additional_data(seq.T, past_ns)
        upd_mrf_model.mean_ = (past_ns * upd_mrf_model.mean_ + np.sum(seq, axis=1)) / (past_ns + seq.shape[1])
        return upd_mrf_model

    def online_update_wiener(self, wiener_model, feature_mat, diff_data, past_ns):
        # feature_mat: d x n
        upd_wiener_model = copy.deepcopy(wiener_model)
        upd_wiener_model.update_mean_std(feature_mat, diff_data, past_ns)

        return upd_wiener_model

    def get_drift_array(self, label, path, stage_id):
        drift_array = label[path == stage_id]
        return drift_array

    def staging_and_forecasting(
        self,
        data_features,
        mrf_models,
        wiener_models,
        verbose=False,
    ):
        n_stages = len(mrf_models)
        path_len = data_features.shape[0]

        incremental_opt_path = np.full(path_len, -1)
        pred_values = np.zeros(path_len)

        path_prob = np.zeros((1, n_stages))
        path_prob_hist = np.zeros((1, n_stages, path_len))
        optimal_path = np.full((n_stages, path_len), -1)

        for s in range(n_stages):
            obj_mrf = self.l_mrf * self.multivariate_normal_logll(mrf_models[s], data_features[0]) / self.ini_mrf_ll
            path_prob[0, s] = obj_mrf
            optimal_path[s, 0] = s
            path_prob_hist[:, s, 0] = obj_mrf

        # stageing
        opt_cur_stage = np.argmax(path_prob[0, :])
        incremental_opt_path[0] = opt_cur_stage

        # forecasting
        feature = data_features[0].reshape(-1, 1)

        mean, _ = wiener_models[opt_cur_stage].estimate_rul_val(feature, cur_state=0)
        pred_values[0] = mean

        for t in range(1, path_len):
            if verbose:
                print(f"time:{t}")

            path_prob_cur = np.zeros(
                (1, n_stages)
            )  # likelihoods for the stage s at the time t

            opt_path_cur = np.zeros(
                (n_stages, t + 1)
            )  # the optimal path to the current stage s by the time t

            for s in range(n_stages):
                # mi = 0 if s - 1 < 0 else s - 1 # only one hop
                mi = 0  # all stages whose upper id
                ma = s + 1 
                # ma = n_stages #assign all stages


                opt_stage = (
                    np.argmax(path_prob[0, mi:ma]) + mi
                )  # an optimal past stage for stage s

                opt_path_cur[s, :t] = optimal_path[opt_stage, :t]
                opt_path_cur[s, t] = s
                ob_mrf = self.l_mrf * self.multivariate_normal_logll(mrf_models[s],data_features[t]) /self.ini_mrf_ll

                path_prob_cur[0, s] = ob_mrf + path_prob[0, opt_stage]
                path_prob_hist[0, s, t] = ob_mrf

                if verbose:
                    print(f"{s}: \nall:{ob_mrf}")

            path_prob = path_prob_cur
            optimal_path[:, : t + 1] = opt_path_cur

            if verbose:
                print(f"optimam: {path_prob}")
                print(optimal_path)

            # stageing
            opt_cur_stage = np.argmax(path_prob[0, incremental_opt_path[t-1]:])
            opt_cur_stage += incremental_opt_path[t - 1]
            incremental_opt_path[t] = opt_cur_stage

            # forecasting
            feature = data_features[t].reshape(-1, 1)

            mean, _ = wiener_models[opt_cur_stage].estimate_rul_val(
                feature, cur_state=0,
            )
            pred_values[t] = mean

        if verbose:
            print("result")
            print(path_prob)
            print(optimal_path)
            print(incremental_opt_path)

        opt = np.argmax(path_prob[0, :])
        opt_val_cum = path_prob[0, opt]
        opt_val_hist = path_prob_hist[:, opt, :]
        final_optimal_path = optimal_path[opt]

        return pred_values, incremental_opt_path, final_optimal_path


    def multivariate_normal_logll(self, mrf_model, data_f):
        """Log-likelihood under multivariate normal (data_f: duration x n_dim)."""
        data_len = data_f.shape[0]
        data_x = data_f - mrf_model.mean_
        ll = (
            -(data_x @ mrf_model.precision_[0]).flatten() @ data_x.flatten()
            - data_len * mrf_model.log_pdet
            - self.n_dim * data_len * np.log(2 * np.pi)
        )
        return 0.5 * ll



def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps