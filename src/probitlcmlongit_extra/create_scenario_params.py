# This file is part of "probitlcmlongit_extra" which is released under GPL v3.
#
# Copyright (c) 2022-2025 Eric Alan Wayman <ericwaymanpublications@mailworks.org>.
#
# This program is FLO (free/libre/open) software: you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse, json, pathlib, shutil,  tomllib

import numpy as np
from probitlcmlongit import _core

# for perform_gamma_optimization
from scipy.stats import norm, multinomial
import pandas as pd
from scipy.optimize import minimize

# for create_beta_and_delta
from probitlcmlongit import run_helpers
import itertools

# for create_situation_params
from scipy.stats import multivariate_normal
from scipy.stats import hmean

class PosRelated():
    def __init__(self):
        self.pos_to_remove_and_effects_tables = dict()
        self.perm_and_inverse_perm_tables = dict()
        self.M_j_s = np.empty((0))
        self.alphas_table = None
    
    def __str__(self):
        my_str = "self.pos_to_remove_and_effects_tables: " + \
                  str(self.pos_to_remove_and_effects_tables) + "\n" + \
                  "self.perm_and_inverse_perm_tables: " + \
                  str(self.perm_and_inverse_perm_tables)
        return my_str

class DatagenQuants():
    def __init__(self):
        self.Rmat = None
        self.gammas_list = list()
        self.beta = None
        self.delta = None
        self.my_lambda = None
        self.kappas_list = list()
        self.thetas_list = list()

def calc_theta_jmc(m, design_vec, beta_j, kappa_j):
    value = norm.cdf(kappa_j[m] - np.dot(design_vec, beta_j)) - \
        norm.cdf(kappa_j[m - 1] - np.dot(design_vec, beta_j)) 
    return value

def calculate_theta_j(H_K, M_j, beta_j, kappa_j, design_matrix):
    theta_j = np.empty((H_K, M_j))
    for c in range(1, H_K + 1):
        for m in range(1, M_j + 1):
            theta_j[c - 1, m - 1] = calc_theta_jmc(
                m, design_matrix[c - 1, :], beta_j, kappa_j)
    return theta_j

def calculate_thetas(J, H_K, M_j_s, beta, kappas_list, design_matrix):
    # calculate thetas
    thetas_list = list()
    for j in range(1, J + 1):
        M_j = M_j_s[j - 1]
        theta_j = np.empty((H_K, M_j))
        beta_j = beta[:, j - 1]
        kappa_j = kappas_list[j - 1]
        for c in range(1, H_K + 1):
            for m in range(1, M_j + 1):
                theta_j[c - 1, m - 1] = calc_theta_jmc(
                    m, design_matrix[c - 1, :], beta_j, kappa_j)
        thetas_list.append(theta_j)
    return thetas_list

# begin theta-related

def transform_eta_to_kappa(eta):
    M_j = len(eta) - 1
    vector_pre = eta[1:M_j]
    vector_pre_exp = np.empty(vector_pre.shape)
    vector_pre_exp[0] = eta[1]
    vector_pre_exp[1:] = np.exp(vector_pre[1:]).cumsum()
    kappa = np.concatenate(([-np.inf], vector_pre_exp, [np.inf]))
    return kappa

def transform_kappa_to_eta(kappa):
    M_j = len(kappa) - 1
    kappa_subset = kappa[1:M_j]
    eta_pre = np.empty(kappa_subset.shape)
    eta_pre[0] = kappa[1]
    eta_pre[1:] = np.log(np.diff(kappa_subset))
    eta = np.concatenate(([-np.inf], eta_pre, [np.inf]))
    return eta

def perform_eta_optimization(J, H_K, M_j_s, beta, design_matrix):
    # find classes to eliminate, for each j
    classes_to_keep = list()
    n_classes = design_matrix.shape[0]
    dMat_beta = np.matmul(design_matrix, beta)
    for j in range(1, J + 1):
        keep_j = list()
        partition_j = partition(dMat_beta[:, j - 1])
        for key, value in partition_j.items():
            keep_j.append(value[-1])
        classes_to_keep.append(keep_j)
    # calculate kappa_j for each j
    kappas_list = list()
    for j in range(1, J + 1):
        M_j = M_j_s[j - 1]
        # initialize kappas
        d_1_beta = np.dot(design_matrix[0, :], beta[:, j - 1])
        eta_j_possibs = list()
        # calculate kappa_j_possibs_1
        quantiles = np.linspace(0, 1, M_j + 1)
        quantiles = quantiles[1:M_j + 1 - 1]
        kappa_j = np.empty((int(M_j + 1)))
        kappa_j[0] = -np.inf
        for idx, m in enumerate(range(1, M_j + 1 - 1)):
            kappa_j[m] = norm.ppf(quantiles[idx]) + d_1_beta
        kappa_j[M_j] = np.inf
        eta_j = transform_kappa_to_eta(kappa_j)
        eta_j_possibs.append(eta_j)
        # calculate kappa_set_2
        kappa_j = np.empty((int(M_j + 1)))
        kappa_j[0] = -np.inf
        kappa_j[1] = 0.0
        kappa_j[M_j] = np.inf
        if M_j > 2:
            for idx in range(2, M_j):
                kappa_j[idx] = kappa_j[idx - 1] + 2.0
        eta_j = transform_kappa_to_eta(kappa_j)
        eta_j_possibs.append(eta_j)
        # try each eta_j possib and select the one with the lowest minimum
        our_results = list()
        for eta_j in eta_j_possibs:
            s_eta_j = eta_j[1:M_j]
            neg_metric = calc_neg_metric_pairwise_class_diff_thetas(
                             s_eta_j, J, H_K, M_j_s[j - 1], beta[:, j - 1],
                             design_matrix, classes_to_keep[j - 1])
            our_results.append(neg_metric)
        idx_of_best_eta_j = np.argmin(our_results)
        # get the best of the eta_j possibs, and perform the minimization using
        #     it as our starting value
        eta_j = eta_j_possibs[idx_of_best_eta_j]
        s_eta_j = eta_j[1:M_j]
        res = minimize(calc_neg_metric_pairwise_class_diff_thetas,
                       s_eta_j,
                       args=(J, H_K, M_j_s[j - 1], beta[:, j - 1],
                             design_matrix, classes_to_keep[j - 1]),
                       method='Powell',
                       options={'maxiter' : 10000, 'xatol': 1e-8, 'disp': True})
        eta_j = build_gammas_from_s_gammas(res.x)
        kappa_j = transform_eta_to_kappa(eta_j)
        kappas_list.append(kappa_j)
    return kappas_list

def calc_penalty_theta_term(min_max_theta_j, M_j):
    penalty_term = 0
    cutoff = 1 / (4 * M_j)
    # first make sure min_max_theta_j is not exactly zero
    if min_max_theta_j < (cutoff / 1000):
        min_max_theta_j = cutoff / 1000
    # now calculate penalty term
    if min_max_theta_j > cutoff:
        penalty_term = 0
    else:
        penalty_term = (1 / min_max_theta_j) - (1 / cutoff)
    return penalty_term

def partition(array):
    return {i: (array == i).nonzero()[0] for i in np.unique(array)}

def calc_neg_metric_pairwise_class_diff_thetas(s_eta_j,
                                               J, H_K, M_j, beta_j,
                                               design_matrix,
                                               classes_to_keep_j):
    eta_j = build_gammas_from_s_gammas(s_eta_j)
    kappa_j = transform_eta_to_kappa(eta_j)
    theta_j = calculate_theta_j(H_K, M_j, beta_j, kappa_j, design_matrix)
    subtheta_j = theta_j[classes_to_keep_j]
    nrows = len(subtheta_j)
    curr_min_dist = np.inf
    vec_of_dists_list = list()
    # calculate min dist
    for comb in itertools.combinations(range(nrows), 2):
        curr_dist = subtheta_j[comb[0], :] - subtheta_j[comb[1], :]
        mynorm = np.linalg.norm(curr_dist, 1)
        vec_of_dists_list.append(mynorm)
        if mynorm < curr_min_dist:
            curr_min_dist = mynorm
    vec_of_dists = np.array(vec_of_dists_list)
    # calculate penalty term based on min_{m}(max_{c} (theta_{jmc}))
    max_c_theta_jmc = np.max(theta_j, axis=0)
    min_max_theta_j = np.min(max_c_theta_jmc)
    penalty_theta_term = calc_penalty_theta_term(min_max_theta_j, M_j)
    # calculate overall metric
    neg_metric = -1 * (4 * curr_min_dist + 2 * hmean(vec_of_dists) \
       - penalty_theta_term) # penalty_theta_term (older:minsum) is opposite
                             # direction
    return neg_metric

def find_best_kappas(data, pos_to_remove, M_j_s, beta, design_matrix):
    # load preliminary info
    ## for theta calcs
    J = situation_dict["J"]
    L_k_s = situation_dict["L_k_s"]
    H_K = np.prod(L_k_s)
    K = situation_dict["K"]
    # do calcs
    kappas_list = perform_eta_optimization(J, H_K, M_j_s, beta, design_matrix)
    # shift kappa_1j's to zero and perform corresponding
    #     transformation on beta_j1's
    for j in range(1, J + 1):
        M_j = M_j_s[j - 1]
        kappa_1j = kappas_list[j - 1][1]
        beta[0, j - 1] = beta[0, j - 1] - kappa_1j
        kappas_list[j - 1][1] = 0.0
        for idx, kappa_ji in enumerate(kappas_list[j - 1]):
            if idx > 1 and idx < M_j:
                kappas_list[j - 1][idx] = \
                    kappas_list[j - 1][idx] - kappa_1j
    return (kappas_list, beta)

# end theta related

# begin gamma related
        
def perform_gamma_optimization(alpha_star_t, N, K, L_k_s):
    s_gammas_list = list()
    for k in range(1, K + 1):
        L_k = L_k_s[k - 1]
        quantiles = np.linspace(0, 1, num = L_k + 1)
        gammas = norm.ppf(quantiles)
        s_gammas = gammas[1:L_k]
        s_gammas_list.append(s_gammas)
    s_gammas_flattened = np.concatenate(s_gammas_list)
    res = minimize(calc_min_class_membership, s_gammas_flattened,
                   args=(alpha_star_t, N, K, L_k_s),
                   method='Powell',
                   options={'maxiter' : 10000, 'xatol': 1e-8, 'disp': True})
    calc_min_class_membership(res.x,
                              alpha_star_t, N, K, L_k_s)
    gammas_list = build_gammas_list(res.x, K, L_k_s)
    alpha_sampled = find_alpha_from_alphastar(
        alpha_star_t, N, K, gammas_list)
    return((gammas_list, alpha_sampled))
    # save these alphas

def find_alpha_from_alphastar(alphastar_sampled, num_samples, K,
                              gammas_list):
    alpha_sampled = np.empty((num_samples, K))
    for k in range(1, K + 1):
        for n in range(1, num_samples + 1):
            alpha_sampled[n - 1, k - 1] = sum(
                gammas_list[k - 1] <= alphastar_sampled[n- 1, k - 1]) - 1
    return alpha_sampled

def build_gammas_from_s_gammas(s_gammas):
    gammas = np.empty((len(s_gammas) + 2))
    gammas[0] = -np.inf
    gammas[1:len(gammas) - 1] = s_gammas
    gammas[len(gammas) - 1] = np.inf
    return gammas

def build_gammas_list(s_params_flattened, K, L_k_s):
    gammas_list = list()
    idx = 0
    for k in range(1, K + 1):
        L_k = L_k_s[k - 1]
        s_gammas = s_params_flattened[idx:idx+L_k + 1 - 2]
        idx += L_k + 1 - 2
        gammas_list.append(
            build_gammas_from_s_gammas(s_gammas))
    return gammas_list

def find_class_memberships(s_params_flattened,
                           alpha_star, N, K, L_k_s):
    gammas_list = build_gammas_list(s_params_flattened, K, L_k_s)
    alpha_sampled = find_alpha_from_alphastar(
        alpha_star, N, K, gammas_list)
    df = pd.DataFrame(alpha_sampled)
    my_obj = df.groupby(list(range(K))).size()
    return my_obj


def calc_min_class_membership(s_params_flattened,
                              alpha_star_t, N, K, L_k_s):
    return -np.min(find_class_memberships(s_params_flattened,
                                          alpha_star_t, N, K, L_k_s))

# end gamma related

## stuff for beta and delta

def find_one_way_and_two_way_effects(L_k_s, K):
    # ref run_helpers.find_post_to_keep
    # initial
    seq_K = list(range(1, K + 1))
    my_dict = dict()
    for k in range(1, K + 1):
        my_dict[k] = list(range(L_k_s[k - 1]))
    basis_vector = run_helpers.calculate_basis_vector(L_k_s, K)
    # next part
    main_effects = list()
    full_levels_vec_set = set()
    for comb in itertools.combinations(seq_K, 1):
        building_blocks = list()
        for j in seq_K:
            if j in comb:
                building_blocks.append(my_dict[j])
            else:
                building_blocks.append([0])
        prod_of_blocks = itertools.product(*building_blocks)
        minilist = list()
        for x in prod_of_blocks:
            minilist.append(
                run_helpers.convert_alpha_to_class_number(x, basis_vector))
        main_effects.append(minilist)
    for x in main_effects:
        x.remove(0)
    main_effects_set = set(itertools.chain.from_iterable(main_effects))
    # do two way effects
    two_way_effects = list()
    two_way_combs_list = list()
    if K == 2:
        comb = list(itertools.combinations(seq_K, 2))[0]
        two_way_combs_list.append(comb)
        building_blocks = list()
        for j in seq_K:
            if j in comb:
                building_blocks.append(my_dict[j])
            else:
                building_blocks.append([0])
        prod_of_blocks = itertools.product(*building_blocks)
        minilist = list()
        for x in prod_of_blocks:
            minilist.append(
                run_helpers.convert_alpha_to_class_number(x,
                                                          basis_vector))
        two_way_effects.append(minilist)
    else:
        for comb in itertools.combinations(seq_K, 2):
            if comb[0] < K - 1: # everything until end of for loop
                two_way_combs_list.append(comb)
                building_blocks = list()
                for j in seq_K:
                    if j in comb:
                        building_blocks.append(my_dict[j])
                    else:
                        building_blocks.append([0])
                prod_of_blocks = itertools.product(*building_blocks)
                minilist = list()
                for x in prod_of_blocks:
                    minilist.append(
                        run_helpers.convert_alpha_to_class_number(x,
                                                                  basis_vector))
                two_way_effects.append(minilist)
    for x in two_way_effects:
        x.remove(0)
    return (main_effects, two_way_effects, two_way_combs_list)

def create_beta_and_delta(L_k_s, K, q):
    main_effects, two_way_effects, two_way_combs_list = \
        find_one_way_and_two_way_effects(L_k_s, K)
    J = (len(main_effects) + len(two_way_effects)) * q
    H_K = np.prod(L_k_s)
    # build delta and beta
    delta = np.zeros((H_K, J), dtype=np.uint)
    beta = np.zeros((H_K, J))
    main_effect_value = 2.0
    two_way_effect_value = 1.0
    q = 5
    z = 0
    start_range = z * q
    end_range = (z + 1) * q
    for x in main_effects:
        for i in range(start_range, end_range):
            delta[0, i] = 1
            beta[0, i] = -1.0
            for w in x:
                delta[w, i] = 1.0
                beta[w, i] = main_effect_value
        z += 1
        start_range = z * q
        end_range = (z + 1) * q
    for idx, x in enumerate(two_way_effects):
        for i in range(start_range, end_range):
            delta[0, i] = 1
            beta[0, i] = -1.0
            for w in x:
                delta[w, i] = 1.0
                beta[w, i] = two_way_effect_value
            main_effect_pos_list = two_way_combs_list[idx]
            for pos in main_effect_pos_list:
                positions = main_effects[pos - 1]
                for w in positions:
                    delta[w, i] = 1.0
                    beta[w, i] = main_effect_value
        z += 1
        start_range = z * q
        end_range = (z + 1) * q
    return(beta, delta, J)
    

# main function

def find_best_gammas(num_samples, alpha_star_t_mean, Rmat, rng,
                     K, L_k_s, my_lambda):
    alpha_star_t = np.empty((num_samples, K))
    for n in range(1, num_samples + 1):
        alpha_star_t[n - 1, :] = multivariate_normal.rvs(
            alpha_star_t_mean[n - 1], Rmat, random_state=rng)
        # do gammas
    gammas_list, alpha_sampled = perform_gamma_optimization(
        alpha_star_t, num_samples, K, L_k_s)
    # shift gamma_1k's to zero and perform corresponding
    #     transformation on lambda_k1's
    for k in range(1, K + 1):
        L_k = L_k_s[k - 1]
        gamma_1k = gammas_list[k - 1][1]
        my_lambda[0, k - 1] = -1.0 * gamma_1k
        gammas_list[k - 1][1] = 0.0
        for idx, gamma in enumerate(gammas_list[k - 1]):
            if idx > 1 and idx < L_k:
                gammas_list[k - 1][idx] = \
                    gammas_list[k - 1][idx] - gamma_1k
    return (gammas_list, my_lambda)


def save_arma_mat(my_mat, var_name, situation_path):
    fname = "datagen_" + var_name + ".txt"
    fpath = situation_path.joinpath(fname)
    _core.save_arma_mat_np(my_mat, str(fpath), "arma_ascii")

def save_arma_mat_given_fname(my_mat, situation_path, fname):
    fpath = situation_path.joinpath(fname)
    _core.save_arma_mat_np(my_mat, str(fpath), "arma_ascii")

def save_arma_umat_given_fname(my_umat, situation_path, fname):
    fpath = situation_path.joinpath(fname)
    _core.save_arma_umat_np(my_umat, str(fpath), "arma_ascii")

def find_pos_related_quantities(situation_dict, fixed_other_vals_dict):    
    pos_related_struct = PosRelated()
    pos_related_struct.alphas_table = run_helpers.create_alphas_table(
        situation_dict)
    L_k_s = situation_dict["L_k_s"]
    mydict = dict() # a dict consisting of keys of max levels and values
                    #     that are lists of dimensions with that max level
    for k, x in enumerate(L_k_s):
        if x not in mydict:
            mydict[x] = list()
        mydict[x].append(k)
    pos_related_struct.pos_to_remove_and_effects_tables = \
        run_helpers.find_pos_to_remove_and_effects_tables(
            situation_dict, fixed_other_vals_dict,
            mydict, pos_related_struct.alphas_table)
    pos_related_struct.perm_and_inverse_perm_tables = \
        run_helpers.create_perm_and_inverse_perm_tables(
            mydict, pos_related_struct.alphas_table, situation_dict,
            fixed_other_vals_dict)
    pos_related_struct.M_j_s = \
        run_helpers.calculate_M_j_s(situation_dict)
    return pos_related_struct

# Note that this function saves the situation's json file as well.
def save_pos_related_quantities(pos_related_struct,
                                run_dir, sim_info_dir_name,
                                situation_dict, situations_list,
                                fixed_other_vals_dict):
    results_path = run_dir.joinpath(sim_info_dir_name, "scenario_files")
    fpath = run_dir.joinpath(sim_info_dir_name,
                             "json_files",
                             "01_list_N_vals.json")
    list_N_vals = json.loads(fpath.read_text())
    situation_num = situation_dict["situation_num"]
    mod_num = len(situations_list)
    total_of_n = len(list_N_vals)
    for n in range(total_of_n):
        scenario_dict = situation_dict.copy()
        scenario_dict["N"] = list_N_vals[n]
        scenario_num = situation_num + mod_num * n
        scenario_dict["scenario_num"] = scenario_num
        scenario_num_dirname = f"scenario_{scenario_num:04}"
        scenario_path = results_path.joinpath(scenario_num_dirname)
        # create directory for storage
        scenario_path.mkdir(parents=True, exist_ok=True)
        # save the json file
        fname = scenario_num_dirname + ".json"
        json_file_path = scenario_path.joinpath(fname)
        with json_file_path.open(mode='w') as f:
            json.dump(scenario_dict, f, indent=4)
        # do the rest
        save_arma_umat_given_fname(pos_related_struct.alphas_table,
                                   scenario_path, "alphas_table.txt")
        save_arma_umat_given_fname(
            pos_related_struct.M_j_s,
            scenario_path,
            "M_j_s.txt")
        print(pos_related_struct.pos_to_remove_and_effects_tables)
        if "pos_to_remove" in \
            pos_related_struct.pos_to_remove_and_effects_tables:
                save_arma_umat_given_fname(
                    pos_related_struct.pos_to_remove_and_effects_tables[
                        "pos_to_remove"],
                    scenario_path,
                    "pos_to_remove.txt")
        save_arma_umat_given_fname(
            pos_related_struct.pos_to_remove_and_effects_tables[
                "effects_table"],
            scenario_path,
            "effects_table.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_perms_of_dims"],
            scenario_path,
            "table_perms_of_dims.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_inverse_perms_of_dims"],
            scenario_path,
            "table_inverse_perms_of_dims.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_perms_of_class_nums"],
            scenario_path,
            "table_perms_of_class_nums.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_inverse_perms_of_class_nums"],
            scenario_path,
            "table_inverse_perms_of_class_nums.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_perms_of_pos_nums"],
            scenario_path,
            "table_perms_of_pos_nums.txt")
        save_arma_umat_given_fname(
            pos_related_struct.perm_and_inverse_perm_tables[
                "table_inverse_perms_of_pos_nums"],
            scenario_path,
            "table_inverse_perms_of_pos_nums.txt")

def create_datagen_quants(situations_list, fixed_other_vals_dict,
                          situation_dict, run_dir, sim_info_dir_name,
                          situation_number_padded, seed_value_situation,
                          pos_related_struct):
    datagen_quants = DatagenQuants()
    D = 0
    if fixed_other_vals_dict["covariates"] == "age_assignedsex":
        D = 3
    K = situation_dict["K"]
    T = fixed_other_vals_dict["T"]
    fpath = run_dir.joinpath(sim_info_dir_name,
                             "json_files",
                             "01_list_N_vals.json")
    list_N_vals = json.loads(fpath.read_text())
    L_k_s = situation_dict["L_k_s"]
    H_K = np.prod(situation_dict["L_k_s"])
    situation_params = dict()
    situation_params["omega"] = 0.5
    # lambdas have already been generated by user separately
    # generate Rmat
    rho = situation_dict["rho"]
    datagen_quants.Rmat = rho * np.ones((K, K), dtype=np.uint) + \
        (1 - rho) * np.eye(K, K, k=0, dtype=np.uint)
    # lambda
    my_lambda = np.empty((D - 1, K))
    intercept_slope = np.zeros((1, K))
    sim_info_src_dir_path = run_dir.joinpath(sim_info_dir_name,
                                             "lambda_files")
    lambdafilename = "datagen_lambda_" + "situation_" + \
        situation_number_padded + ".txt";
    lambdafile_src_path = sim_info_src_dir_path.joinpath(lambdafilename)
    my_lambda = _core.load_arma_mat_np(str(lambdafile_src_path))
    print(my_lambda)
    # add intercept of 0 to lambda
    my_lambda = np.vstack((intercept_slope, my_lambda))
    # load covariates for estimation of gammas
    num_samples = 10000
    Xmat = _core.generate_covariates_scenario_np(
        1, num_samples, seed_value_situation)
    # add intercept to Xmat
    intercept_Xmat = np.ones((1 * num_samples, 1))
    Xmat = np.hstack((intercept_Xmat, Xmat))
    # calc alpha_star and gammas
    rng = np.random.default_rng(seed_value_situation)
    alpha_star_t_mean = np.matmul(Xmat, my_lambda)
    datagen_quants.gammas_list, datagen_quants.my_lambda = \
        find_best_gammas(num_samples,
                         alpha_star_t_mean, datagen_quants.Rmat,
                         rng,
                         K, L_k_s,
                         my_lambda)
    # continue
    q = 5
    beta, delta, J = create_beta_and_delta(
        L_k_s, K, q)
    pos_to_remove = list()
    if "pos_to_remove" in pos_related_struct.pos_to_remove_and_effects_tables:
         pos_to_remove = pos_related_struct.pos_to_remove_and_effects_tables[
             "pos_to_remove"]
         pos_to_remove = pos_to_remove.ravel()
         print("pos_to_remove", pos_to_remove)
    beta = np.delete(beta, pos_to_remove, axis=0)
    datagen_quants.delta = np.delete(delta, pos_to_remove, axis=0)    
    M_j_s = pos_related_struct.M_j_s.ravel()
    M_j_s = M_j_s.astype(int)
    design_matrix = _core.generate_design_matrix_np(
        pos_related_struct.alphas_table, K, L_k_s,
        fixed_other_vals_dict["order"],
        pos_to_remove)
    datagen_quants.kappas_list, datagen_quants.beta = \
        find_best_kappas(situation_dict, pos_to_remove,
                         M_j_s, beta,
                         design_matrix)
    datagen_quants.thetas_list = calculate_thetas(J, H_K, M_j_s,
                                                  datagen_quants.beta,
                                                  datagen_quants.kappas_list,
                                                  design_matrix)    
    ## save off stuff
    print("about to save datagen quants")
    results_path = run_dir.joinpath(sim_info_dir_name, "scenario_files")
    situation_num = situation_dict["situation_num"]
    mod_num = len(situations_list)
    total_of_n = len(list_N_vals)
    for n in range(total_of_n):
        scenario_dict = situation_dict.copy()
        scenario_dict["N"] = list_N_vals[n]
        scenario_num = situation_num + mod_num * n
        scenario_dict["scenario_num"] = scenario_num
        scenario_num_dirname = f"scenario_{scenario_num:04}"
        scenario_path = results_path.joinpath(scenario_num_dirname)
        datagen_params_path = scenario_path.joinpath(
            "datagen_params")
        # create directory for storage
        datagen_params_path.mkdir(parents=False, exist_ok=True)
        # save the json file
        fname = scenario_num_dirname + ".json"
        json_file_path = scenario_path.joinpath(fname)
        with json_file_path.open(mode='w') as f:
            json.dump(scenario_dict, f, indent=4)
        # do the rest
        save_arma_mat(datagen_quants.Rmat, "Rmat", datagen_params_path)
        save_arma_mat(datagen_quants.my_lambda, "lambda",
                      datagen_params_path)
        for idx, gamma_k_list in enumerate(datagen_quants.gammas_list):
            k = idx + 1
            gamma_k = np.reshape(gamma_k_list, (1, len(gamma_k_list)))
            fname = 'datagen_gamma_' + str(k) + '.txt'
            fpath = datagen_params_path.joinpath(fname)
            _core.save_arma_mat_np(gamma_k, str(fpath), "arma_ascii")
        save_arma_mat(datagen_quants.beta, "beta", datagen_params_path)
        fpath = datagen_params_path.joinpath("datagen_delta.txt")
        _core.save_arma_umat_np(datagen_quants.delta, str(fpath),
                                "arma_ascii")
        for idx, kappa_j_list in enumerate(datagen_quants.kappas_list):
            j = idx + 1
            kappa_j = np.reshape(kappa_j_list, (1, len(kappa_j_list)))
            fname = f"datagen_kappa_j_{j:03}.txt"
            save_arma_mat_given_fname(kappa_j, datagen_params_path, fname)
        # save thetas
        for idx, theta_j in enumerate(datagen_quants.thetas_list):
            j = idx + 1
            fname = f"datagen_theta_j_mat_{j:03}.txt"
            save_arma_mat_given_fname(theta_j, datagen_params_path, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_info_dir_name', required=True)
    parser.add_argument('--situation_num', required=True)
    # parse arguments
    args = parser.parse_args()
    sim_info_dir_name = args.sim_info_dir_name
    situation_num = int(args.situation_num)
    print("situation_num", situation_num)
    run_dir = pathlib.Path.cwd()
    # load fixed_other_vals
    fpath = run_dir.joinpath(sim_info_dir_name,
                             "json_files",
                             "01_fixed_vals.json")
    fixed_other_vals_dict = json.loads(fpath.read_text())
    # find situations_dict
    situations_list_path = run_dir.joinpath(sim_info_dir_name,
                                           "json_files",
                                           "02_list_situations.json")
    with open(situations_list_path, 'r') as user_file:
        situations_list = json.load(user_file)
    situation_dict = None
    for d in situations_list:
        if d["situation_num"] == situation_num:
            situation_dict = d
    # set up path
    situation_num_padded = f"{situation_num:04}"
    # run other functions before running create_situation_params function
    # set seed
    config_file_path = run_dir.joinpath("config_simulation.toml")
    with open(config_file_path, "rb") as fileObj:
        config = tomllib.load(fileObj)
    process_dir = config['laptop_process_dir']
    number_of_replics = config['number_of_replics']
    situation_num_zb = situation_num - 1
    seed_value_situation = number_of_replics * situation_num_zb
    pos_related_struct = find_pos_related_quantities(situation_dict,
                                                     fixed_other_vals_dict)
    # continue to generate parameters
    save_pos_related_quantities(pos_related_struct,
                                run_dir, sim_info_dir_name,
                                situation_dict, situations_list,
                                fixed_other_vals_dict)
    create_datagen_quants(situations_list, fixed_other_vals_dict,
                          situation_dict, run_dir, sim_info_dir_name,
                          situation_num_padded, # needed to get the
                                                #   datagen lambda
                          seed_value_situation,
                          pos_related_struct)
