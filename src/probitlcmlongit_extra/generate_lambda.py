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

from scipy.stats import uniform, bernoulli
import numpy as np
import argparse
import pathlib
import json

from probitlcmlongit import _core

def generate_lambda(sim_info_dir_name, situations_list, situation_dict):
    current_path = pathlib.Path.cwd()    
    fpath = current_path.joinpath(sim_info_dir_name,
                                  "json_files",
                                  "01_fixed_vals.json")
    m = json.loads(fpath.read_text())
    D = None
    if m["covariates"] == "age_assignedsex":
        D = 3
    K = situation_dict["K"]
    L_k_s = situation_dict["L_k_s"]
    H_K = np.prod(L_k_s)
    # set up initial lambdas
    lambdas = np.empty((D - 1, K))
    samp = uniform.rvs(0, 1 / np.sqrt(0.24), size=K)
    lambdas[0, :] = np.sqrt((1 - (0.24 * samp)) / 175.17)
    lambdas[1, :] = np.sqrt(samp)
    # create and apply mask
    mask = bernoulli.rvs(0.5, size=(D - 1) * K).reshape((D - 1, K))
    mask[mask == 0] = -1
    lambdas = mask * lambdas
    # show resulting covariance matrix
    var_xi = np.array([[175.1714, 0], [0, 0.24]])
    part1 = np.matmul(np.transpose(lambdas), var_xi)
    myresults = np.matmul(part1, lambdas)
    print(myresults)
    # prepare paths for lambda files
    results_dir = run_dir.joinpath(sim_info_dir_name, "lambda_files")
    results_dir.mkdir(parents=True, exist_ok=True)
    # save it
    situation_num = situation_dict["situation_num"]
    fstring = f"datagen_lambda_situation_{situation_num:04}.txt"
    fpath = results_dir.joinpath(fstring)
    _core.save_arma_mat_np(lambdas, str(fpath), "arma_ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_info_dir_name", required=True)    
    parser.add_argument("--situation_num", required=True)    
    args = parser.parse_args()
    sim_info_dir_name = args.sim_info_dir_name
    situation_num = int(args.situation_num)
    run_dir = pathlib.Path.cwd()
    situations_list_path = run_dir.joinpath(sim_info_dir_name,
                                           "json_files",
                                           "02_list_situations.json")
    with open(situations_list_path, 'r') as user_file:
        situations_list = json.load(user_file)
    desired_dict = None
    for d in situations_list:
        if d["situation_num"] == situation_num:
            desired_dict = d
    generate_lambda(sim_info_dir_name, situations_list, desired_dict)

