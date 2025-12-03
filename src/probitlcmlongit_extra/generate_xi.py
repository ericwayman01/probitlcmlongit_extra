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

import argparse, json, pathlib, tomllib

import numpy as np
from scipy.stats import truncnorm

from probitlcmlongit import _core

def save_arma_mat(my_mat, var_name, my_path):
    fname = "datagen_" + var_name + ".txt"
    fpath = my_path.joinpath(fname)
    _core.save_arma_mat_np(my_mat, str(fpath), "arma_ascii")

def generate_xi(sim_info_dir_name, situations_list,
                situation_dict, seed_value_situation):
    rng = np.random.default_rng(seed_value_situation)
    K = situation_dict["K"]
    L_k_s = situation_dict["L_k_s"]
    H_K = np.prod(L_k_s)
    results_path = run_dir.joinpath(sim_info_dir_name, "scenario_files")
    fpath = run_dir.joinpath(sim_info_dir_name,
                             "json_files",
                             "01_list_N_vals.json")
    list_N_vals = json.loads(fpath.read_text())
    situation_num = situation_dict["situation_num"]
    mod_num = len(situations_list)
    total_of_n = len(list_N_vals)
    # use the first scenario to generate everything, then save it to all
    scenario_num = situation_num
    scenario_num_dirname = f"scenario_{scenario_num:04}"
    scenario_path = results_path.joinpath(scenario_num_dirname)
    # continue
    fpath = scenario_path.joinpath("effects_table_trans.txt")
    # resulting xi
    effects_table_trans = _core.load_arma_umat_np(str(fpath))
    xi_content = np.zeros(effects_table_trans.shape)
    my_mask = np.ma.masked_where(effects_table_trans == 0, xi_content)
    n_xi = my_mask.count()
    a_trunc = 0.0
    b_trunc = 0.10
    loc = 0.05
    scale = 1.0
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    xi_values = truncnorm.rvs(a = a, b = b, loc = loc, scale = scale,
                              size=n_xi, random_state=rng)
    my_counter = 0
    for i in range(xi_content.shape[0]):
        for j in range(xi_content.shape[1]):
            if my_mask[i, j] == False:
                xi_content[i, j] = xi_values[my_counter]
                my_counter += 1
    for n in range(total_of_n):
        scenario_dict = situation_dict.copy()
        scenario_dict["N"] = list_N_vals[n]
        scenario_num = situation_num + mod_num * n
        scenario_dict["scenario_num"] = scenario_num
        scenario_num_dirname = f"scenario_{scenario_num:04}"
        scenario_path = results_path.joinpath(scenario_num_dirname)

        datagen_params_path = results_path.joinpath(scenario_num_dirname,
                                                    "datagen_params")
        # save the file
        save_arma_mat(xi_content, "xi", datagen_params_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_info_dir_name', required=True)
    parser.add_argument('--situation_num', required=True)
    args = parser.parse_args()
    sim_info_dir_name = args.sim_info_dir_name
    situation_num = int(args.situation_num)
    # check if group equals 1
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
    # set seed
    config_file_path = run_dir.joinpath("config_simulation.toml")
    with open(config_file_path, "rb") as fileObj:
        config = tomllib.load(fileObj)
    process_dir = config['laptop_process_dir']
    number_of_replics = config['number_of_replics']
    situation_num_zb = situation_num - 1
    seed_value_situation = number_of_replics * situation_num_zb
    # generate xi's for group
    generate_xi(sim_info_dir_name, situations_list,
                desired_dict, seed_value_situation)
