import copy
from math import log
from clients import ClientsGroup, client

def aggrated_FedAvg(list_dicts_local_params, list_nums_local_data):
    # fedavg
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

def aggrated_Fedthree(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_alpk):
    # fedthree
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        list_valuse_beta = []
        for dict_local_params, num_local_data, num_minclass_data, alpk in zip(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_alpk):
            # print(dict_local_params[name_param])
            # print(num_local_data)
            # print(num_minclass_data)
            # print(alpk)
            list_values_param.append(dict_local_params[name_param] * num_local_data * num_minclass_data * alpk)
            list_valuse_beta.append(num_local_data * num_minclass_data * alpk)
        # print(list_values_param)
        value_global_param = sum(list_values_param) / (sum(list_nums_local_data) * sum(list_nums_minclass_data) * log(2))
        value_global_beta = sum(list_valuse_beta) / (sum(list_nums_local_data) * sum(list_nums_minclass_data) * log(2))
        fedavg_global_params[name_param] = value_global_param / value_global_beta
    return fedavg_global_params

def aggrated_Fedthree_GINI(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_GINI):
    # fedthree_GINI
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        list_valuse_beta = []
        for dict_local_params, num_local_data, num_minclass_data, GINI in zip(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_GINI):
            # print(dict_local_params[name_param])
            # print(num_local_data)
            # print(num_minclass_data)
            # print(alpk)
            list_values_param.append(dict_local_params[name_param] * num_local_data * num_minclass_data * GINI)
            list_valuse_beta.append(num_local_data * num_minclass_data * GINI)
        # print(list_values_param)
        value_global_param = sum(list_values_param) / (sum(list_nums_local_data) * sum(list_nums_minclass_data) * (1/2))
        value_global_beta = sum(list_valuse_beta) / (sum(list_nums_local_data) * sum(list_nums_minclass_data) * (1/2))
        fedavg_global_params[name_param] = value_global_param / value_global_beta
    return fedavg_global_params

def aggrated_Fedthree_inform(list_dicts_local_params, list_nums_local_data, list_inform, list_alpk):
    # fedthree_inform
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        list_valuse_beta = []
        for dict_local_params, num_local_data, inform, alpk in zip(list_dicts_local_params, list_nums_local_data, list_inform, list_alpk):
            # print(dict_local_params[name_param])
            # print(num_local_data)
            # print(num_minclass_data)
            # print(alpk)
            list_values_param.append(dict_local_params[name_param] * num_local_data * inform * alpk)
            list_valuse_beta.append(num_local_data * inform * alpk)
        # print(list_values_param)
        value_global_param = sum(list_values_param) / (sum(list_nums_local_data) * sum(list_inform) * log(2))
        value_global_beta = sum(list_valuse_beta) / (sum(list_nums_local_data) * sum(list_inform) * log(2))
        fedavg_global_params[name_param] = value_global_param / value_global_beta
    return fedavg_global_params

def aggrated_Fed_inform(list_dicts_local_params, list_inform):
    # fedthree_inform
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        list_valuse_beta = []
        for dict_local_params, inform in zip(list_dicts_local_params,  list_inform):
            list_values_param.append(dict_local_params[name_param] * inform)
        # print(list_values_param)
        value_global_param = sum(list_values_param) / sum(list_inform)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

def aggrated_Fedtwo_inform(list_dicts_local_params, list_nums_local_data, list_inform):
    # fedthree_inform
    fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        list_valuse_beta = []
        list_value_param = []
        for dict_local_params, num_local_data, inform in zip(list_dicts_local_params, list_nums_local_data, list_inform):
            # print(dict_local_params[name_param])
            # print(num_local_data)
            # print(num_minclass_data)
            # print(alpk)
            list_values_param.append(dict_local_params[name_param] * num_local_data * inform)
            list_valuse_beta.append(num_local_data * inform)
            list_value_param.append(dict_local_params[name_param] * num_local_data)
        # print(list_values_param)
        if sum(list_inform) == 0:
            value_global_param = sum(list_value_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        else:
            value_global_param = sum(list_values_param) / (sum(list_nums_local_data) * sum(list_inform) )
            value_global_beta = sum(list_valuse_beta) / (sum(list_nums_local_data) * sum(list_inform))
            fedavg_global_params[name_param] = value_global_param / value_global_beta
    return fedavg_global_params