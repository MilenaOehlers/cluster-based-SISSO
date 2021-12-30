import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Union
import json

class ExecutionFolder():
    def __init__(self, permanent_location: Union[Path, str, None], refers_to_data_file: Union[Path, str]):
        execution_folder_path = Path(refers_to_data_file).parent.joinpath("tmp_sisso_exe_folder") \
            if permanent_location is None \
            else Path(permanent_location)
        execution_folder_path.mkdir(parents=True, exist_ok=True)
        self.is_permanent = permanent_location is not None
        self.path = execution_folder_path

    def delete_if_not_permanent(self):
        if not self.is_permanent:
            shutil.rmtree(self.path, ignore_errors=False, onerror=None)
            self.path = None

def raise_exception(txt:str):
    raise Exception(txt)

def as_inputs(inputs_jsonfilepath, data_file, property_key,
              task_key=None, opset=['add', 'sub', 'mult', 'div', 'sq', 'cb', 'cbrt', 'sqrt'],
              param_opset=[], calc_type='regression', desc_dim=3, n_sis_select=100, max_rung=2,
              n_residual=1, n_models_store=1, n_rung_store=1, n_rung_generate=0, min_abs_feat_val=1e-05,
              max_abs_feat_val=100000000.0, leave_out_inds=[], leave_out_frac=0.25, fix_intercept=False,
              max_feat_cross_correlation=1.0, nlopt_seed=13, global_param_opt=False, reparam_residual=True
              ):
    """writes jsonfile with sisso_execution- or derived_space_construction_parameters to inputs_jsonfilepath
    and returns inputs_jsonfilepath"""
    jsondict = {'data_file': str(data_file),
                'property_key': property_key,
                'leave_out_inds': leave_out_inds,
                'leave_out_frac': leave_out_frac,
                'task_key': task_key,
                'opset': opset,
                'param_opset': param_opset,
                'calc_type': calc_type,
                'desc_dim': desc_dim,
                'n_sis_select': n_sis_select,
                'max_rung': max_rung,
                'n_residual': n_residual,
                'n_models_store': n_models_store,
                'n_rung_store': n_rung_store,
                'n_rung_generate': n_rung_generate,
                'min_abs_feat_val': min_abs_feat_val,
                'max_abs_feat_val': max_abs_feat_val,
                'fix_intercept': fix_intercept,
                'max_feat_cross_correlation': max_feat_cross_correlation,
                'nlopt_seed': nlopt_seed,
                'global_param_opt': global_param_opt,
                'reparam_residual': reparam_residual}
    if Path(inputs_jsonfilepath).suffix != '.json':
        inputs_jsonfilepath = inputs_jsonfilepath.with_suffix('.json')
        print("'.json' was appended to inputs_jsonfilepath")
    try:
        jsondict['leave_out_inds'] = [int(ind) for ind in jsondict['leave_out_inds']]
    except:
        data = pd.read_csv(jsondict['data_file'], sep=',', index_col='material')
        jsondict['leave_out_inds'] = [list(data.index).index(mat) for mat in jsondict['leave_out_inds']]
    with open(inputs_jsonfilepath, 'w') as jsonfile:
        json.dump(jsondict, jsonfile)
    return str(inputs_jsonfilepath)


if False:

    def force_leave_out_inds_representation(**space_params):
        assert set(["data_file", "property_key", "leave_out_inds", "leave_out_frac"]).issubset(set(space_params.keys())), \
            'which_space must contain keys "data_file", "property_key", "leave_out_inds", "leave_out_frac"'
        testset_chosenby_index = (isinstance(space_params['leave_out_inds'], list)
                                  and len(space_params['leave_out_inds']) > 0)
        testset_chosen_randomly = not testset_chosenby_index \
                                  and (isinstance(space_params['leave_out_frac'], tuple([int, float])) \
                                       and 0 <= space_params['leave_out_frac'] < 1)
        space_params['data_file'] = str(space_params['data_file'])
        print(space_params['data_file'])
        whole_set = pd.read_csv(space_params['data_file'], index_col='material')
        work_set = whole_set.loc[set(whole_set.index) - set(space_params['leave_out_inds']), :] if testset_chosenby_index \
            else whole_set if space_params['leave_out_frac'] == 0.0 \
            else train_test_split(whole_set, test_size=space_params['leave_out_frac'])[0] if testset_chosen_randomly \
            else raise_exception(
            "leave_out_inds must be list of length > 0 and/or leave_out_frac must be float between 0 and 1")
        space_params['leave_out_inds'] = [list(whole_set.index).index(mat)
                                          for mat in list(set(whole_set.index) - set(work_set.index))]
        space_params['leave_out_frac'] = None
        return space_params

    def primary_space_construction_parameters(basic_params_path:Path, data_file, property_key,
                                              leave_out_inds=[], leave_out_frac=0.25):
        """returns dict of all parameters that determine primary space
        data_file: str or Path that points to csv file
        for meaning of the other args, please refer to https://sissopp_developers.gitlab.io/sissopp/quick_start/code_ref.html#input-files
        """
        primary_space_dict = {'data_file': data_file,
                              'property_key': property_key,
                              'leave_out_inds': leave_out_inds,
                              'leave_out_frac': leave_out_frac}
        with open(basic_params_path, 'w') as jsonfile:
            json.dump(primary_space_dict, jsonfile)
        return force_leave_out_inds_representation(**primary_space_dict)
