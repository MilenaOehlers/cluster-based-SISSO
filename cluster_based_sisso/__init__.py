from typing import Dict,List,Tuple,Union,NamedTuple,Optional
from typing_extensions import Literal
import json,re
import daa_luigi
from common_functions import ExecutionFolder,raise_exception,as_inputs
from copy import copy
import pandas as pd
from pathlib import Path
import sissopp
from sissopp.py_interface import get_fs_solver
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Repr2Members = Dict[str,List[str]]
ExecutionType = Literal["singletask_on_representatives", "multitask_on_all"]

class PrimarySpaceParams(NamedTuple):
    data_file:str
    property_key:str = 'lat'
    leave_out_inds: List[int] = []
    leave_out_frac: Optional[float] = 0.25
    def deterministic(self) -> "PrimarySpaceParams":
        testset_chosenby_index = (len(self.leave_out_inds) > 0)
        testset_chosen_randomly = not testset_chosenby_index and (0 <= self.leave_out_frac < 1)
        whole_set = pd.read_csv(self.data_file, index_col='material')
        work_set = whole_set.loc[set(whole_set.index) - set(self.leave_out_inds), :] if testset_chosenby_index \
            else whole_set if self.leave_out_frac == 0.0 \
            else train_test_split(whole_set, test_size=self.leave_out_frac)[0] if testset_chosen_randomly \
            else raise_exception(
            "leave_out_inds must be list of length > 0 and/or leave_out_frac must be float between 0 and 1")
        leave_out_inds = [list(whole_set.index).index(mat)
                          for mat in list(set(whole_set.index) - set(work_set.index))]
        return PrimarySpaceParams(str(self.data_file),self.property_key,leave_out_inds,None)

class DerivedSpaceParams(NamedTuple):
    data_file: str
    property_key: str = 'lat'
    leave_out_inds: List[int] = []
    leave_out_frac: float = 0.25
    task_key:str=None
    opset:List[Literal['add', 'sub', 'mult', 'div', 'sq', 'cb', 'cbrt', 'sqrt']] = ['add', 'sub', 'mult', 'div', 'sq', 'cb', 'cbrt', 'sqrt']
    param_opset:List[Literal['add', 'sub', 'mult', 'div', 'sq', 'cb', 'cbrt', 'sqrt']] = ['add', 'sub', 'mult', 'div', 'sq', 'cb', 'cbrt', 'sqrt']
    calc_type:Literal["regression", "log regression", "classification"] = "regression"
    desc_dim:int = 3
    n_sis_select:int = 100
    max_rung:int = 2
    n_residual:int = 1
    n_models_store: int = 1
    n_rung_store:int =1
    n_rung_generate:int = 0
    min_abs_feat_val: float = 1e-05
    max_abs_feat_val:float = 100000000.0
    fix_intercept:bool = False
    max_feat_cross_correlation:float = 1.0
    nlopt_seed:int = 13
    global_param_opt:bool = False
    reparam_residual:bool = True
    def deterministic(self) -> "DerivedSpaceParams":
        testset_chosenby_index = (len(self.leave_out_inds) > 0)
        testset_chosen_randomly = not testset_chosenby_index and (0 <= self.leave_out_frac < 1)
        whole_set = pd.read_csv(self.data_file, index_col='material')
        work_set = whole_set.loc[set(whole_set.index) - set(self.leave_out_inds), :] if testset_chosenby_index \
            else whole_set if self.leave_out_frac == 0.0 \
            else train_test_split(whole_set, test_size=self.leave_out_frac)[0] if testset_chosen_randomly \
            else raise_exception(
            "leave_out_inds must be list of length > 0 and/or leave_out_frac must be float between 0 and 1")
        leave_out_inds = [list(whole_set.index).index(mat)
                          for mat in list(set(whole_set.index) - set(work_set.index))]
        deterministic_repr = self._asdict()
        deterministic_repr.update({"data_file":str(self.data_file),
            "leave_out_inds":leave_out_inds, "leave_out_frac":None})
        return DerivedSpaceParams(**deterministic_repr)
    def get_primary_space_params(self)->PrimarySpaceParams:
        return PrimarySpaceParams(**{k:v for k,v in self._asdict().items()
                                     if k in ['data_file','property_key','leave_out_inds','leave_out_frac']})

data_csvpath = "/home/oehlers/Documents/masterthesis/02-data/csvs/cubic_perovskites.csv"

class MyKmeans():
    def __init__(self, space_params: Union[PrimarySpaceParams,DerivedSpaceParams],
                 n_clusters: int,
                 save_proto2members_at: Union[Path,str] = None,
                 interm_results_path: Union[Path,str] = None):
        """
        In the following, the Work Set will be defined as the dataset that remains after subtracting the Test Set.
        Cluster extraction is based on Work Set only; cluster assignment of Test Set materials is conducted afterwards.
        This function returns dictionary of Work Set cluster centers pointing to list of Work and Test Set cluster members
        space_params (short: space_params): dictionary returned by either
                     PrimarySpaceParams or DerivedSpaceParams class,
                     determines in which space kmeans will be applied
                     (either the space spanned by standardized primary features without target property,
                     or the space spanned by the derived features times respective fitting coefficients of sisso
                     - in the latter case, all execution parameters for sisso are defined by the
                     DerivedSpaceParams class)
        n_clusters: number of clusters
        save_proto2members_at: path to future file location
        interm_results_path: if set to path or str, folder it points to is created if non-existent and used
                                       if set to None, temp folder is created in data_file folder and deleted after exe
        """
        self.space_params = space_params
        self.n_clusters = n_clusters
        self.save_proto2members_at = save_proto2members_at
        self.interm_results_path = interm_results_path
        self.proto2members = self._get_kmeans_center2members()

    def _train_and_test_streched_derived_feature_dfs(self, sisso, exe_folder: ExecutionFolder) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """"sisso must be SISSOregressor as returned by 'feat_space,sisso = get_fs_solver
        (for some reason this cannot be indicated via typing)
        """
        # Extract derived space from SISSO results by reading train and testfile:
        # (no python binding was found for directly extracting derived space)

        streched_derived_feature_space = {}
        for train_or_test in ['train', 'test']:
            filepath = exe_folder.path.joinpath(train_or_test + "file")
            derivedspacepath = exe_folder.path.joinpath(train_or_test + "_derived_space.csv")
            sisso.models[-1][0].to_file(str(filepath), train_or_test == train_or_test)
            with open(filepath, "r") as file:
                with open(derivedspacepath, 'w') as derivedspace:
                    lines = file.readlines()
                    writeit = 0
                    for line in lines:
                        if line[:len("# Sample ID")] == "# Sample ID":
                            line = line.replace("# Sample ID", "material")
                            writeit = 1
                        if writeit == 1:
                            derivedspace.write(line)
            target_and_derived_features = pd.read_csv(derivedspacepath, sep=',', index_col=0)
            derived_features = target_and_derived_features.iloc[:, 2:]
            strech_coefs = sisso.models[-1][0].coefs[0][:-1]
            streched_derived_features = derived_features.multiply(strech_coefs).values
            streched_derived_feature_space[train_or_test] = streched_derived_features

        return streched_derived_feature_space['train'], streched_derived_feature_space['test']

    def _get_kmeans_center2members(self) -> Repr2Members:

        space_params = self.space_params

        testset_chosenby_index = ( len(space_params.leave_out_inds)>0 and space_params.leave_out_frac is None )
        clustering_in_primary_space = isinstance(space_params,PrimarySpaceParams)
        clustering_in_derived_space = isinstance(space_params,DerivedSpaceParams)

        whole_set = pd.read_csv(space_params.data_file, index_col='material').astype(float)
        test_set_materials = [list(whole_set.index)[ind] for ind in space_params.leave_out_inds]
        test_set = whole_set.loc[test_set_materials, :]
        work_set = whole_set.loc[set(whole_set.index) - set(test_set_materials), :]

        if testset_chosenby_index and clustering_in_primary_space:
            # Get standardized primary space without target:
            target_property_col = [col for col in whole_set.columns
                                   if re.match(r"{} (...)".format(space_params.property_key), col)][0]
            work_primary_features = work_set.drop(target_property_col, axis=1).values
            test_primary_features = test_set.drop(target_property_col, axis=1).values
            standardized_work_primary_features = StandardScaler().fit_transform(work_primary_features)
            standardized_test_primary_features = StandardScaler().fit_transform(test_primary_features)

            # Use standardized primary space for Kmeans clustering:
            kmeans_results = KMeans(self.n_clusters).fit(standardized_work_primary_features)
            virtual_cluster_centers = kmeans_results.cluster_centers_
            actual_cluster_center_inds = pairwise_distances_argmin_min(virtual_cluster_centers,
                                                                       standardized_work_primary_features)[0]
            test_labels = kmeans_results.predict(standardized_test_primary_features)

        elif testset_chosenby_index and clustering_in_derived_space:
            assert space_params.task_key is None, """This code applies to clustering in space derived by Single-Task SISSO only,
            for clustering in Multi-Task-SISSO space code has to be checked for necessary adaptations"""

            # Create execution folder:
            exe_folder = ExecutionFolder(permanent_location=self.interm_results_path,
                                         refers_to_data_file=space_params.data_file)

            # Execute SISSO:
            space_params_dict = space_params._asdict()
            inp = as_inputs(exe_folder.path.joinpath("derived_space_constr_params"), **space_params_dict)
            print(inp)
            inputs = sissopp.Inputs(inp)
            feature_space, sisso = get_fs_solver(inputs)
            sisso.fit()

            # Extract streched derived space:
            work_streched_derived_features, test_streched_derived_features \
                = self._train_and_test_streched_derived_feature_dfs(sisso, exe_folder)

            # Use streched derived space for Kmeans clustering:
            kmeans_results = KMeans(self.n_clusters).fit(work_streched_derived_features)
            virtual_cluster_centers = kmeans_results.cluster_centers_
            actual_cluster_center_inds = pairwise_distances_argmin_min(virtual_cluster_centers,
                                                                       work_streched_derived_features)[0]
            test_labels = kmeans_results.predict(test_streched_derived_features)

            # Remove execution folder if set to be temporal:
            exe_folder.delete_if_not_permanent()

        else:
            raise Exception("""space_construction_parameters must be either of PrimarySpaceParams or DerivedSpaceParams 
            class""")

        # Determine cluster center and member material names:
        actual_cluster_center_materials = [list(work_set.index)[ind] for ind in actual_cluster_center_inds]

        work_set_with_tasks = copy(work_set).assign(task=kmeans_results.labels_)
        test_set_with_tasks = copy(test_set).assign(task=test_labels)
        whole_set_with_tasks = work_set_with_tasks.append(test_set_with_tasks)

        center2members = {}
        for task in set(whole_set_with_tasks['task']):
            members = list(whole_set_with_tasks[whole_set_with_tasks['task'] == task].index)
            center = list(set(actual_cluster_center_materials).intersection(set(members)))[0]
            center2members[center] = members
        if self.save_proto2members_at is not None:
            with open(self.save_proto2members_at, 'w') as jsonfile:
                json.dump(center2members, jsonfile)
        return center2members

class MyDeepAA():
    def __init__(self,space_params:Union[PrimarySpaceParams,DerivedSpaceParams],
                 at_loss_factor:float, target_loss_factor:float, recon_loss_factor:float, kl_loss_factor:float,
                 latent_dim:int,n_epochs:int,arche2members_path:Union[str,Path]=None):
        self.space_params = space_params
        self.at_loss_factor = at_loss_factor
        self.target_loss_factor = target_loss_factor
        self.recon_loss_factor = recon_loss_factor
        self.kl_loss_factor = kl_loss_factor
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.arche2members_path = arche2members_path

        self.nn = daa_luigi.build_network(latent_dim=latent_dim,epochs=n_epochs)
        As, arche2members = self._extract_weightdfs_and_arche2members()
        self.weight_dfs = As
        self.arche2members = arche2members

    def _get_stan_work_and_test_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        whole_set = pd.read_csv(self.space_params.data_file, index_col='material').astype(float)
        test_set_materials = [list(whole_set.index)[ind] for ind in self.space_params.leave_out_inds]
        test_set = whole_set.loc[test_set_materials, :]
        work_set = whole_set.loc[set(whole_set.index) - set(test_set_materials), :]
        # Standardize iot weigh reconstruction of each feature equally important
        work_mean, work_std = work_set.mean(axis=0), work_set.std(axis=0)
        stan_work_set = (work_set - work_mean) / work_std
        stan_test_set = (test_set - work_mean) / work_std
        return stan_work_set, stan_test_set

    def _get_nn_data_input(self) -> Dict:
        standardized_work, standardized_test = self._get_stan_work_and_test_df()
        target = [col for col in standardized_work if self.space_params.property_key in col][0]
        nn_data_input = {'train_feat': standardized_work.drop([target], axis=1).values,
                         'train_targets': standardized_work[target].values,
                         'test_feat': standardized_test.drop([target], axis=1).values,
                         'test_targets': standardized_test[target].values}
        return nn_data_input

    def _extract_weightdfs_and_arche2members(self) -> Tuple[Dict[str,pd.DataFrame],Repr2Members]:
        """
        nn_results: result yielded by nn
        space_params: yielded by primary_space_construction_parameters function
        arche2members_path: path to future file location
        """
        nn_data_input = self._get_nn_data_input()
        nn_results = self.nn(nn_data_input, self.at_loss_factor, self.target_loss_factor,
                             self.recon_loss_factor, self.kl_loss_factor)
        stan_work_set, stan_test_set = self._get_stan_work_and_test_df()
        As = {'train': pd.DataFrame(nn_results[('train', 'latent space', 'As')], index=stan_work_set.index),
              'test': pd.DataFrame(nn_results[('test', 'latent space', 'As')], index=stan_test_set.index)}

        # collect actual arche and resp. clusters IFF virtual arche really the one closest to actual arche candidate
        def mat_closest_to(virtualarche: int, train_or_test: str = 'train') -> str:
            return As[train_or_test].loc[:, virtualarche].idxmax()

        def virtualarche_clostest_to(mat: str, train_or_test: str = 'train') -> int:
            return As[train_or_test].loc[mat, :].idxmax()

        actualarches = [mat_closest_to(virtualarche, 'train') for virtualarche in As['train'].columns
                        if virtualarche_clostest_to(mat_closest_to(virtualarche, 'train'), 'train') == virtualarche]
        assert len(actualarches) == len(As['train'].columns), \
            "For at least one virtual archetype, no material could be found which would be assigned to it, when all " \
            "materials are assigned to closest virtual archetype in latent space; latent space cannot be used; " \
            "NN has to be retrained "
        for train_or_test in ['train', 'test']:
            As[train_or_test].columns = actualarches
        arche2members = {arche: [] for arche in As['train'].columns}
        for train_or_test in ['train', 'test']:
            for mat in As[train_or_test].index:
                arche = As[train_or_test].loc[mat, :].idxmax()
                arche2members[arche] += [mat]
        with open(self.arche2members_path, 'w') as jsonfile:
            json.dump(arche2members, jsonfile)
        return As, arche2members

class MySisso():
    def __init__(self,execution_parameters:DerivedSpaceParams,
              clusters:Repr2Members,
              singletask_on_representatives_or_multitask_on_all:ExecutionType = "singletask_on_representatives",
              store_intermediate_results_in:Path = None):
        self.execution_parameters = execution_parameters
        self.clusters = clusters
        self.singletask_on_representatives_or_multitask_on_all = singletask_on_representatives_or_multitask_on_all
        self.intermediate_results_path = store_intermediate_results_in
        self.results = self._get_sisso()

    def _get_sisso(self):
        """returns SISSOregressor, see https://sissopp_developers.gitlab.io/sissopp/quick_start/code_ref.html#input-files
        sisso_execution_parameters (short: sisso_params): must be output of function sisso_execution_parameters
            or derived_space_construction_parameters
        clusters: must be dict yielded by get_kmeans_center2members_dict function
        singletask_on_representatives_or_multitask_on_all:
                        if set to 'singletask_on_representatives', single-task sisso is trained on representatives, and
                                             tested on materials determined by sisso_execution_parameters['leave_out_inds']
                        if set to 'multitask_on_all', multi-task sisso is trained on all Work Set materials, an tested
                                             on all Test Set materials as determined by
                                             sisso_execution_parameters['leave_out_inds'],
                                             where tasks are determined by :clusters: arg
        store_intermediate_results_in: if set to None, temporal folder will be created and deleted after execution
        """
        sisso_params = self.execution_parameters._asdict()

        # Create execution folder:
        exe_folder = ExecutionFolder(permanent_location=self.intermediate_results_path,
                                     refers_to_data_file=sisso_params['data_file'])

        # Prepare csv and json file for sisso execution:
        whole_set = pd.read_csv(sisso_params['data_file'], sep=',', index_col='material')
        test_materials = [list(whole_set.index)[ind] for ind in sisso_params['leave_out_inds']]

        whole_data_filepath = Path(sisso_params['data_file'])
        training_and_test_data_filepath = exe_folder.path.joinpath("training_and_test_" + whole_data_filepath.name)

        if self.singletask_on_representatives_or_multitask_on_all == 'singletask_on_representatives':
            training_materials = list(self.clusters.keys())
            training_and_test_set = whole_set.loc[training_materials + test_materials, :]

        elif self.singletask_on_representatives_or_multitask_on_all == 'multitask_on_all':
            mat2center = {mat: center for center in self.clusters.keys()
                          for mat in self.clusters[center]}
            tasks = [str(mat2center[mat_in_sequence]) for mat_in_sequence in whole_set.index]
            training_and_test_set = whole_set.assign(task=tasks)
            training_and_test_set = training_and_test_set[['task'] +
                                                          [col for col in list(whole_set.columns) if col != 'task']]
            sisso_params['task_key'] = 'task'

        else:
            raise_exception(""":singletask_on_centers_or_multitask_on_all: must be set to 'singletask_on_centers'
            or 'multitask_on_all'""")

        training_and_test_set.to_csv(training_and_test_data_filepath, sep=',')
        sisso_params['data_file'] = training_and_test_data_filepath
        sisso_params['leave_out_inds'] = [list(training_and_test_set.index).index(mat) for mat in test_materials]

        # Execute SISSO and extract results:
        inputspath = as_inputs(exe_folder.path.joinpath("sisso_exe"), **sisso_params)
        inputs = sissopp.Inputs(inputspath)
        feature_space, sisso = get_fs_solver(inputs)
        sisso.fit()

        # Remove execution folder if set to be temporal:
        exe_folder.delete_if_not_permanent()

        return sisso
