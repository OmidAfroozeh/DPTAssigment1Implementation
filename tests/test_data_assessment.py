import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer

from apt.anonymization import Anonymize
from apt.risk.data_assessment.dataset_assessment_manager import DatasetAssessmentManager, DatasetAssessmentManagerConfig
from apt.utils.dataset_utils import get_iris_dataset_np, get_diabetes_dataset_np, get_adult_dataset_pd, \
    get_nursery_dataset_pd
from apt.utils.datasets import ArrayDataset
from apt.risk.data_assessment.dataset_attack_membership_classification import DatasetAttackScoreMembershipClassification
from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import DatasetAttackScoreMembershipKnnProbabilities
from apt.risk.data_assessment.dataset_attack_whole_dataset_knn_distance import DatasetAttackScoreWholeDatasetKnnDistance

MIN_SHARE = 0.5
MIN_ROC_AUC = 0.0
MIN_PRECISION = 0.0

NUM_SYNTH_SAMPLES = 100
NUM_SYNTH_COMPONENTS = 4

iris_dataset_np = get_iris_dataset_np()
diabetes_dataset_np = get_diabetes_dataset_np()
nursery_dataset_pd = get_nursery_dataset_pd()
adult_dataset_pd = get_adult_dataset_pd()

mgr = DatasetAssessmentManager(DatasetAssessmentManagerConfig(persist_reports=False, generate_plots=False))


def teardown_function():
    print("dump_all_scores_to_files")
    mgr.dump_all_scores_to_files()


anon_testdata = ([('iris_np', iris_dataset_np, 'np', k, mgr) for k in range(2, 10, 4)]
                 + [('diabetes_np', diabetes_dataset_np, 'np', k, mgr) for k in range(2, 10, 4)]
                 + [('nursery_pd', nursery_dataset_pd, 'pd', k, mgr) for k in range(2, 10, 4)]
                 + [('adult_pd', adult_dataset_pd, 'pd', k, mgr) for k in range(2, 10, 4)])


@pytest.mark.parametrize("name, data, dataset_type, k, mgr", anon_testdata)
def test_risk_anonymization(name, data, dataset_type, k, mgr):
    (x_train, y_train), (x_test, y_test) = data

    if dataset_type == 'np':
        # no need to preprocess
        preprocessed_x_train = x_train
        preprocessed_x_test = x_test
        QI = [0, 2]
        anonymizer = Anonymize(k, QI, train_only_QI=True)
        categorical_features = []
    elif "adult" in name:
        preprocessed_x_train, preprocessed_x_test, categorical_features = preprocess_adult_x_data(x_train, x_test)
        QI = list(range(15, 27))
        anonymizer = Anonymize(k, QI)
    elif "nursery" in name:
        preprocessed_x_train, preprocessed_x_test, categorical_features = preprocess_nursery_x_data(x_train, x_test)
        QI = list(range(15, 23))
        anonymizer = Anonymize(k, QI, categorical_features=categorical_features, train_only_QI=True)
    else:
        raise ValueError('Pandas dataset missing a preprocessing step')

    anonymized_data = ArrayDataset(anonymizer.anonymize(ArrayDataset(preprocessed_x_train, y_train)))
    original_data_members = ArrayDataset(preprocessed_x_train, y_train)
    original_data_non_members = ArrayDataset(preprocessed_x_test, y_test)

    dataset_name = f'anon_k{k}_{name}'
    assess_privacy_and_validate_result(mgr, original_data_members, original_data_non_members, anonymized_data,
                                       dataset_name, categorical_features)


testdata = [('iris_np', iris_dataset_np, 'np', mgr),
            ('diabetes_np', diabetes_dataset_np, 'np', mgr),
            ('nursery_pd', nursery_dataset_pd, 'pd', mgr),
            ('adult_pd', adult_dataset_pd, 'pd', mgr)]


@pytest.mark.parametrize("name, data, dataset_type, mgr", testdata)
def test_risk_kde(name, data, dataset_type, mgr):
    (x_train, y_train), (x_test, y_test) = data

    if dataset_type == 'np':
        encoded = x_train
        encoded_test = x_test
        num_synth_components = NUM_SYNTH_COMPONENTS
        categorical_features = []
    elif "adult" in name:
        encoded, encoded_test, categorical_features = preprocess_adult_x_data(x_train, x_test)
        num_synth_components = 10
    elif "nursery" in name:
        encoded, encoded_test, categorical_features = preprocess_nursery_x_data(x_train, x_test)
        num_synth_components = 10
    else:
        raise ValueError('Pandas dataset missing a preprocessing step')

    synth_data = ArrayDataset(
        kde(NUM_SYNTH_SAMPLES, n_components=num_synth_components, original_data=encoded))
    original_data_members = ArrayDataset(encoded, y_train)
    original_data_non_members = ArrayDataset(encoded_test, y_test)

    dataset_name = 'kde' + str(NUM_SYNTH_SAMPLES) + name
    assess_privacy_and_validate_result(mgr, original_data_members, original_data_non_members, synth_data, dataset_name,
                                       categorical_features)


def kde(n_samples, n_components, original_data):
    """
    Simple synthetic data genrator: estimates the kernel density of data using a Gaussian kernel and then generates
    samples from this distribution
    """
    digit_data = original_data
    pca = PCA(n_components=n_components, whiten=False)
    data = pca.fit_transform(digit_data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)

    kde_estimator = grid.best_estimator_

    new_data = kde_estimator.sample(n_samples, random_state=0)
    new_data = pca.inverse_transform(new_data)
    return new_data


def preprocess_adult_x_data(x_train, x_test):
    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']

    # prepare data for DT
    def to_float(x):
        return x.astype(float)

    numeric_features = [f for f in features if f not in categorical_features]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
               ('to_float', FunctionTransformer(to_float, feature_names_out='one-to-one'))]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x_train)
    preprocessor.fit(x_train)
    encoded_test = preprocessor.transform(x_test)
    return encoded, encoded_test, filter_categorical(preprocessor.get_feature_names_out(), return_feature_names=False)


def preprocess_nursery_x_data(x_train, x_test):
    x_train = x_train.astype(str)
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]
    # prepare data for DT
    numeric_features = [f for f in features if f not in categorical_features]
    numeric_transformer = OrdinalEncoder(encoded_missing_value=-1)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x_train)
    preprocessor.fit(x_train)
    encoded_test = preprocessor.transform(x_test)
    return encoded, encoded_test, filter_categorical(preprocessor.get_feature_names_out(), return_feature_names=False)


def filter_categorical(feature_names, return_feature_names: bool = True):
    feature_name_strs = feature_names.astype('U')
    if return_feature_names:
        return list(feature_names[np.char.startswith(feature_name_strs, 'cat__')])
    else:
        return list(np.flatnonzero(np.char.startswith(feature_name_strs, 'cat__')))


def assess_privacy_and_validate_result(dataset_assessment_manager, original_data_members, original_data_non_members,
                                       synth_data, dataset_name, categorical_features):
    attack_scores = dataset_assessment_manager.assess(original_data_members, original_data_non_members, synth_data,
                                                      dataset_name, categorical_features)

    for i, (assessment_type, scores) in enumerate(attack_scores.items()):
        if assessment_type == 'MembershipKnnProbabilities':
            score_g: DatasetAttackScoreMembershipKnnProbabilities = scores[0]
            assert score_g.roc_auc_score > MIN_ROC_AUC
            assert score_g.average_precision_score > MIN_PRECISION
        elif assessment_type == 'WholeDatasetKnnDistance':
            score_h: DatasetAttackScoreWholeDatasetKnnDistance = scores[0]
            assert score_h.share > MIN_SHARE
        if assessment_type == 'MembershipClassification':
            score_mc: DatasetAttackScoreMembershipClassification = scores[0]
            assert score_mc.synthetic_data_quality_warning is False
            assert 0 <= score_mc.normalized_ratio <= 1
