from pathlib import Path


class Paths:
    # general
    IMMOSCOUT_SOURCE_DATA = Path('data/source/immoscout_v2.zip')

    # regressor
    REGRESSOR_DATA_WRANGLING_DATA = Path('data/regressor/01_0_data_wrangling.dump')
    REGRESSOR_SCALING_DATA = Path('data/regressor/02_0_scaling.dump')
    @staticmethod
    def REGRESSOR_RELEVANT_FEATURES_DATA(threshold):
        return Path(f'data/regressor/helpers/relevant_features_{threshold}.dump')

    @staticmethod
    def REGRESSOR_MODEL_DATA(model):
        return Path(f'models/regressor/{model}.dump')

    # kaggle
    KAGGLE_SOURCE_DATA = Path('data/source/test_data-Kaggle-v0.11.csv.zip')
    KAGGLE_DATA_WRANGLING_DATA = Path('data/kaggle/01_0_data_wrangling.dump')
    KAGGLE_IDS_TYPE_NONE_DATA = Path('data/kaggle/01_1_no_type_ids_id.dump')
    KAGGLE_IDS_TO_PREDICT_DATA = Path('data/kaggle/01_2_ids_to_predict.dump')
    KAGGLE_SCALING_DATA = Path('data/kaggle/02_0_scaling.dump')

    @staticmethod
    def KAGGLE_SUBMISSIONS_PATH(filename):
        return Path(f'data/kaggle/submissions/{filename}.csv')

    # web service
    WEBSERVICE_META_DATA = Path('data/web_service/01_0_meta_data.dump')

    # classifier
    CLASSIFIER_DATA_WRANGLING_DATA = Path('data/classifier/01_0_data_wrangling.dump')
    CLASSIFIER_SCALING_DATA = Path('data/classifier/02_0_scaling.dump')

    @staticmethod
    def CLASSIFIER_MODEL_DATA(model):
        return Path(f'models/classifier/{model}.dump')
