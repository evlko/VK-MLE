import json

from sklearn.preprocessing import MinMaxScaler


class ScalerSerializator:
    def __init__(self, scaler: MinMaxScaler) -> None:
        self.scaler = scaler

    def serialize(self, file_path: str) -> None:
        features = self.scaler.feature_names_in_
        mins, maxs = self.scaler.data_min_, self.scaler.data_max_
        min_dict, max_dict = dict(zip(features, mins)), dict(zip(features, maxs))
        result = {"max": max_dict, "min": min_dict}
        with open(file_path, "w") as json_file:
            json.dump(result, json_file)
