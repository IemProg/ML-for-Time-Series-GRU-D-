import numpy as np
import pandas as pd
import os

idx2features = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2',
                'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'Mg', 'Na', 'PaCO2', 'PaO2', 'Platelets',
                'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']
features2idx = dict(zip(idx2features, range(len(idx2features))))


def timestamp2minute(time_str):
    """
    Converts str timestamp to minute (int). assumes time in '00:00' string format
    :param str: 
    :return: int
    """
    hours, minutes = time_str.split(':')

    return int(hours)*60 + int(minutes)


def read_record_as_time_series(record_id, data_dir='./set-a'):
    """
    Return three series: x, s, m
    x: data (timesteps, 33 features)
    s: timestamps (timesteps)
    m: binary missing indicator (timesteps, 33 features)
    """
    df = pd.read_csv(open(os.path.join(data_dir, str(record_id) + '.txt')))
    groupby = df.groupby('Time')
    ts = []
    timestamps = sorted(groupby.groups)

    for i, time in enumerate(timestamps):
        features = [np.nan] * len(idx2features)
        for i, row in groupby.get_group(time).iterrows():
            if row['Parameter'] in features2idx:
                features[features2idx[row['Parameter']]] = row['Value']
        ts.append(features)
    ts = np.array(ts)
    # convert timestamp to minutes
    timestamps = np.array([timestamp2minute(s) for s in timestamps])

    # missing values
    m = np.isnan(ts).astype(int)
    return ts, timestamps, m


if __name__ == "__main__":
    x, s, m = read_record_as_time_series(132539)
    print(x, s, m)
