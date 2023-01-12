from functools import reduce

import pandas as pd


def scale_and_predict(model, metadata, inputs) -> str:
    feature_means = metadata['feature_means']
    scaler = metadata['scaler']
    power_options = metadata['power_options']
    options = metadata['options']

    df = pd.DataFrame({
        **inputs,
        **reduce(
            lambda state, x: {**state, f'type_unified_{x}': 1. if x == inputs['type_unified'] else 0.},
            options['type_unified'],
            {}
        ),
        **feature_means[inputs['Zip']]
    }, index=[0])
    
    df.drop(['type_unified'], 
            axis=1, 
            inplace=True)
    
    df = df.reindex(columns=scaler.feature_names_in_)

    df[df.columns] = scaler.transform(df)

    power_values = {}
    for col in [
        col for col in df.columns
        if not col.startswith('type_unified')
    ]:
        for p in power_options:
            power_values[f'{col}_{p}'] = df[col] ** p

    df = pd.concat(
            [df, pd.DataFrame(power_values)],
            axis=1
        )[model.feature_names_in_]

    return model.predict(df)[0]
