import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from itertools import product


cat_cols = [
    'genero',
    'ocupacion',
    'sector',
    'escolaridad',
    'municipio',
    'edad_cat'
]

cat_cols_no_edad = [
    'genero',
    'ocupacion',
    'sector',
    'escolaridad',
    'municipio',
    'edad_cat'
]

num_cols = [
    'tamaño_hogar',
    'edad_num',
]


models_LR = {
    'LR_th_scl_ed_scl': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("standard_scaler", StandardScaler(), num_cols),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad)
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10)
        }
    },

    'LR_th_scl_ed_dsc': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("standard_scaler", StandardScaler(), ['tamaño_hogar']),
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            ['edad_num']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': [5, 6, 7, 8, 9, 10, 15, 20],
        }
    },

    'LR_th_scl_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("standard_scaler", StandardScaler(), ['tamaño_hogar']),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
        }
    },

    'LR_th_scl_ed_oh': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("standard_scaler", StandardScaler(), ['tamaño_hogar']),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad + ['edad_num']
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
        }
    },

    'LR_th_dsc_ed_scl': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("standard_scaler", StandardScaler(), ['edad_num']),
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            ['tamaño_hogar']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': [2, 3, 4, 5, 6],
        }
    },

    'LR_th_dsc_ed_dsc': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            num_cols
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': list(
                product([2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10, 15, 20])
            ),
        }
    },

    'LR_th_dsc_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            ['tamaño_hogar']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': [2, 3, 4, 5, 6],
        }
    },

    'LR_th_dsc_ed_oh': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            ['tamaño_hogar']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad + ['edad_num']
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': [2, 3, 4, 5, 6],
        }
    },

    'LR_th_oh_ed_scl': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "standard_scaler",
                            StandardScaler(),
                            ['edad_num']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad + ['tamaño_hogar']
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
        }
    },

    'LR_th_oh_ed_dsc': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot', strategy='quantile'),
                            ['edad_num']
                        ),
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad + ['tamaño_hogar']
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': [5, 6, 7, 8, 9, 10, 15, 20],
        }
    },

    'LR_th_oh_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols + ['tamaño_hogar']
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
        }
    },

    'LR_th_oh_ed_oh': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "one-hot-encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            cat_cols_no_edad + num_cols
                        )
                    ])
                ),
                (
                    'classifier',
                    LogisticRegression(max_iter=1000, solver='liblinear')
                )
             ]),
        'params': {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
        }
    },
}
