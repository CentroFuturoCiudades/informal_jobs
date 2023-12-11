import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


kbins_th = [2, 3, 4]
kbins_ed = [5, 6, 7, 8, 9, 10]


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
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_ed,
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
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_th,
        }
    },

    'LR_th_dsc_ed_dsc': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot'),
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
                product(kbins_th, kbins_ed)),
        }
    },

    'LR_th_dsc_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_th,
        }
    },

    'LR_th_dsc_ed_oh': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_th,
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
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_ed,
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

    'LR_th_no_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
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

    'LR_th_no_ed_oh': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
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

    'LR_th_no_ed_dsc': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        (
                            "discretizer",
                            KBinsDiscretizer(encode='onehot'),
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
            'preprocessor__discretizer__n_bins': kbins_ed,
        }
    },
}


models_RF = {
    'RF_th_no_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("drop", "drop", ['edad_num', 'tamaño_hogar'])
                    ], remainder='passthrough')
                ),
                (
                    'classifier',
                    RandomForestClassifier()
                )
             ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': [1, 2, 3, 4, 5, 6],
            'classifier__max_leaf_nodes': [10, 100, 1000, None],
            'classifier__min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
        }
    },

    'RF_th_si_ed_cnss': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("drop", "drop", ['edad_num'])
                    ], remainder='passthrough')
                ),
                (
                    'classifier',
                    RandomForestClassifier()
                )
             ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': [1, 2, 3, 4, 5, 6, 7],
            'classifier__max_leaf_nodes': [10, 100, 1000, None],
            'classifier__min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
        }
    },

    'RF_th_no_ed_num': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("drop", "drop", ['edad_cat', 'tamaño_hogar'])
                    ], remainder='passthrough')
                ),
                (
                    'classifier',
                    RandomForestClassifier()
                )
             ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': [1, 2, 3, 4, 5, 6],
            'classifier__max_leaf_nodes': [10, 100, 1000, None],
            'classifier__min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
        }
    },

    'RF_th_si_ed_num': {
        'model': Pipeline([
                (
                    'preprocessor',
                    ColumnTransformer([
                        ("drop", "drop", ['edad_cat'])
                    ], remainder='passthrough')
                ),
                (
                    'classifier',
                    RandomForestClassifier()
                )
             ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': [1, 2, 3, 4, 5, 6, 7],
            'classifier__max_leaf_nodes': [10, 100, 1000, None],
            'classifier__min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
        }
    },
}
