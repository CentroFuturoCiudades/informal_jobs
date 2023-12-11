import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
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
                    LogisticRegression(max_iter=1000)
                )
             ]),
        'params': {
            'classifier__C': np.geomspace(1e-4, 1e4, 10),
            'preprocessor__discretizer__n_bins': kbins_ed,
        }
    },
}

max_leaf_nodes_grid = [5, 7, 10, 25, 50, 65, 85, 100, 250, 500]

models_RF = {
    'RF_th_no_ed_cnss': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols
                    )
                ])
            ),
            (
                'classifier',
                RandomForestClassifier()
            )
        ]),
        'params': {
            'classifier__max_features': [1, 2, 3, 4, 5, 6],
            'classifier__max_leaf_nodes': max_leaf_nodes_grid,
        }
    },

    'RF_th_si_ed_cnss': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols
                    ),
                    ('passthrough', 'passthrough', ['tamaño_hogar'])
                ])
            ),
            (
                'classifier',
                RandomForestClassifier()
            )
        ]),
        'params': {
            'classifier__max_features': [1, 2, 3, 4, 5, 6, 7],
            'classifier__max_leaf_nodes': max_leaf_nodes_grid,
        }
    },

    'RF_th_no_ed_num': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols_no_edad
                    ),
                    ('passthrough', 'passthrough', ['edad_num'])
                ])
            ),
            (
                'classifier',
                RandomForestClassifier()
            )
        ]),
        'params': {
            'classifier__max_features': [1, 2, 3, 4, 5, 6],
            'classifier__max_leaf_nodes': max_leaf_nodes_grid,
        }
    },

    'RF_th_si_ed_num': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols_no_edad
                    ),
                    ('passthrough', 'passthrough', ['edad_num', 'tamaño_hogar'])
                ])
            ),
            (
                'classifier',
                RandomForestClassifier()
            )
        ]),
        'params': {
            'classifier__max_features': [1, 2, 3, 4, 5, 6, 7],
            'classifier__max_leaf_nodes': max_leaf_nodes_grid,
        }
    },
}

models_GB = {
    'GB_th_no_ed_cnss': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols
                    )
                ])
            ),
            (
                'classifier',
                HistGradientBoostingClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=0
                )
            )
        ]),
        'params': {
            'classifier__learning_rate': np.geomspace(0.01, 1, 10),
            'classifier__max_leaf_nodes': [2, 5, 10, 20, 50, 100, 500, 1000, None],
        }
    },

    'GB_th_si_ed_cnss': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols
                    ),
                    ('passthrough', 'passthrough', ['tamaño_hogar'])
                ])
            ),
            (
                'classifier',
                HistGradientBoostingClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=0
                )
            )
        ]),
        'params': {
            'classifier__learning_rate': np.geomspace(0.01, 1, 10),
            'classifier__max_leaf_nodes': [2, 5, 10, 20, 50, 100, 500, 1000, None],
        }
    },

    'GB_th_no_ed_num': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols_no_edad
                    ),
                    ('passthrough', 'passthrough', ['edad_num'])
                ])
            ),
            (
                'classifier',
                HistGradientBoostingClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=0
                )
            )
        ]),
        'params': {
            'classifier__learning_rate': np.geomspace(0.01, 1, 10),
            'classifier__max_leaf_nodes': [2, 5, 10, 20, 50, 100, 500, 1000, None],
        }
    },

    'GB_th_si_ed_num': {
        'model': Pipeline([
            (
                'preprocessor',
                ColumnTransformer([
                    (
                        "od_encoder",
                        OrdinalEncoder(
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        ),
                        cat_cols_no_edad
                    ),
                    ('passthrough', 'passthrough', ['edad_num', 'tamaño_hogar'])
                ])
            ),
            (
                'classifier',
                HistGradientBoostingClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=0
                )
            )
        ]),
        'params': {
            'classifier__learning_rate': np.geomspace(0.01, 1, 10),
            'classifier__max_leaf_nodes': [2, 5, 10, 20, 50, 100, 500, 1000, None],
        }
    },
}
