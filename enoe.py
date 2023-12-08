import pandas as pd


def load_enoe(enoe_path='enoe_juntado4t19.csv'):

    enoe = (
        pd.read_csv(enoe_path, low_memory=False)
        .pipe(
            lambda df: df.assign(
                tamaño_hogar=df.groupby(
                    ['CD_A', 'ENT', 'CON', 'V_SEL']
                ).transform('size')
            )
        )
        .query("`P1.coe1` == 1")
        .set_index(['CD_A', 'ENT', 'CON', 'V_SEL'])
        .loc[
            (slice(None), 19),
            [
                'SEX', 'POS_OCU', 'SCIAN', 'EDA',
                'CS_P13_1', 'EMP_PPAL', 'MUN', 'tamaño_hogar'
            ]
        ]
    )

    enoe = enoe.assign(
        genero=enoe.SEX.map(
            {
                1: 'H',
                2: 'F'
            }
        ),
        ocupacion=enoe.POS_OCU.map(
            {
                1: 'trabajador',
                2: 'trabajador',
                3: 'independiente',
                4: 'otro'
            }
        ),
        edad_num=enoe.EDA.astype(int),
        edad_cat=pd.cut(
            enoe.EDA,
            (0, 3, 5, 6, 8, 12, 15, 18, 25, 50, 60, 65, 131),
            right=False
        ).astype(str),
        sector=enoe.SCIAN.map(
            {
                1: "Agricultura y ganadería",
                2: "Minería",
                3: "Otro",
                4: "Construcción",
                5: "Industria manufacturera",
                6: "Comercio",
                7: "Comercio",
                8: "Transporte y comunicaciones",
                9: "Otro",
                10: "Servicios",
                11: "Servicios",
                12: "Servicios",
                13: "Servicios",
                14: "Servicios",
                15: "Servicios",
                16: "Servicios",
                17: "Servicios",
                18: "Servicios",
                19: "Servicios",
                20: "Gobierno",
                21: "Otro",
            }
        ),
        escolaridad=enoe.CS_P13_1.map(
            {
                0: "Sin Instrucción",
                1: "Sin Instrucción",
                2: "Primaria o Secundaria",
                3: "Primaria o Secundaria",
                4: "Carrera técnica o preparatoria",
                5: "Carrera técnica o preparatoria",
                6: "Carrera técnica o preparatoria",
                7: "Licenciatura",
                8: "Postgrado",
                9: "Postgrado",
                # 99: "Otro",
            }
        ),
        informal=enoe.EMP_PPAL.map(
            {
                1: 1,
                2: 0,
            }
        ),
        municipio=enoe.MUN.map(
            {
                1: "abasolo",
                6: "apodaca",
                9: "cadereyta",
                12: "flores",
                10: "carmen",
                18: "garcia",
                21: "escobedo",
                25: "zuazua",
                26: "guadalupe",
                47: "hidalgo",
                31: "juarez",
                39: "monterrey",
                41: "pesqueria",
                45: "salinas",
                46: "san_nicolas",
                19: "san_pedro",
                48: "santa_catarina",
                49: "santiago",
            }
        ).fillna('otro')
    ).drop(
        columns=[
            'SEX', 'POS_OCU', 'SCIAN', 'CS_P13_1',
            'EMP_PPAL', 'MUN', 'EDA']
    ).dropna().reset_index(drop=True)

    return enoe
