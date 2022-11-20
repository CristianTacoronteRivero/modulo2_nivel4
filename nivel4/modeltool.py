import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import pickle
from sklearn.decomposition import PCA


class SuperPipeline:
    def __init__(self, dataframe: pd.DataFrame, seed: int = None, indx_target=-1):
        self.dataframe = dataframe
        X = dataframe.drop(columns=dataframe.columns[indx_target])
        y = dataframe.iloc[:, indx_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def add_pipeline(self, proceso, add_to_pipeline=None, params=None, name=None):
        """Funcion encargada de crear pipelines o añadir steps a un pipeline ya creado

        :param proceso: Proceso que se quiere añadir al pipeline
        :type proceso: Clase de sklearn permitida en un pipeline
        :param add_to_pipeline: Si se especifica un pipeline de sklearn lo que hace es añadirlo a la cola, por defecto es None
        :type add_to_pipeline: Pipeline sklearn, optional
        :param params: Parametros de la clase de sklearn, por defecto es none
        :type params: dict, optional
        :param name: Nombre del pipeline sklearn
        :type name: str, optional
        :return: Pipeline sklearn
        :rtype: Pipeline sklearn

        >>> from nivel4.modeltool import SuperPipeline
        >>> import pandas as pd
        >>> df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",header=None)
        >>> sp = SuperPipeline(df)
        >>> pipe_num = sp.add_pipeline(StandardScaler())
        >>> pipe_num
        Pipeline(steps=[('standardscaler', StandardScaler())])
        >>> sp.add_pipeline(KNNImputer(), add_to_pipeline=pipe_num)
        >>> pipe_num
        Pipeline(steps=[('standardscaler', StandardScaler()),
                        ('knnimputer', KNNImputer())])
        """
        if add_to_pipeline:
            try:
                return add_to_pipeline.steps.append(
                    (
                        str(proceso).replace("()", "").lower(),
                        proceso.set_params(**params),
                    )
                )
            except TypeError:
                return add_to_pipeline.steps.append(
                    (str(proceso).replace("()", "").lower(), proceso)
                )
        else:
            try:
                return make_pipeline(proceso.set_params(**params))
            except TypeError:
                return make_pipeline(proceso)

    def build_estimator(
        self,
        name_pipelines: list,
        pipelines: list,
        cols: list,
        estimator,
        params_stimator: dict = None,
        X=None,
        y=None,
    ):
        """Funcion encargada de aplicar los pipelines en las columnas correspondientes y encadenarlo con un modelo final
        ya listo para hacer las predicciones

        :param name_pipelines: Nombres de los pipelines creados anteriormente
        :type name_pipelines: list
        :param pipelines: Lista de pipelines creados
        :type pipelines: list
        :param cols: Lista de columnas correspondientes a cada pipeline
        :type cols: list
        :param estimator: Estimador a usar en el pipeline
        :type estimator: Estimador sklearn
        :param params_stimator: Parametros del estimador, por defecto None
        :type params_stimator: dict, optional
        :param X: _description_, defaults to None
        :type X: _type_, optional
        :param y: _description_, defaults to None
        :type y: _type_, optional
        :return: _description_
        :rtype: _type

        >>> from nivel4.modeltool import SuperPipeline
        >>> import pandas as pd
        >>> df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",header=None)
        >>> sp = SuperPipeline(df)
        >>> pipe_num = sp.add_pipeline(StandardScaler())
        >>> sp.add_pipeline(KNNImputer(), add_to_pipeline=pipe_num)
        >>> pipe_cat = sp.add_pipeline(KNNImputer())
        >>> pipeline = sp.build_estimator(name_pipelines=["pipeline_num", "pipeline_cat"],\
            pipelines=[pipe_num, pipe_cat],\
            cols=[df.iloc[:, :-1].columns, df.iloc[:, :-1].columns],\
            estimator=RandomForestClassifier())
        """
        lista_trans = list()
        for n, p, c in zip(name_pipelines, pipelines, cols):
            lista_trans.append((n, p, c))

        ct = ColumnTransformer(lista_trans)
        try:
            pipeline = make_pipeline(ct, estimator.set_params(**params_stimator))
        except TypeError:
            pipeline = make_pipeline(ct, estimator)
        if X == None:
            X = self.X_train
        if y == None:
            y = self.y_train
        return pipeline.fit(X, y)

    def export_pipeline(self, name, pipeline, mode="wb"):
        with open(f"{name}.pkl", mode) as f:
            pickle.dump(pipeline, f)


if __name__ == "__main__":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
        header=None,
    )

    sp = SuperPipeline(df)
    pipe_num = sp.add_pipeline(StandardScaler())
    sp.add_pipeline(KNNImputer(), add_to_pipeline=pipe_num)

    pipe_cat = sp.add_pipeline(KNNImputer())

    pipeline = sp.build_estimator(
        name_pipelines=["pipeline_num", "pipeline_cat"],
        pipelines=[pipe_num, pipe_cat],
        cols=[df.iloc[:, :-1].columns, df.iloc[:, :-1].columns],
        estimator=RandomForestClassifier(),
    )

    sp.export_pipeline("model", pipeline)
