from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class LinearModel(Protocol):
    """Contrato do modelo treinado que o sistema usa.

    Depender do protocolo em vez da classe do scikit-learn mantém as funções de avaliação
    utilizáveis com qualquer estimador equivalente, incluindo o challenger que a promoção
    compara contra o campeão.

    Os coeficientes fazem parte do contrato porque o painel de Explainable AI decompõe o
    preço em intercepto mais a contribuição de cada feature. Trocar por um modelo não linear
    exigiria outra estratégia de explicação, e é por isso que o projeto assume linearidade
    num protocolo só em vez de fingir que qualquer estimador serve.
    """

    coef_: NDArray[np.float64]
    intercept_: float
    feature_names_in_: NDArray[np.str_]

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]: ...


# Nome anterior, mantido porque metade do código fala de "predizer" e a outra metade de
# "modelo linear" — são o mesmo contrato desde que o projeto passou a ter um modelo só.
Predictor = LinearModel
