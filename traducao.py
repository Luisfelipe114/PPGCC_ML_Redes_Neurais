"""
Tradução e dicionário de dados do dataset *Heart Failure Prediction* (UCI / Kaggle) para português.

O dataset reúne 918 registros de quatro coortes clínicas clássicas do repositório UCI
(Cleveland, Hungarian, Switzerland e Long Beach VA), com 11 atributos clínicos/demográficos
e a variável-alvo binária ``HeartDisease``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Colunas: nome original -> identificador em português (snake_case, sem acento)
# e rótulo legível (para títulos, tabelas e gráficos).
# ---------------------------------------------------------------------------
COLUNAS: dict[str, str] = {
    "Age": "idade",
    "Sex": "sexo",
    "ChestPainType": "tipo_dor_peito",
    "RestingBP": "pressao_repouso",
    "Cholesterol": "colesterol",
    "FastingBS": "glicemia_jejum",
    "RestingECG": "ecg_repouso",
    "MaxHR": "freq_cardiaca_max",
    "ExerciseAngina": "angina_exercicio",
    "Oldpeak": "depressao_st",
    "ST_Slope": "inclinacao_st",
    "HeartDisease": "doenca_cardiaca",
}

ROTULOS: dict[str, str] = {
    "Age": "Idade (anos)",
    "Sex": "Sexo",
    "ChestPainType": "Tipo de dor no peito",
    "RestingBP": "Pressão arterial em repouso (mmHg)",
    "Cholesterol": "Colesterol sérico (mg/dL)",
    "FastingBS": "Glicemia em jejum",
    "RestingECG": "ECG em repouso",
    "MaxHR": "Frequência cardíaca máxima (bpm)",
    "ExerciseAngina": "Angina induzida por exercício",
    "Oldpeak": "Depressão do segmento ST (Oldpeak, mV)",
    "ST_Slope": "Inclinação do segmento ST no pico",
    "HeartDisease": "Doença cardíaca",
}

TIPOS: dict[str, str] = {
    "Age": "numérica (contínua)",
    "Sex": "categórica (binária)",
    "ChestPainType": "categórica (nominal)",
    "RestingBP": "numérica (contínua)",
    "Cholesterol": "numérica (contínua)",
    "FastingBS": "categórica (binária)",
    "RestingECG": "categórica (nominal)",
    "MaxHR": "numérica (contínua)",
    "ExerciseAngina": "categórica (binária)",
    "Oldpeak": "numérica (contínua)",
    "ST_Slope": "categórica (ordinal)",
    "HeartDisease": "alvo (binária)",
}

UNIDADES: dict[str, str] = {
    "Age": "anos",
    "Sex": "—",
    "ChestPainType": "—",
    "RestingBP": "mmHg",
    "Cholesterol": "mg/dL",
    "FastingBS": "> 120 mg/dL (binário)",
    "RestingECG": "—",
    "MaxHR": "bpm",
    "ExerciseAngina": "—",
    "Oldpeak": "mV (ou mm)",
    "ST_Slope": "—",
    "HeartDisease": "0 / 1",
}

DESCRICOES: dict[str, str] = {
    "Age": "Idade do paciente no momento do exame, em anos.",
    "Sex": "Sexo biológico do paciente (M = Masculino, F = Feminino).",
    "ChestPainType": (
        "Classificação clínica da dor torácica: ASY (Assintomática), NAP (Dor não anginosa), "
        "ATA (Angina atípica) ou TA (Angina típica)."
    ),
    "RestingBP": "Pressão arterial sistólica em repouso na admissão hospitalar (mmHg). Contém 1 registro igual a 0.",
    "Cholesterol": (
        "Colesterol sérico total (mg/dL). Contém 172 registros iguais a 0 (ausentes codificados como zero "
        "nas coortes da Suíça e de Long Beach VA)."
    ),
    "FastingBS": "Glicemia em jejum: 1 se > 120 mg/dL (indicativo de hiperglicemia/diabetes), 0 caso contrário (≤ 120 mg/dL).",
    "RestingECG": (
        "Resultado do eletrocardiograma em repouso: Normal, ST (anomalia de onda ST-T: inversão de onda T "
        "e/ou elevação/depressão de ST > 0,05 mV) ou LVH (hipertrofia ventricular esquerda provável ou definitiva "
        "pelos critérios de Estes)."
    ),
    "MaxHR": "Frequência cardíaca máxima alcançada durante o teste ergométrico (entre 60 e 202 bpm).",
    "ExerciseAngina": "Presença de angina induzida pelo exercício durante o teste ergométrico (Y = Sim, N = Não).",
    "Oldpeak": (
        "Depressão do segmento ST induzida pelo exercício em relação ao repouso (mV). "
        "Valores negativos (13 casos) indicam elevação do segmento ST."
    ),
    "ST_Slope": "Inclinação do segmento ST no pico do exercício: Up (Ascendente), Flat (Plana) ou Down (Descendente).",
    "HeartDisease": "Variável-alvo binária: 1 = Presença de doença cardíaca (estenose > 50% em ao menos um vaso principal), 0 = Normal.",
}

# ---------------------------------------------------------------------------
# Valores categóricos: coluna original -> {valor original: tradução}
# ---------------------------------------------------------------------------
VALORES: dict[str, dict[Any, str]] = {
    "Sex": {
        "M": "Masculino",
        "F": "Feminino",
    },
    "ChestPainType": {
        "ASY": "Assintomática (ASY)",
        "NAP": "Dor não anginosa (NAP)",
        "ATA": "Angina atípica (ATA)",
        "TA": "Angina típica (TA)",
    },
    "FastingBS": {
        0: "≤ 120 mg/dL (0)",
        1: "> 120 mg/dL (1)",
        "0": "≤ 120 mg/dL (0)",
        "1": "> 120 mg/dL (1)",
    },
    "RestingECG": {
        "Normal": "Normal",
        "LVH": "Hipertrofia ventricular esq. (LVH)",
        "ST": "Anomalia de ST-T (ST)",
    },
    "ExerciseAngina": {
        "N": "Não (N)",
        "Y": "Sim (Y)",
    },
    "ST_Slope": {
        "Up": "Ascendente (Up)",
        "Flat": "Plana (Flat)",
        "Down": "Descendente (Down)",
    },
    "HeartDisease": {
        0: "Sem doença (0)",
        1: "Com doença (1)",
        "0": "Sem doença (0)",
        "1": "Com doença (1)",
    },
}

ORDEM_VALORES: dict[str, list[Any]] = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ASY", "NAP", "ATA", "TA"],
    "FastingBS": [0, 1],
    "RestingECG": ["Normal", "LVH", "ST"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"],
    "HeartDisease": [0, 1],
}

_COLUNAS_INV: dict[str, str] = {pt: en for en, pt in COLUNAS.items()}


def coluna_original(nome: str) -> str | None:
    """Devolve o nome original (inglês) da coluna, aceitando o nome em inglês ou em português."""
    nome = str(nome).strip()
    if nome in COLUNAS:
        return nome
    return _COLUNAS_INV.get(nome)


def traduzir_coluna(nome: str) -> str:
    """Traduz o nome de uma coluna para o identificador em português."""
    original = coluna_original(nome)
    return COLUNAS[original] if original else nome


def rotulo(nome: str) -> str:
    """Rótulo legível da coluna (ex.: 'RestingBP' ou 'pressao_repouso' -> 'Pressão arterial em repouso (mmHg)')."""
    original = coluna_original(nome)
    return ROTULOS[original] if original else str(nome)


def traduzir_valor(coluna: str, valor: Any) -> Any:
    """Traduz um valor categórico da coluna indicada."""
    original = coluna_original(coluna)
    if original is None or original not in VALORES:
        return valor
    return VALORES[original].get(valor, valor)


def ordem_classes(coluna: str, traduzido: bool = True) -> list[str] | None:
    """Ordem lógica das classes de uma coluna categórica/ordinal, ou None."""
    original = coluna_original(coluna)
    if original is None or original not in ORDEM_VALORES:
        return None
    ordem = ORDEM_VALORES[original]
    return [VALORES[original][v] for v in ordem] if traduzido else list(ordem)


def traduzir_dataframe(
    df,
    colunas: bool = True,
    valores: bool = True,
    copiar: bool = True,
):
    """
    Traduz um DataFrame do Heart Failure Prediction (nomes de colunas e/ou valores categóricos).
    """
    if copiar:
        df = df.copy()

    if valores:
        for col in df.columns:
            original = coluna_original(col)
            if original is None or original not in VALORES:
                continue
            mapa = VALORES[original]
            df[col] = df[col].map(lambda v: mapa.get(v, v))

    if colunas:
        df = df.rename(columns={c: traduzir_coluna(c) for c in df.columns})

    return df


def dicionario_dados():
    """Tabela (DataFrame) com coluna original, identificador em português, rótulo, tipo, unidade e descrição."""
    import pandas as pd

    linhas = [
        {
            "Coluna original": en,
            "Coluna (pt)": COLUNAS[en],
            "Rótulo": ROTULOS[en],
            "Tipo": TIPOS[en],
            "Unidade": UNIDADES[en],
            "Nº de classes": len({v for v in VALORES[en].values()}) if en in VALORES else None,
            "Descrição clínica": DESCRICOES[en],
        }
        for en in COLUNAS
    ]
    tabela = pd.DataFrame(linhas)
    tabela["Nº de classes"] = tabela["Nº de classes"].astype("Int64")
    return tabela


__all__ = [
    "COLUNAS",
    "ROTULOS",
    "TIPOS",
    "UNIDADES",
    "DESCRICOES",
    "VALORES",
    "ORDEM_VALORES",
    "coluna_original",
    "traduzir_coluna",
    "rotulo",
    "traduzir_valor",
    "ordem_classes",
    "traduzir_dataframe",
    "dicionario_dados",
]

