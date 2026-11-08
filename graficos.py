"""Funções de gráfico e tabela compartilhadas pelos notebooks do projeto de Redes Neurais.

Convenções (alinhadas ao padrão visual do projeto):
- paleta sóbria: azul principal para categorias e séries primárias, azul-claro/coral para contrastes e cinza para referências;
- toda figura vem acompanhada da tabela com os mesmos números;
- decimais com vírgula nos rótulos ("55,3%", "244,6").
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from IPython.display import display

from traducao import rotulo

AZUL = "#2a78d6"
AZUL_CLARO = "#9cc3ee"
CORAL = "#d95f43"
CINZA_OUTROS = "#c3c2b7"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
GRADE = "#e6e5e2"

ALVO_PADRAO = "doenca_cardiaca"
POSITIVO_PADRAO = "Com doença (1)"


def configurar_estilo():
    """Aplica o estilo padrão do projeto ao matplotlib e ao pandas."""
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 160)
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": GRADE,
        "axes.linewidth": 1,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 9.5,
        "axes.labelcolor": TEXTO_SEC,
        "xtick.color": TEXTO_SEC,
        "ytick.color": TEXTO_SEC,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXTO,
        "legend.frameon": False,
    })


def mostrar(arquivo=None):
    """Mostra a figura atual e, se `arquivo` for informado, salva-a em PNG antes (150 dpi)."""
    if arquivo:
        plt.savefig(arquivo, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def formatar_int(n):
    return f"{int(n):,}".replace(",", ".")


def formatar_pct(p, casas=1):
    if 0 < p < 0.05:
        return "<0,1%"
    return f"{p:.{casas}f}%".replace(".", ",")


def formatar_num(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def eixo_compacto(ax, eixo="x"):
    fmt = FuncFormatter(lambda v, _: (f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}").replace(".", ","))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)


def eixo_percentual(ax, eixo="x"):
    fmt = FuncFormatter(lambda v, _: f"{v:g}%".replace(".", ","))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Distribuições univariadas
# ---------------------------------------------------------------------------
def tabela_frequencia(serie, ordem=None):
    """Contagem e percentual de cada classe, por frequência ou na ordem informada."""
    contagem = serie.value_counts(dropna=False)
    if ordem is not None:
        presentes = [c for c in ordem if c in contagem.index]
        restantes = [c for c in contagem.index if c not in presentes]
        contagem = contagem.reindex(presentes + restantes)
    tabela = pd.DataFrame({
        "Contagem": contagem,
        "Percentual (%)": (contagem / contagem.sum() * 100).round(2),
    })
    tabela.index.name = "Classe"
    return tabela


def grafico_barras(contagem, titulo, max_classes=20, rotular=True, xlabel="Contagem", total=None,
                   cores=None, arquivo=None):
    """Barras horizontais limpas com rótulos de contagem e percentual."""
    contagem = contagem.copy()
    if cores is None:
        cores = [AZUL] * len(contagem)
    if max_classes is not None and len(contagem) > max_classes:
        principais = contagem.nlargest(max_classes).index
        cauda = contagem.drop(principais)
        contagem = pd.concat([
            contagem[contagem.index.isin(principais)],
            pd.Series({f"Outros ({len(cauda)} classes)": cauda.sum()}),
        ])
        cores = [AZUL] * max_classes + [CINZA_OUTROS]

    total = total if total is not None else contagem.sum()
    posicoes = np.arange(len(contagem))[::-1]
    altura = max(2.2, 0.44 * len(contagem) + 1.1)

    fig, ax = plt.subplots(figsize=(8.8, altura))
    ax.barh(posicoes, contagem.values, color=cores, height=0.6)
    ax.set_yticks(posicoes, [str(c) for c in contagem.index])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if rotular:
        for pos, valor in zip(posicoes, contagem.values):
            ax.text(valor + contagem.max() * 0.015, pos,
                    f"{formatar_int(valor)}  ({formatar_pct(valor / total * 100)})",
                    va="center", fontsize=8.5, color=TEXTO_SEC)
    ax.set_xlim(0, contagem.max() * 1.26)
    eixo_compacto(ax, "x")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    mostrar(arquivo)


def distribuicao_categorica(df, coluna, ordem=None, max_classes=20, titulo=None):
    """Gráfico de barras + tabela (contagem e %) da distribuição das classes de uma coluna."""
    tabela = tabela_frequencia(df[coluna], ordem=ordem)
    titulo = titulo or f"{rotulo(coluna)}: distribuição das classes (n = {formatar_int(len(df))})"
    grafico_barras(tabela["Contagem"], titulo, max_classes=max_classes)
    display(tabela)
    return tabela


def resumo_estatistico_numerico(serie, nome=None):
    """Resumo descritivo enriquecido (média, DP, quartis, IQR, assimetria, curtose e outliers IQR)."""
    s = serie.dropna()
    q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    out_inf = int((s < q1 - 1.5 * iqr).sum())
    out_sup = int((s > q3 + 1.5 * iqr).sum())
    dados = {
        "n": int(len(s)),
        "média": round(float(s.mean()), 2),
        "desvio padrão": round(float(s.std(ddof=1)), 2),
        "mínimo": round(float(s.min()), 2),
        "Q1 (25%)": round(float(q1), 2),
        "mediana": round(float(med), 2),
        "Q3 (75%)": round(float(q3), 2),
        "máximo": round(float(s.max()), 2),
        "IQR": round(float(iqr), 2),
        "assimetria": round(float(stats.skew(s)), 2),
        "curtose": round(float(stats.kurtosis(s)), 2),
        "outliers IQR": out_inf + out_sup,
    }
    return pd.DataFrame([dados], index=[nome or serie.name])


def distribuicao_numerica(df, coluna, bins=15, titulo=None, filtro_positivos=False, casas_faixa=0):
    """Histograma + boxplot horizontal + resumo estatístico + tabela de contagem por faixa."""
    serie = df[coluna].dropna()
    if filtro_positivos:
        serie = serie[serie > 0]

    nome = rotulo(coluna) + (" (apenas valores > 0)" if filtro_positivos else "")
    resumo = resumo_estatistico_numerico(serie, nome=nome)

    contagens, bordas = np.histogram(serie, bins=bins)
    fmt_borda = (lambda x: formatar_int(round(x))) if casas_faixa == 0 else (lambda x: formatar_num(x, casas_faixa))
    faixas = [f"[{fmt_borda(a)}, {fmt_borda(b)})" for a, b in zip(bordas[:-1], bordas[1:])]
    faixas[-1] = faixas[-1][:-1] + "]"
    tabela = pd.DataFrame({
        "Contagem": contagens,
        "Percentual (%)": (contagens / contagens.sum() * 100).round(2),
    }, index=pd.Index(faixas, name="Faixa"))

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8.8, 4.3), sharex=True, gridspec_kw={"height_ratios": [1, 3.6]}, layout="constrained"
    )
    ax_box.boxplot(
        serie, vert=False, widths=0.55, patch_artist=True,
        boxprops={"facecolor": AZUL_CLARO, "edgecolor": AZUL, "linewidth": 1.2},
        medianprops={"color": CORAL, "linewidth": 2},
        whiskerprops={"color": AZUL, "linewidth": 1.1},
        capprops={"color": AZUL, "linewidth": 1.1},
        flierprops={"marker": "o", "markerfacecolor": CORAL, "markeredgecolor": "none", "markersize": 4.5, "alpha": 0.7},
    )
    ax_box.set_yticks([])
    ax_box.grid(axis="y", visible=False)
    ax_box.spines["bottom"].set_visible(False)
    ax_box.set_title(titulo or f"{nome}: distribuição e boxplot (n = {formatar_int(len(serie))})")

    ax_hist.bar(
        bordas[:-1], contagens, width=np.diff(bordas), align="edge",
        color=AZUL, edgecolor="white", linewidth=1.2
    )
    ax_hist.axvline(serie.mean(), color=TEXTO, linestyle="--", linewidth=1.2, label=f"Média: {formatar_num(serie.mean(), 1)}")
    ax_hist.axvline(serie.median(), color=CORAL, linestyle="-", linewidth=1.4, label=f"Mediana: {formatar_num(serie.median(), 1)}")
    ax_hist.grid(axis="x", visible=False)
    ax_hist.set_xlabel(rotulo(coluna))
    ax_hist.set_ylabel("Pacientes")
    ax_hist.legend(loc="upper right", fontsize=8.5)
    eixo_compacto(ax_hist, "x")
    eixo_compacto(ax_hist, "y")
    plt.show()

    display(resumo)
    display(tabela)
    return tabela


# ---------------------------------------------------------------------------
# Análise bivariada: atributo × doença cardíaca
# ---------------------------------------------------------------------------
def tabela_taxa(df, coluna, alvo=ALVO_PADRAO, positivo=POSITIVO_PADRAO, ordem=None):
    """Para cada classe de `coluna`: total, casos positivos, casos negativos e taxa de doença (%)."""
    cruzada = pd.crosstab(df[coluna], df[alvo])
    if positivo not in cruzada.columns:
        cruzada[positivo] = 0
    negativos_cols = [c for c in cruzada.columns if c != positivo]
    negativo_col = negativos_cols[0] if negativos_cols else "Sem doença (0)"
    if negativo_col not in cruzada.columns:
        cruzada[negativo_col] = 0

    tabela = pd.DataFrame({
        "Total": cruzada.sum(axis=1),
        negativo_col: cruzada[negativo_col],
        positivo: cruzada[positivo],
    })
    tabela["Taxa de doença (%)"] = (tabela[positivo] / tabela["Total"] * 100).round(2)
    if ordem is not None:
        presentes = [c for c in ordem if c in tabela.index]
        restantes = [c for c in tabela.index if c not in presentes]
        tabela = tabela.reindex(presentes + restantes)
    else:
        tabela = tabela.sort_values("Taxa de doença (%)", ascending=False)
    tabela.index.name = rotulo(coluna) if coluna in df.columns else coluna
    return tabela


def grafico_taxa(tabela, titulo, referencia=None, xlabel="Pacientes com doença cardíaca (%)",
                 max_classes=25):
    """Barras horizontais da taxa de positivos por classe, com rótulo 'taxa (n total)'."""
    tabela = tabela.copy()
    cores = [CORAL if t >= (referencia or 55.34) else AZUL for t in tabela["Taxa de doença (%)"].values]
    if max_classes is not None and len(tabela) > max_classes:
        tabela = tabela.iloc[:max_classes]
        cores = cores[:max_classes]
    taxa = tabela["Taxa de doença (%)"].values
    posicoes = np.arange(len(tabela))[::-1]
    altura = max(2.3, 0.46 * len(tabela) + 1.1)

    fig, ax = plt.subplots(figsize=(8.8, altura))
    ax.barh(posicoes, taxa, color=cores, height=0.6)
    ax.set_yticks(posicoes, [str(c) for c in tabela.index])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for pos, t, n in zip(posicoes, taxa, tabela["Total"].values):
        ax.text(t + 1.2, pos, f"{formatar_pct(t)}  (n = {formatar_int(n)})",
                va="center", fontsize=8.5, color=TEXTO_SEC)
    if referencia is not None:
        ax.axvline(referencia, color=TEXTO_SEC, linestyle=":", linewidth=1.2)
        ax.text(referencia, posicoes.max() + 0.65, f"taxa geral {formatar_pct(referencia)}",
                ha="center", va="bottom", fontsize=8.5, color=TEXTO_SEC)
    ax.set_xlim(0, 112)
    eixo_percentual(ax, "x")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def taxa_por_categoria(df, coluna, ordem=None, alvo=ALVO_PADRAO, positivo=POSITIVO_PADRAO, titulo=None):
    """Gráfico da taxa de doença cardíaca por classe + tabela (total, sem doença, com doença, taxa)."""
    tabela = tabela_taxa(df, coluna, alvo=alvo, positivo=positivo, ordem=ordem)
    referencia = (df[alvo] == positivo).mean() * 100
    titulo = titulo or f"{rotulo(coluna)}: prevalência de doença cardíaca por classe"
    grafico_taxa(tabela, titulo, referencia=referencia)
    display(tabela)
    return tabela


def taxa_por_faixa(df, coluna, bins, alvo=ALVO_PADRAO, positivo=POSITIVO_PADRAO, titulo=None,
                   filtro_positivos=False, casas_faixa=0):
    """Taxa de doença cardíaca por faixa de uma variável numérica (faixas [a, b), última fechada)."""
    sub = df[df[coluna] > 0].copy() if filtro_positivos else df.copy()
    bordas = np.asarray(list(bins), dtype=float)
    serie = sub[coluna]
    indices = np.clip(np.searchsorted(bordas, serie, side="right") - 1, 0, len(bordas) - 2)
    indices = pd.Series(indices, index=sub.index)
    dentro = (serie >= bordas[0]) & (serie <= bordas[-1])

    fmt_borda = (lambda x: formatar_int(round(x))) if casas_faixa == 0 else (lambda x: formatar_num(x, casas_faixa))
    faixas = [f"[{fmt_borda(a)}, {fmt_borda(b)})" for a, b in zip(bordas[:-1], bordas[1:])]
    faixas[-1] = faixas[-1][:-1] + "]"

    positivos = (sub[alvo] == positivo)
    total = indices[dentro].value_counts().reindex(range(len(faixas)), fill_value=0)
    pos = indices[dentro & positivos].value_counts().reindex(range(len(faixas)), fill_value=0)
    neg = total - pos
    tabela = pd.DataFrame({
        "Total": total.values,
        "Sem doença (0)": neg.values,
        positivo: pos.values,
    }, index=pd.Index(faixas, name="Faixa"))
    tabela["Taxa de doença (%)"] = (tabela[positivo] / tabela["Total"].replace(0, np.nan) * 100).round(2)

    referencia = positivos.mean() * 100
    taxas_vals = tabela["Taxa de doença (%)"].fillna(0).values
    cores = [CORAL if t >= referencia else AZUL for t in taxas_vals]

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(faixas))
    barras = ax.bar(x, taxas_vals, width=0.68, color=cores, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x, faixas, rotation=15 if len(faixas) > 6 else 0)
    ax.axhline(referencia, color=TEXTO_SEC, linestyle=":", linewidth=1.2)
    ax.text(len(faixas) - 0.5, referencia + 2, f"taxa geral {formatar_pct(referencia)}",
            va="bottom", ha="right", fontsize=8.5, color=TEXTO_SEC)
    for b, t, n_tot in zip(barras, taxas_vals, tabela["Total"].values):
        ax.text(b.get_x() + b.get_width() / 2, t + 1.8, f"{formatar_pct(t)}\n(n={n_tot})",
                ha="center", va="bottom", fontsize=8, color=TEXTO_SEC)
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    sufixo = " (apenas valores > 0)" if filtro_positivos else ""
    ax.set_title(titulo or f"{rotulo(coluna)}{sufixo}: prevalência de doença cardíaca por faixa")
    ax.set_xlabel(rotulo(coluna))
    ax.set_ylabel("Doença cardíaca (%)")
    eixo_percentual(ax, "y")
    plt.tight_layout()
    plt.show()
    display(tabela)
    return tabela


def comparar_grupos_numerico(df, coluna, grupo=ALVO_PADRAO, ordem=None, filtro_positivos=False, titulo=None):
    """Compara distribuições de `coluna` entre classes do alvo com boxplot + resumo por grupo."""
    sub = df[df[coluna] > 0].copy() if filtro_positivos else df.copy()
    if ordem is None:
        ordem = ["Sem doença (0)", "Com doença (1)"]
    ordem = [c for c in ordem if c in sub[grupo].unique()]

    agrupado = sub.groupby(grupo, observed=True)[coluna]
    tabela = agrupado.agg(
        n="count",
        Média="mean",
        DP="std",
        Mínimo="min",
        Q1=lambda s: s.quantile(0.25),
        Mediana="median",
        Q3=lambda s: s.quantile(0.75),
        Máximo="max",
    ).reindex(ordem).round(2)
    tabela.index.name = rotulo(grupo)

    dados_grupos = [sub.loc[sub[grupo] == c, coluna].dropna().values for c in ordem]
    cores_box = [AZUL_CLARO, CORAL] if len(ordem) == 2 else [AZUL_CLARO] * len(ordem)

    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    bp = ax_box.boxplot(
        dados_grupos, tick_labels=ordem, vert=True, widths=0.45, patch_artist=True,
        medianprops={"color": TEXTO, "linewidth": 1.8},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5, "markerfacecolor": TEXTO_SEC, "markeredgecolor": "none"},
    )
    for patch, cor in zip(bp["boxes"], cores_box):
        patch.set_facecolor(cor)
        patch.set_edgecolor(TEXTO_SEC)
        patch.set_alpha(0.85)
    ax_box.grid(axis="x", visible=False)
    sufixo = " (valores > 0)" if filtro_positivos else ""
    ax_box.set_title(titulo or f"{rotulo(coluna)}{sufixo}: distribuição por grupo")
    ax_box.set_ylabel(rotulo(coluna))

    x = np.arange(len(ordem))
    medias = tabela["Média"].values
    medianas = tabela["Mediana"].values
    largura = 0.34
    ax_bar.bar(x - largura / 2, medias, width=largura, color=AZUL, label="Média")
    ax_bar.bar(x + largura / 2, medianas, width=largura, color=CORAL, label="Mediana")
    ax_bar.set_xticks(x, ordem)
    ax_bar.grid(axis="x", visible=False)
    for xi, m, md in zip(x, medias, medianas):
        ax_bar.text(xi - largura / 2, m + max(medias.max(), 1) * 0.02, formatar_num(m, 1),
                    ha="center", va="bottom", fontsize=8.5, color=TEXTO_SEC)
        ax_bar.text(xi + largura / 2, md + max(medias.max(), 1) * 0.02, formatar_num(md, 1),
                    ha="center", va="bottom", fontsize=8.5, color=TEXTO_SEC)
    ax_bar.set_ylim(0, max(medias.max(), medianas.max()) * 1.22)
    ax_bar.set_title("Média vs. Mediana por grupo")
    ax_bar.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.show()
    display(tabela)
    return tabela
