"""Camada de ML: risco de atraso e preços anômalos.

1) RISCO DE ATRASO
   Preço não é tudo: um fornecedor que atrasa muito "custa" mais do que cota.
   Treinamos uma regressão logística no histórico de entregas para estimar
   P(atraso | fornecedor, distância, prazo, valor, chuva, categoria).
   Essa probabilidade vira uma penalidade no objetivo do otimizador.

   Por que regressão logística e não algo "mais forte"?
   - pouco dado (centenas de entregas), e ela regulariza bem;
   - é interpretável: dá pra dizer à banca/ao comprador POR QUE o risco é alto;
   - sai probabilidade calibrada, que é exatamente o que entra num custo esperado.

2) PREÇO ANÔMALO
   Compara cada preço com a referência e com os outros fornecedores (mediana/MAD,
   que não se deixam levar por um valor absurdo). Se a razão preço/referência
   bate com um fator de unidade (ex.: 1/50), provavelmente é erro de unidade.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .catalogo import POR_CODIGO
from .modelos import Cotacao, Entrega

MESES_CHUVA = {11, 12, 1, 2, 3}


def _linhas(entregas: list[Entrega]) -> list[list]:
    return [[e.fornecedor, e.categoria, e.distancia_km, e.prazo_prometido,
             math.log(max(e.valor, 1) / 1000), 1.0 if e.mes in MESES_CHUVA else 0.0] for e in entregas]


@dataclass
class MetricasRisco:
    n_treino: int
    n_teste: int
    auc: float            # 0,5 = chute; 1,0 = perfeito
    brier: float          # erro quadrático das probabilidades (menor é melhor)
    brier_base: float     # Brier de quem chuta sempre a taxa média
    taxa_atraso: float


class ModeloRisco:
    def __init__(self):
        cat = [0, 1]
        num = [2, 3, 4, 5]
        self.pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                ("num", StandardScaler(), num),
            ])),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        self.metricas: MetricasRisco | None = None

    def treinar(self, historico: list[Entrega], seed: int = 0) -> MetricasRisco:
        X = np.array(_linhas(historico), dtype=object)
        y = np.array([e.atrasou for e in historico], dtype=int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        self.pipe.fit(Xtr, ytr)
        p = self.pipe.predict_proba(Xte)[:, 1]
        base = np.full_like(p, ytr.mean(), dtype=float)
        self.metricas = MetricasRisco(len(ytr), len(yte), float(roc_auc_score(yte, p)),
                                      float(brier_score_loss(yte, p)), float(brier_score_loss(yte, base)),
                                      float(y.mean()))
        self.pipe.fit(X, y)  # depois de medir, treina com tudo
        return self.metricas

    def prob_atraso(self, fornecedor: str, categoria: str, distancia_km: float,
                    prazo: int, valor: float, mes: int) -> float:
        e = Entrega(fornecedor, categoria, valor, prazo, distancia_km, mes, False)
        X = np.array(_linhas([e]), dtype=object)
        return float(self.pipe.predict_proba(X)[0, 1])


# ---------------------------------------------------------------- anomalias

@dataclass
class Anomalia:
    fornecedor: str
    insumo: str
    preco: float
    referencia: float
    razao: float
    motivo: str


def detectar_anomalias(cotacoes: dict[str, Cotacao], limite_desvio: float = 0.35) -> list[Anomalia]:
    por_insumo: dict[str, list[tuple[str, float]]] = {}
    for c in cotacoes.values():
        for it in c.itens:
            por_insumo.setdefault(it.insumo, []).append((c.fornecedor, it.preco_unitario))

    achados = []
    for cod, precos in por_insumo.items():
        ins = POR_CODIGO[cod]
        valores = [p for _, p in precos] + [ins.preco_referencia]
        med = statistics.median(valores)
        mad = statistics.median([abs(v - med) for v in valores]) or 1e-9
        for forn, p in precos:
            razao = p / ins.preco_referencia
            z = 0.6745 * (p - med) / mad          # z robusto
            # anômalo = longe da referência, OU destoa muito dos concorrentes (z robusto)
            # E ainda assim com desvio relevante (evita marcar diferenças de 5-10%)
            longe_ref = abs(razao - 1) > limite_desvio
            destoa = abs(z) > 3.5 and abs(p / med - 1) > 0.2
            if not (longe_ref or destoa):
                continue
            motivo = f"{(razao - 1) * 100:+.0f}% vs referência"
            for unid, fator in ins.fatores.items():   # erro de unidade?
                if fator < 1 and abs(razao / fator - 1) < 0.25:
                    motivo += f"; parece preço por {unid} lançado como por {ins.unidade}"
            achados.append(Anomalia(forn, cod, p, ins.preco_referencia, razao, motivo))
    return achados
