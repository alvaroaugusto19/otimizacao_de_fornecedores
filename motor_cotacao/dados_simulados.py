"""Gerador de dados SIMULADOS (fornecedores, cotações e histórico de entregas).

Sem dados reais, a única forma honesta de testar o motor é simular — e dizer
que é simulado. O gerador tem uma "verdade escondida" (a confiabilidade real de
cada fornecedor) para podermos checar se o modelo de risco a recupera.
"""
from __future__ import annotations

import math
import random

from .catalogo import CATALOGO
from .modelos import Cotacao, Entrega, Fornecedor, ItemCotado

# (id, nome, frete, pedido mínimo, km, propensão a atraso [logit])
_FORNECEDORES = [
    ("F1", "Depósito Planalto",        180, 0,    8,  -2.2),
    ("F2", "Casa do Construtor Norte", 250, 1500, 15, -1.6),
    ("F3", "Atacadão de Materiais DF", 450, 5000, 32, -0.6),
    ("F4", "Areial Rio Preto",         320, 800,  45, -1.0),
    ("F5", "Cerâmica Goiás",           600, 3000, 120, 0.2),
    ("F6", "Aço & Ferro Centro-Oeste", 280, 1000, 22, -1.4),
]

# nível de preço por categoria (1,00 = referência). Ausente = não vende.
# Especialistas são mais baratos no que fazem; o depósito do bairro é caro mas confiável.
_NIVEL = {
    "F1": {"cimento": 1.03, "agregado": 1.05, "alvenaria": 1.04, "aco": 1.05},
    "F2": {"cimento": 0.97, "alvenaria": 1.00, "aco": 0.99},
    "F3": {"cimento": 0.91, "agregado": 1.00, "alvenaria": 0.97, "aco": 0.98},
    "F4": {"agregado": 0.88},
    "F5": {"alvenaria": 0.85},
    "F6": {"aco": 0.90, "cimento": 1.00},
}
_MIX = {k: set(v) for k, v in _NIVEL.items()}


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def gerar(seed: int = 7):
    rnd = random.Random(seed)
    fornecedores = [Fornecedor(i, n, fr, pm, km) for i, n, fr, pm, km, _ in _FORNECEDORES]
    propensao = {f[0]: f[5] for f in _FORNECEDORES}

    # --- cotações atuais ---
    cotacoes: dict[str, Cotacao] = {}
    for f in fornecedores:
        itens = []
        for ins in CATALOGO:
            if ins.categoria not in _MIX[f.id] or rnd.random() < 0.10:
                continue
            preco = ins.preco_referencia * _NIVEL[f.id][ins.categoria] * rnd.gauss(1, 0.03)
            prazo = max(1, int(round(f.distancia_km / 30 + rnd.randint(1, 4))))
            disp = None
            if rnd.random() < 0.25:  # às vezes o estoque do fornecedor é limitado
                disp = float(rnd.choice([50, 100, 200, 3000, 8]))
            itens.append(ItemCotado(ins.codigo, round(preco, 2), prazo, disp))
        cotacoes[f.id] = Cotacao(f.id, itens, origem="simulado")

    # anomalia proposital: alguém cotou cimento "por kg" e foi digitado como "por saco"
    cotacoes["F2"].itens = [it for it in cotacoes["F2"].itens if it.insumo != "CIM-50"]
    cotacoes["F2"].itens.append(ItemCotado("CIM-50", 0.74, 2))

    # --- histórico de entregas (alimenta o ML) ---
    historico: list[Entrega] = []
    for _ in range(600):
        f = rnd.choice(fornecedores)
        cat = rnd.choice(sorted(_MIX[f.id]))
        valor = math.exp(rnd.uniform(math.log(300), math.log(20000)))
        prazo = rnd.randint(1, 10)
        mes = rnd.randint(1, 12)
        chuva = 1.0 if mes in (11, 12, 1, 2, 3) else 0.0      # período chuvoso no DF
        logit = (propensao[f.id] + 0.004 * f.distancia_km + 0.6 * chuva
                 - 0.12 * prazo + 0.15 * math.log(valor / 1000) + (0.4 if cat == "agregado" else 0))
        historico.append(Entrega(f.id, cat, round(valor, 2), prazo, f.distancia_km, mes,
                                 rnd.random() < _sig(logit)))
    return fornecedores, cotacoes, historico
