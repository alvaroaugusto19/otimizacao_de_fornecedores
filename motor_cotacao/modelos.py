"""Estruturas de dados do motor.

Tudo que é preço fica na UNIDADE BASE do insumo (a mesma do orçamento da obra).
A conversão de "R$/kg", "R$/milheiro" etc. acontece antes, em unidades.py.
Isso é o que permite comparar cotações de fornecedores diferentes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Insumo:
    codigo: str
    descricao: str
    unidade: str                      # unidade base (ex.: "saco", "m3", "un")
    categoria: str                    # cimento, agregado, alvenaria, aço...
    preco_referencia: float           # referência de mercado (tipo SINAPI) na unidade base
    sinonimos: tuple[str, ...] = ()
    # quantas unidades BASE existem em 1 unidade alternativa.
    # ex.: cimento (base = saco de 50 kg): {"kg": 1/50, "t": 20}
    fatores: dict[str, float] = field(default_factory=dict)


@dataclass
class Fornecedor:
    id: str
    nome: str
    frete_fixo: float                 # custo fixo por pedido (é o que torna a COMBINAÇÃO importante)
    pedido_minimo: float = 0.0        # valor mínimo em R$ para aceitar o pedido
    distancia_km: float = 0.0


@dataclass
class ItemCotado:
    insumo: str                       # código do insumo
    preco_unitario: float             # R$ por unidade base
    prazo_dias: int                   # dias até a entrega
    quantidade_disponivel: float | None = None   # None = sem limite de estoque


@dataclass
class Cotacao:
    fornecedor: str                   # id do fornecedor
    itens: list[ItemCotado]
    validade: date | None = None
    origem: str = "manual"            # manual | llm | regras


@dataclass
class ItemRequisicao:
    insumo: str
    quantidade: float                 # na unidade base
    prazo_max_dias: int               # precisa chegar em até X dias


@dataclass
class Requisicao:
    obra: str
    itens: list[ItemRequisicao]


@dataclass
class Entrega:
    """Uma entrega passada: é o histórico que alimenta o modelo de risco."""
    fornecedor: str
    categoria: str
    valor: float
    prazo_prometido: int
    distancia_km: float
    mes: int
    atrasou: bool
