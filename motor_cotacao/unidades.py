"""Normalização de unidades.

Problema real: um fornecedor cota cimento "R$ 0,74/kg", outro "R$ 37/saco",
outro "R$ 740/t". Sem converter para a mesma unidade, o otimizador compara
banana com laranja. Aqui tudo vira R$ por unidade BASE do insumo.
"""
from __future__ import annotations

import unicodedata

from .modelos import Insumo

# grafias diferentes -> nome canônico da unidade
_SINONIMOS_UNIDADE = {
    "saco": "saco", "sacos": "saco", "sc": "saco", "sco": "saco",
    "kg": "kg", "quilo": "kg", "quilos": "kg",
    "t": "t", "ton": "t", "tonelada": "t", "toneladas": "t",
    "m3": "m3", "m³": "m3", "metro cubico": "m3", "metros cubicos": "m3",
    "un": "un", "und": "un", "unid": "un", "unidade": "un", "unidades": "un",
    "pc": "un", "pç": "un", "peca": "un", "pecas": "un",
    "mil": "milheiro", "milheiro": "milheiro", "milh": "milheiro",
    "barra": "barra", "barras": "barra", "br": "barra",
}


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def canonizar(unidade: str) -> str | None:
    u = _sem_acento(unidade.strip().lower()).rstrip(".")
    return _SINONIMOS_UNIDADE.get(u) or _SINONIMOS_UNIDADE.get(unidade.strip().lower())


def fator_para_base(unidade: str, insumo: Insumo) -> float | None:
    """Quantas unidades base cabem em 1 `unidade`. None = conversão desconhecida."""
    u = canonizar(unidade)
    if u is None:
        return None
    if u == insumo.unidade:
        return 1.0
    return insumo.fatores.get(u)


def preco_para_base(preco: float, unidade: str, insumo: Insumo) -> float | None:
    """R$/unidade -> R$/unidade base. Ex.: R$ 0,74/kg de cimento -> R$ 37/saco."""
    f = fator_para_base(unidade, insumo)
    return None if f is None else preco / f


def quantidade_para_base(qtd: float, unidade: str, insumo: Insumo) -> float | None:
    f = fator_para_base(unidade, insumo)
    return None if f is None else qtd * f
