"""Catálogo de insumos e casamento de descrições.

Cada fornecedor escreve o mesmo material de um jeito:
"CIMENTO CP-II 50KG VOTORAN", "cimento portland saco", "Cim. CPII-E32".
Antes de otimizar é preciso saber que tudo isso é o insumo CIM-50.

Estratégia em camadas (da mais barata/confiável para a mais cara):
  1. similaridade de texto (rapidfuzz) contra descrição + sinônimos
  2. se a confiança ficar baixa -> NÃO adivinha: vai para revisão humana
     (o LLM pode sugerir, mas quem confirma é uma pessoa)
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

from .modelos import Insumo

# Preços de referência SIMULADOS, na ordem de grandeza de 2026.
# Num sistema real viriam do SINAPI (Caixa/IBGE) do estado da obra.
CATALOGO: list[Insumo] = [
    Insumo("CIM-50", "Cimento Portland CP II 50 kg", "saco", "cimento", 38.0,
           ("cimento cp ii", "cimento cp2", "cimento portland", "cimento saco 50kg", "cp ii e 32", "cp ii f 32"),
           {"kg": 1 / 50, "t": 20.0}),
    Insumo("ARE-M", "Areia média lavada", "m3", "agregado", 130.0,
           ("areia media", "areia lavada", "areia grossa", "areia de rio"),
           {"t": 1 / 1.5}),   # ~1,5 t por m³
    Insumo("BRI-1", "Brita 1", "m3", "agregado", 140.0,
           ("pedra britada 1", "brita n 1", "brita 19mm", "pedra 1"),
           {"t": 1 / 1.4}),
    Insumo("TIJ-9F", "Tijolo cerâmico 9 furos 9x19x19", "un", "alvenaria", 1.10,
           ("tijolo 9 furos", "bloco ceramico 9 furos", "tijolo baiano", "tijolo ceramico"),
           {"milheiro": 1000.0}),
    Insumo("BLC-14", "Bloco de concreto 14x19x39", "un", "alvenaria", 4.20,
           ("bloco concreto 14", "bloco estrutural 14x19x39", "bloco de cimento 14"),
           {"milheiro": 1000.0}),
    Insumo("VER-10", "Vergalhão CA-50 10 mm barra 12 m", "barra", "aco", 55.0,
           ("vergalhao 10mm", "ferro 3/8", "ca 50 10mm", "barra de aco 10mm"),
           {"kg": 1 / 7.4}),  # barra de 12 m de 10 mm pesa ~7,4 kg
    Insumo("CAL-20", "Cal hidratada CH-III 20 kg", "saco", "cimento", 18.0,
           ("cal hidratada", "cal ch iii", "cal para reboco"),
           {"kg": 1 / 20}),
    Insumo("ARG-AC3", "Argamassa colante AC-III 20 kg", "saco", "cimento", 32.0,
           ("argamassa ac iii", "argamassa colante", "ac3", "cimentcola"),
           {"kg": 1 / 20}),
]

POR_CODIGO = {i.codigo: i for i in CATALOGO}

LIMIAR_CONFIANCA = 75  # abaixo disso o casamento vai para revisão humana


def _normalizar(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _indice() -> dict[str, str]:
    """texto normalizado -> código do insumo"""
    idx = {}
    for ins in CATALOGO:
        idx[_normalizar(ins.descricao)] = ins.codigo
        for s in ins.sinonimos:
            idx[_normalizar(s)] = ins.codigo
    return idx


_INDICE = _indice()


def casar_descricao(descricao: str) -> tuple[str | None, float]:
    """Devolve (código do insumo, confiança 0-100). Código None = mandar para revisão."""
    alvo = _normalizar(descricao)
    melhor = process.extractOne(alvo, list(_INDICE.keys()), scorer=fuzz.token_set_ratio)
    if melhor is None:
        return None, 0.0
    texto, score, _ = melhor
    return (_INDICE[texto] if score >= LIMIAR_CONFIANCA else None), float(score)
