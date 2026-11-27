"""Explicação em português da recomendação.

Os números vêm SEMPRE do otimizador. Este texto é montado por template, então
não tem como "alucinar" um valor. Se quiser um texto mais natural, dá para pedir
a um LLM que REESCREVA este texto — nunca que recalcule.
"""
from __future__ import annotations

from .catalogo import POR_CODIGO
from .otimizador import Resultado, brl


def explicar(ot: Resultado, base: Resultado, cmp: dict, nomes: dict[str, str]) -> str:
    l = []
    l.append(f"Recomendação: comprar de {len(ot.fornecedores_usados)} fornecedor(es), "
             f"desembolso de {brl(ot.desembolso)} ({brl(ot.custo_material)} em material + {brl(ot.custo_frete)} em frete).")
    for j in ot.fornecedores_usados:
        itens = [a for a in ot.alocacoes if a.fornecedor == j]
        desc = "; ".join(f"{a.quantidade:g} {POR_CODIGO[a.insumo].unidade} de {POR_CODIGO[a.insumo].descricao} "
                         f"a {brl(a.preco_unitario)}" for a in itens)
        risco = max(a.prob_atraso for a in itens)
        l.append(f"  - {nomes[j]}: {desc} (risco de atraso estimado {risco:.0%}).")
    if cmp["economia_custo_total"] > 0.005:
        motivo = []
        if cmp["fretes_evitados"] > 0:
            motivo.append(f"evita {cmp['fretes_evitados']} frete(s)")
        if base.custo_risco > ot.custo_risco:
            motivo.append("troca fornecedores com mais risco de atraso")
        if base.avisos:
            motivo.append("respeita pedidos mínimos que a compra 'mais barato por item' violaria")
        l.append(f"Comparado a comprar o mais barato de cada item ({brl(base.desembolso)} de desembolso, "
                 f"{len(base.fornecedores_usados)} fornecedores), a recomendação economiza "
                 f"{brl(cmp['economia_custo_total'])} em custo total ({cmp['economia_percentual']:.1f}%)"
                 + (": " + ", ".join(motivo) if motivo else "") + ".")
    else:
        l.append("A compra 'mais barato por item' já era a melhor opção neste caso.")
    if ot.custo_risco > 0:
        l.append(f"O custo total inclui {brl(ot.custo_risco)} de custo ESPERADO de atraso (não é desembolso; "
                 "é o peso que o modelo de risco dá à chance de atrasar).")
    for a in ot.avisos:
        l.append(f"AVISO: {a}")
    for x in ot.itens_inviaveis:
        l.append(f"ATENÇÃO: {POR_CODIGO[x.insumo].descricao} ficou de fora: {x.motivo}.")
    return "\n".join(l)
