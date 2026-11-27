"""Servidor MCP: expõe o motor como FERRAMENTAS que um agente de IA pode chamar.

O MCP não tem inteligência própria. Ele só padroniza "quais funções existem,
que parâmetros recebem e o que devolvem", para que o Claude (ou o copiloto do
app) consiga, sozinho, ouvir "preciso de 200 sacos de cimento pra sexta" e
chamar listar_insumos -> otimizar_compra -> responder com a explicação.

Rodar:      python -m motor_cotacao.servidor_mcp        (transporte stdio)
Inspecionar: npx @modelcontextprotocol/inspector python -m motor_cotacao.servidor_mcp
"""
from __future__ import annotations

try:                                    # SDK do MCP 2.x
    from mcp.server.mcpserver import MCPServer as _Servidor
except ImportError:                     # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Servidor

from .catalogo import CATALOGO, POR_CODIGO
from .explicacao import explicar
from .motor import MotorCotacao

mcp = _Servidor("motor-cotacao")
motor = MotorCotacao.simulado()   # dados SIMULADOS; troque por dados reais aqui


def _res(r):
    return {
        "fornecedores": r.fornecedores_usados,
        "alocacoes": [{"fornecedor": a.fornecedor, "insumo": a.insumo, "quantidade": a.quantidade,
                       "preco_unitario": a.preco_unitario, "subtotal": round(a.subtotal, 2),
                       "prazo_dias": a.prazo_dias, "prob_atraso": round(a.prob_atraso, 3)} for a in r.alocacoes],
        "custo_material": r.custo_material, "custo_frete": r.custo_frete,
        "custo_risco_esperado": r.custo_risco, "desembolso": round(r.desembolso, 2),
        "itens_inviaveis": [{"insumo": x.insumo, "motivo": x.motivo} for x in r.itens_inviaveis],
        "avisos": r.avisos,
    }


@mcp.tool()
def listar_insumos() -> list[dict]:
    """Lista os insumos do catálogo (código, descrição, unidade base, preço de referência).
    Use o código ao pedir uma otimização."""
    return [{"codigo": i.codigo, "descricao": i.descricao, "unidade": i.unidade,
             "categoria": i.categoria, "preco_referencia": i.preco_referencia} for i in CATALOGO]


@mcp.tool()
def listar_fornecedores() -> list[dict]:
    """Lista fornecedores com frete fixo, pedido mínimo, distância e quantos itens cotaram."""
    return [{"id": f.id, "nome": f.nome, "frete_fixo": f.frete_fixo, "pedido_minimo": f.pedido_minimo,
             "distancia_km": f.distancia_km, "itens_cotados": len(motor.cotacoes.get(f.id).itens if f.id in motor.cotacoes else [])}
            for f in motor.fornecedores.values()]


@mcp.tool()
def precos_do_insumo(codigo: str) -> dict:
    """Mostra todas as cotações de um insumo, a referência de mercado e anomalias encontradas."""
    ins = POR_CODIGO[codigo]
    cot = [{"fornecedor": fid, "preco": it.preco_unitario, "prazo_dias": it.prazo_dias,
            "disponivel": it.quantidade_disponivel}
           for fid, c in motor.cotacoes.items() for it in c.itens if it.insumo == codigo]
    anom = [a.__dict__ for a in motor.anomalias() if a.insumo == codigo]
    return {"insumo": ins.descricao, "unidade": ins.unidade, "referencia": ins.preco_referencia,
            "cotacoes": sorted(cot, key=lambda x: x["preco"]), "anomalias": anom}


@mcp.tool()
def registrar_cotacao_texto(fornecedor_id: str, texto: str) -> dict:
    """Registra uma cotação a partir de texto livre (e-mail, WhatsApp, PDF convertido).
    Itens que não passarem na validação voltam como pendências e NÃO são usados."""
    r = motor.registrar_texto(fornecedor_id, texto)
    return {"provedor": r.provedor, "frete": r.frete, "pedido_minimo": r.pedido_minimo,
            "itens_registrados": [it.__dict__ for it in r.cotacao.itens],
            "confianca_casamento": r.confiancas,
            "pendencias": [p.__dict__ for p in r.pendencias]}


@mcp.tool()
def risco_fornecedores(valor_pedido: float = 3000, prazo_dias: int = 5) -> dict:
    """Probabilidade estimada de atraso por fornecedor (modelo treinado no histórico) e métricas do modelo."""
    out = {}
    for fid in motor.fornecedores:
        itens = motor.cotacoes.get(fid)
        if itens and itens.itens:
            out[fid] = round(motor.prob_atraso(fid, itens.itens[0].insumo, valor_pedido, prazo_dias), 3)
    return {"prob_atraso": out, "metricas_modelo": motor.metricas.__dict__ if motor.metricas else None}


@mcp.tool()
def otimizar_compra(itens: list[dict], permitir_divisao: bool = True,
                    max_fornecedores: int | None = None, peso_risco: float = 0.15) -> dict:
    """Escolhe a combinação de fornecedores de menor custo total (material + frete + risco de atraso).

    itens: lista de {"insumo": código, "quantidade": número na unidade base, "prazo_max_dias": inteiro}
    permitir_divisao: se um mesmo item pode ser comprado de mais de um fornecedor
    max_fornecedores: limite de fornecedores no pedido (None = sem limite)
    peso_risco: fração do valor perdida se atrasar (0 = ignora risco)
    """
    tup = [(i["insumo"], float(i["quantidade"]), int(i["prazo_max_dias"])) for i in itens]
    try:
        ot, base, cmp = motor.cotar(tup, peso_risco=peso_risco, permitir_divisao=permitir_divisao,
                                    max_fornecedores=max_fornecedores)
    except ValueError as e:
        return {"erro": str(e)}
    nomes = {f.id: f.nome for f in motor.fornecedores.values()}
    return {"recomendacao": _res(ot), "linha_de_base_mais_barato_por_item": _res(base),
            "comparacao": cmp, "explicacao": explicar(ot, base, cmp, nomes)}


if __name__ == "__main__":
    mcp.run()
