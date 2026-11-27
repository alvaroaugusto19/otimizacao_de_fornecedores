"""Demonstração ponta a ponta do motor (dados SIMULADOS).

    python demo.py            # usa LLM se houver ANTHROPIC_API_KEY ou GROQ_API_KEY; senão, regras
    python demo.py --regras   # força o extrator por regras
"""
import sys

from motor_cotacao import MotorCotacao
from motor_cotacao.catalogo import POR_CODIGO
from motor_cotacao.explicacao import explicar
from motor_cotacao.otimizador import brl

try:
    sys.stdout.reconfigure(encoding="utf-8")   # acentos no terminal do Windows
except Exception:
    pass

WHATSAPP = """Boa tarde! Segue orçamento Depósito Planalto:
Cimento CP-II 50kg Votoran ..... R$ 36,90/saco
Areia média lavada: R$ 82,00 a tonelada
Tijolo 9 furos R$ 980,00 o milheiro
Brita 1 R$ 135,00/m³
Tinta acrílica 18L R$ 289,00/lata
Frete: R$ 150,00
Pedido mínimo R$ 1.000
Entrega em 2 dias"""

REQUISICAO = [  # (insumo, quantidade na unidade base, prazo máximo em dias)
    ("CIM-50", 120, 7),
    ("ARE-M", 12, 7),
    ("BRI-1", 8, 7),
    ("TIJ-9F", 6000, 10),
    ("VER-10", 80, 7),
    ("ARG-AC3", 40, 7),
]


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def tabela(res, nomes):
    print(f"{'fornecedor':28} {'insumo':34} {'qtd':>8} {'preço':>11} {'subtotal':>13} {'atraso':>7}")
    for a in res.alocacoes:
        ins = POR_CODIGO[a.insumo]
        print(f"{nomes[a.fornecedor][:28]:28} {ins.descricao[:34]:34} {a.quantidade:>8g} "
              f"{brl(a.preco_unitario):>11} {brl(a.subtotal):>13} {a.prob_atraso:>6.0%}")
    print(f"material {brl(res.custo_material)} | frete {brl(res.custo_frete)} | "
          f"risco esperado {brl(res.custo_risco)} | DESEMBOLSO {brl(res.desembolso)} | "
          f"CUSTO TOTAL {brl(res.custo_total)}")
    for a in res.avisos:
        print("  aviso:", a)


def main():
    m = MotorCotacao.simulado(seed=7, mes_entrega=11)
    nomes = {f.id: f.nome for f in m.fornecedores.values()}

    titulo("1. Modelo de risco de atraso (regressão logística no histórico simulado)")
    k = m.metricas
    print(f"treino {k.n_treino} entregas | teste {k.n_teste} | taxa de atraso {k.taxa_atraso:.0%}")
    print(f"AUC {k.auc:.2f} (0,5 = chute) | Brier {k.brier:.3f} vs {k.brier_base:.3f} chutando a média")
    for fid in m.fornecedores:
        print(f"  {nomes[fid]:28} P(atraso) num pedido de R$ 3.000, prazo 5 dias: "
              f"{m.prob_atraso(fid, m.cotacoes[fid].itens[0].insumo, 3000, 5):.0%}")

    titulo("2. Preços anômalos (antes de otimizar)")
    for a in m.anomalias():
        print(f"  {nomes[a.fornecedor]}: {POR_CODIGO[a.insumo].descricao} a {brl(a.preco)} -> {a.motivo}")
    print("  -> preços suspeitos de BARATOS demais ficam fora da otimização até alguém confirmar.")

    titulo("3. Cotação por WhatsApp -> dado estruturado")
    prov = "regras" if "--regras" in sys.argv else None
    r = m.registrar_texto("F1", WHATSAPP, provedor=prov)
    print(f"provedor: {r.provedor} | frete {r.frete} | pedido mínimo {r.pedido_minimo}")
    for it in r.cotacao.itens:
        ins = POR_CODIGO[it.insumo]
        print(f"  ok  {ins.descricao:34} {brl(it.preco_unitario)}/{ins.unidade}  prazo {it.prazo_dias}d "
              f"(confiança {r.confiancas[it.insumo]:.0f}%)")
    for p in r.pendencias:
        print(f"  PENDENTE  '{p.linha}': {p.motivo}")

    titulo("4. Otimização da requisição")
    for c, q, p in REQUISICAO:
        print(f"  {q:>6g} {POR_CODIGO[c].unidade:6} {POR_CODIGO[c].descricao} (até {p} dias)")
    ot, base, cmp = m.cotar(REQUISICAO, obra="Residencial Exemplo")
    print("\n>> Linha de base: mais barato por item (o que se faz na planilha)")
    tabela(base, nomes)
    print("\n>> Ótimo (MILP)")
    tabela(ot, nomes)
    print("\n" + explicar(ot, base, cmp, nomes))

    titulo("5. Sensibilidade: o que muda com outras regras de negócio")
    urgente = [(c, q, 2) for c, q, _ in REQUISICAO]
    cenarios = [
        ("padrão (peso de risco 0,15)", REQUISICAO, {}),
        ("ignorando risco (peso 0)", REQUISICAO, dict(peso_risco=0.0)),
        ("risco pesado (peso 1,0)", REQUISICAO, dict(peso_risco=1.0)),
        ("um fornecedor só", REQUISICAO, dict(max_fornecedores=1)),
        ("urgente: tudo em até 2 dias", urgente, {}),
    ]
    for nome, req, kw in cenarios:
        try:
            o, _, _ = m.cotar(req, **kw)
            print(f"  {nome:38} desembolso {brl(o.desembolso):>13} | custo total {brl(o.custo_total):>13} "
                  f"| fornecedores {', '.join(o.fornecedores_usados)}"
                  + (f" | fora: {', '.join(x.insumo for x in o.itens_inviaveis)}" if o.itens_inviaveis else ""))
        except ValueError as e:
            print(f"  {nome:38} {e}")


if __name__ == "__main__":
    main()
