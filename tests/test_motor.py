import itertools
import random

import pytest

from motor_cotacao import MotorCotacao
from motor_cotacao.catalogo import POR_CODIGO, casar_descricao
from motor_cotacao.extracao import extrair
from motor_cotacao.modelos import Cotacao, Fornecedor, ItemCotado, ItemRequisicao, Requisicao
from motor_cotacao.otimizador import guloso, otimizar
from motor_cotacao.unidades import preco_para_base

CODS = ["CIM-50", "ARE-M", "BRI-1", "TIJ-9F", "VER-10", "CAL-20"]


def _instancia(rnd: random.Random, n_forn: int):
    forns = {f"F{j}": Fornecedor(f"F{j}", f"F{j}", rnd.choice([0, 100, 300, 600, 900]), 0, 0) for j in range(n_forn)}
    cot = {}
    for j in forns:
        itens = [ItemCotado(c, round(POR_CODIGO[c].preco_referencia * rnd.uniform(0.8, 1.2), 2), rnd.randint(1, 8))
                 for c in CODS if rnd.random() < 0.6]
        cot[j] = Cotacao(j, itens)
    req = Requisicao("t", [ItemRequisicao(c, rnd.choice([10, 50, 200]), rnd.randint(3, 8)) for c in rnd.sample(CODS, 4)])
    return forns, cot, req


def _forca_bruta(forns, cot, req, risco, lam):
    """Testa TODOS os subconjuntos de fornecedores (2^m). Só serve para m pequeno."""
    melhor = float("inf")
    ids = list(forns)
    for k in range(1, len(ids) + 1):
        for S in itertools.combinations(ids, k):
            total = sum(forns[j].frete_fixo for j in S)
            for it in req.itens:
                opcoes = [x.preco_unitario * it.quantidade * (1 + lam * risco(j, it.insumo, 0, 0))
                          for j in S for x in cot[j].itens if x.insumo == it.insumo and x.prazo_dias <= it.prazo_max_dias]
                if not opcoes:
                    total = float("inf")
                    break
                total += min(opcoes)
            melhor = min(melhor, total)
    return melhor


@pytest.mark.parametrize("seed", range(25))
def test_otimo_igual_forca_bruta(seed):
    rnd = random.Random(seed)
    forns, cot, req = _instancia(rnd, n_forn=6)
    riscos = {j: rnd.uniform(0, 0.6) for j in forns}
    risco = lambda j, *_: riscos[j]
    bf = _forca_bruta(forns, cot, req, risco, 0.2)
    try:
        r = otimizar(req, forns, cot, risco, 0.2, permitir_divisao=False)
    except ValueError:
        assert bf == float("inf")
        return
    if r.itens_inviaveis:      # força bruta não sabe excluir itens; ignora esses casos
        return
    assert r.custo_total == pytest.approx(bf, abs=0.05)
    # e o ótimo nunca é pior que o guloso
    g = guloso(req, forns, cot, risco, 0.2, permitir_divisao=False)
    assert r.custo_total <= g.custo_total + 0.05


def test_frete_faz_consolidar():
    forns = {"A": Fornecedor("A", "A", 150), "B": Fornecedor("B", "B", 400), "C": Fornecedor("C", "C", 400)}
    cot = {"A": Cotacao("A", [ItemCotado("CIM-50", 35, 2), ItemCotado("ARE-M", 125, 2)]),
           "B": Cotacao("B", [ItemCotado("CIM-50", 33, 2)]),
           "C": Cotacao("C", [ItemCotado("ARE-M", 120, 2)])}
    req = Requisicao("t", [ItemRequisicao("CIM-50", 50, 5), ItemRequisicao("ARE-M", 5, 5)])
    r = otimizar(req, forns, cot)
    g = guloso(req, forns, cot)
    assert g.fornecedores_usados == ["B", "C"]
    assert r.fornecedores_usados == ["A"]
    assert r.custo_total < g.custo_total


def test_pedido_minimo_respeitado():
    forns = {"A": Fornecedor("A", "A", 0, pedido_minimo=5000), "B": Fornecedor("B", "B", 100)}
    cot = {"A": Cotacao("A", [ItemCotado("CIM-50", 30, 2)]), "B": Cotacao("B", [ItemCotado("CIM-50", 34, 2)])}
    req = Requisicao("t", [ItemRequisicao("CIM-50", 50, 5)])   # 50 x 30 = 1500 < 5000
    r = otimizar(req, forns, cot)
    assert r.fornecedores_usados == ["B"]
    assert guloso(req, forns, cot).avisos          # o guloso escolheria A e violaria o mínimo


def test_divisao_por_estoque_limitado_e_inteira():
    forns = {"A": Fornecedor("A", "A", 0), "B": Fornecedor("B", "B", 0)}
    cot = {"A": Cotacao("A", [ItemCotado("CIM-50", 30, 2, quantidade_disponivel=70)]),
           "B": Cotacao("B", [ItemCotado("CIM-50", 34, 2)])}
    req = Requisicao("t", [ItemRequisicao("CIM-50", 100, 5)])
    r = otimizar(req, forns, cot)
    qs = {a.fornecedor: a.quantidade for a in r.alocacoes}
    assert qs == {"A": 70, "B": 30}
    # sem divisão, A não tem tudo -> B leva tudo
    r2 = otimizar(req, forns, cot, permitir_divisao=False)
    assert {a.fornecedor for a in r2.alocacoes} == {"B"}


def test_diagnostico_inviavel():
    forns = {"A": Fornecedor("A", "A", 0)}
    cot = {"A": Cotacao("A", [ItemCotado("CIM-50", 30, 9)])}
    req = Requisicao("t", [ItemRequisicao("CIM-50", 10, 3), ItemRequisicao("BRI-1", 2, 5)])
    r = otimizar(req, forns, cot)
    motivos = {x.insumo: x.motivo for x in r.itens_inviaveis}
    assert "3 dias" in motivos["CIM-50"] and "nenhum" in motivos["BRI-1"]


def test_max_fornecedores_impossivel_da_erro_claro():
    forns = {"A": Fornecedor("A", "A", 0), "B": Fornecedor("B", "B", 0)}
    cot = {"A": Cotacao("A", [ItemCotado("CIM-50", 30, 2)]), "B": Cotacao("B", [ItemCotado("ARE-M", 120, 2)])}
    req = Requisicao("t", [ItemRequisicao("CIM-50", 10, 5), ItemRequisicao("ARE-M", 2, 5)])
    with pytest.raises(ValueError, match="viável"):
        otimizar(req, forns, cot, max_fornecedores=1)


def test_unidades():
    cim, are, tij = POR_CODIGO["CIM-50"], POR_CODIGO["ARE-M"], POR_CODIGO["TIJ-9F"]
    assert preco_para_base(0.74, "kg", cim) == pytest.approx(37.0)
    assert preco_para_base(740, "tonelada", cim) == pytest.approx(37.0)
    assert preco_para_base(82, "t", are) == pytest.approx(123.0)
    assert preco_para_base(980, "milheiro", tij) == pytest.approx(0.98)
    assert preco_para_base(10, "lata", cim) is None


def test_casamento_de_descricoes():
    assert casar_descricao("CIMENTO CP-II E-32 50KG VOTORAN")[0] == "CIM-50"
    assert casar_descricao("Ferro 3/8 CA50 12m")[0] == "VER-10"
    assert casar_descricao("Tinta acrílica 18L")[0] is None   # não adivinha


def test_extracao_por_regras():
    txt = """Cimento CP-II 50kg ..... R$ 0,74 por kg - entrega 3 dias
Areia média: R$ 82,00 a tonelada
Prego 17x21 R$ 25,00/kg
Frete grátis
Pedido mínimo R$ 2.000,00
Prazo de entrega: 4 dias"""
    r = extrair(txt, "F1", provedor="regras")
    itens = {i.insumo: i for i in r.cotacao.itens}
    assert itens["CIM-50"].preco_unitario == pytest.approx(37.0) and itens["CIM-50"].prazo_dias == 3
    assert itens["ARE-M"].preco_unitario == pytest.approx(123.0) and itens["ARE-M"].prazo_dias == 4
    assert r.frete == 0.0 and r.pedido_minimo == 2000.0
    assert [p.linha for p in r.pendencias] == ["Prego 17x21"]


def test_motor_simulado_ponta_a_ponta():
    m = MotorCotacao.simulado(seed=7, mes_entrega=11)
    assert m.metricas.auc > 0.6                      # o modelo aprende algo além do chute
    anom = m.anomalias()
    assert any(a.insumo == "CIM-50" and "por kg" in a.motivo for a in anom)
    ot, base, cmp = m.cotar([("CIM-50", 100, 7), ("ARE-M", 10, 7), ("TIJ-9F", 3000, 10)])
    # o preço de R$ 0,74 (erro de unidade) não pode ter sido escolhido
    assert all(a.preco_unitario > 10 for a in ot.alocacoes if a.insumo == "CIM-50")
    assert any("suspeito" in a for a in ot.avisos)
    assert ot.custo_total <= base.custo_total + 0.01
    assert cmp["economia_custo_total"] >= 0
