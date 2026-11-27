"""O coração do motor: programação linear inteira mista (MILP).

Conjuntos:   i = itens da requisição,  j = fornecedores
Dados:       Q_i  quantidade pedida          p_ij  preço unitário (unidade base)
             F_j  frete fixo                 M_j   pedido mínimo (R$)
             C_ij estoque disponível         r_ij  P(atraso) estimada pelo ML
             λ    peso do risco (fração do valor que se "perde" se atrasar)
Variáveis:   q_ij >= 0   quanto do item i comprar do fornecedor j
             y_j ∈ {0,1} se o fornecedor j entra no pedido

  min   Σ p_ij q_ij (1 + λ r_ij)  +  Σ F_j y_j
  s.a.  Σ_j q_ij = Q_i                 (cada item é atendido por inteiro)
        q_ij <= C_ij y_j               (só compra de quem foi ativado: é isso que "cobra" o frete)
        Σ_i p_ij q_ij >= M_j y_j       (se ativar, respeita o pedido mínimo)
        Σ_j y_j <= K                   (opcional: no máximo K fornecedores)
        pares (i,j) sem cotação ou com prazo > prazo máximo nem viram variável

Por que não testar todas as combinações? Com m fornecedores são 2^m subconjuntos
(e ainda dá pra dividir itens). O solver (SCIP, via OR-Tools) usa branch-and-bound:
resolve relaxações lineares e corta ramos que provadamente não melhoram.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ortools.linear_solver import pywraplp

from .catalogo import POR_CODIGO
from .modelos import Cotacao, Fornecedor, Requisicao

UNIDADES_INTEIRAS = {"saco", "un", "barra"}   # não se compra 0,37 saco


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@dataclass
class Alocacao:
    fornecedor: str
    insumo: str
    quantidade: float
    preco_unitario: float
    prazo_dias: int
    prob_atraso: float

    @property
    def subtotal(self) -> float:
        return self.quantidade * self.preco_unitario


@dataclass
class ItemInviavel:
    insumo: str
    motivo: str


@dataclass
class Resultado:
    metodo: str                            # "otimo" | "guloso"
    alocacoes: list[Alocacao]
    fornecedores_usados: list[str]
    custo_material: float
    custo_frete: float
    custo_risco: float                     # custo ESPERADO de atraso (não é desembolso)
    itens_inviaveis: list[ItemInviavel] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    @property
    def desembolso(self) -> float:        # o que sai do caixa
        return self.custo_material + self.custo_frete

    @property
    def custo_total(self) -> float:       # o que o otimizador minimiza
        return self.desembolso + self.custo_risco


# r(fornecedor, insumo, valor_do_par, prazo) -> probabilidade de atraso
FuncRisco = Callable[[str, str, float, int], float]


def _pares(req: Requisicao, cotacoes: dict[str, Cotacao], permitir_divisao: bool):
    """Monta os pares viáveis (i,j) e diagnostica itens impossíveis ANTES do solver,
    para poder explicar o porquê em vez de só dizer 'infeasible'."""
    pares, inviaveis = {}, []
    for item in req.itens:
        cotaram, no_prazo, cap_total = 0, 0, 0.0
        for forn, cot in cotacoes.items():
            for it in cot.itens:
                if it.insumo != item.insumo:
                    continue
                cotaram += 1
                if it.prazo_dias > item.prazo_max_dias:
                    continue
                no_prazo += 1
                cap = item.quantidade if it.quantidade_disponivel is None else min(item.quantidade, it.quantidade_disponivel)
                if not permitir_divisao and cap < item.quantidade:
                    continue
                pares[(item.insumo, forn)] = (it, cap)
                cap_total += cap
        if cotaram == 0:
            inviaveis.append(ItemInviavel(item.insumo, "nenhum fornecedor cotou"))
        elif no_prazo == 0:
            inviaveis.append(ItemInviavel(item.insumo, f"ninguém entrega em até {item.prazo_max_dias} dias"))
        elif cap_total + 1e-9 < item.quantidade:
            msg = "estoque somado dos fornecedores é insuficiente" if permitir_divisao \
                else "nenhum fornecedor tem tudo sozinho (permita dividir o item)"
            inviaveis.append(ItemInviavel(item.insumo, msg))
    bloqueados = {x.insumo for x in inviaveis}
    pares = {k: v for k, v in pares.items() if k[0] not in bloqueados}
    return pares, inviaveis


def otimizar(req: Requisicao, fornecedores: dict[str, Fornecedor], cotacoes: dict[str, Cotacao],
             risco: FuncRisco | None = None, peso_risco: float = 0.15,
             permitir_divisao: bool = True, max_fornecedores: int | None = None,
             tempo_limite_s: int = 20) -> Resultado:
    risco = risco or (lambda *_: 0.0)
    qtd = {it.insumo: it.quantidade for it in req.itens}
    pares, inviaveis = _pares(req, cotacoes, permitir_divisao)
    itens = sorted({i for i, _ in pares})
    forns = sorted({j for _, j in pares})
    if not itens:
        return Resultado("otimo", [], [], 0, 0, 0, inviaveis, ["nada a otimizar"])

    r = {(i, j): risco(j, i, it.preco_unitario * qtd[i], it.prazo_dias) for (i, j), (it, _) in pares.items()}

    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(tempo_limite_s * 1000)
    y = {j: s.BoolVar(f"y_{j}") for j in forns}
    q = {}
    for (i, j), (it, cap) in pares.items():
        if permitir_divisao:
            inteiro = POR_CODIGO[i].unidade in UNIDADES_INTEIRAS
            q[i, j] = (s.IntVar if inteiro else s.NumVar)(0, cap, f"q_{i}_{j}")
        else:   # tudo ou nada: q = Q * x, x binária
            q[i, j] = qtd[i] * s.BoolVar(f"x_{i}_{j}")

    s.Minimize(sum(it.preco_unitario * q[i, j] * (1 + peso_risco * r[i, j]) for (i, j), (it, _) in pares.items())
               + sum(fornecedores[j].frete_fixo * y[j] for j in forns))

    for i in itens:                                            # demanda
        s.Add(sum(q[i, j] for j in forns if (i, j) in q) == qtd[i])
    for (i, j), (_, cap) in pares.items():                     # ativação (cobra o frete)
        s.Add(q[i, j] <= cap * y[j])
    for j in forns:                                            # pedido mínimo
        mj = fornecedores[j].pedido_minimo
        if mj > 0:
            s.Add(sum(pares[i, j][0].preco_unitario * q[i, j] for i in itens if (i, j) in q) >= mj * y[j])
    if max_fornecedores is not None:                           # limite de fornecedores
        s.Add(sum(y.values()) <= max_fornecedores)

    status = s.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("Sem solução viável: as restrições se contradizem. Tente permitir divisão, "
                         "aumentar o número máximo de fornecedores ou o prazo.")

    aloc = []
    for (i, j), (it, _) in pares.items():
        v = q[i, j].solution_value()
        if v > 1e-6:
            aloc.append(Alocacao(j, i, round(v, 4), it.preco_unitario, it.prazo_dias, r[i, j]))
    return _montar("otimo", aloc, fornecedores, peso_risco, inviaveis)


def guloso(req: Requisicao, fornecedores: dict[str, Fornecedor], cotacoes: dict[str, Cotacao],
           risco: FuncRisco | None = None, peso_risco: float = 0.15,
           permitir_divisao: bool = True) -> Resultado:
    """O que um comprador faria na planilha: o mais barato de cada item, ignorando frete.
    Serve de linha de base para medir quanto o otimizador economiza."""
    risco = risco or (lambda *_: 0.0)
    qtd = {it.insumo: it.quantidade for it in req.itens}
    pares, inviaveis = _pares(req, cotacoes, permitir_divisao)
    aloc = []
    for i in sorted({i for i, _ in pares}):
        opcoes = sorted(((it.preco_unitario, j, it, cap) for (ii, j), (it, cap) in pares.items() if ii == i),
                        key=lambda t: t[0])
        falta = qtd[i]
        for _, j, it, cap in opcoes:
            if falta <= 1e-9:
                break
            tomar = min(cap, falta)
            aloc.append(Alocacao(j, i, tomar, it.preco_unitario, it.prazo_dias,
                                 risco(j, i, it.preco_unitario * qtd[i], it.prazo_dias)))
            falta -= tomar
    res = _montar("guloso", aloc, fornecedores, peso_risco, inviaveis)
    for j in res.fornecedores_usados:
        v = sum(a.subtotal for a in res.alocacoes if a.fornecedor == j)
        if v < fornecedores[j].pedido_minimo:
            res.avisos.append(f"{fornecedores[j].nome}: pedido de {brl(v)} abaixo do mínimo "
                              f"de {brl(fornecedores[j].pedido_minimo)} (o fornecedor recusaria)")
    return res


def _montar(metodo, aloc, fornecedores, peso_risco, inviaveis) -> Resultado:
    usados = sorted({a.fornecedor for a in aloc})
    material = sum(a.subtotal for a in aloc)
    frete = sum(fornecedores[j].frete_fixo for j in usados)
    risco = sum(peso_risco * a.prob_atraso * a.subtotal for a in aloc)
    return Resultado(metodo, sorted(aloc, key=lambda a: (a.fornecedor, a.insumo)), usados,
                     round(material, 2), round(frete, 2), round(risco, 2), inviaveis)


def comparar(otimo: Resultado, base: Resultado) -> dict:
    eco = base.custo_total - otimo.custo_total
    return {
        "economia_custo_total": round(eco, 2),
        "economia_desembolso": round(base.desembolso - otimo.desembolso, 2),
        "economia_percentual": round(100 * eco / base.custo_total, 2) if base.custo_total else 0.0,
        "fretes_evitados": len(base.fornecedores_usados) - len(otimo.fornecedores_usados),
    }

