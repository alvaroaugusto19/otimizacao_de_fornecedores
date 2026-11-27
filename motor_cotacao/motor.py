"""Fachada: junta catálogo, cotações, modelo de risco e otimizador num objeto só.
É isso que o servidor MCP (e, no futuro, a API do app) chama."""
from __future__ import annotations

from datetime import date

from . import dados_simulados
from .catalogo import CATALOGO, POR_CODIGO
from .extracao import ResultadoExtracao, extrair
from .modelos import Cotacao, Entrega, Fornecedor, ItemRequisicao, Requisicao
from .otimizador import Resultado, brl, comparar, guloso, otimizar
from .risco import Anomalia, ModeloRisco, detectar_anomalias


class MotorCotacao:
    def __init__(self, fornecedores: list[Fornecedor], cotacoes: dict[str, Cotacao],
                 historico: list[Entrega], mes_entrega: int | None = None):
        self.fornecedores = {f.id: f for f in fornecedores}
        self.cotacoes = cotacoes
        self.historico = historico
        self.mes = mes_entrega or date.today().month
        self.modelo = ModeloRisco()
        self.metricas = self.modelo.treinar(historico) if historico else None

    @classmethod
    def simulado(cls, seed: int = 7, mes_entrega: int | None = None) -> "MotorCotacao":
        f, c, h = dados_simulados.gerar(seed)
        return cls(f, c, h, mes_entrega)

    # ---------------------------------------------------------------- risco
    def prob_atraso(self, fornecedor: str, insumo: str, valor: float, prazo: int) -> float:
        if self.metricas is None:
            return 0.0
        f = self.fornecedores[fornecedor]
        return self.modelo.prob_atraso(fornecedor, POR_CODIGO[insumo].categoria, f.distancia_km,
                                       prazo, valor, self.mes)

    def anomalias(self) -> list[Anomalia]:
        return detectar_anomalias(self.cotacoes)

    # ---------------------------------------------------------------- cotações
    def registrar_texto(self, fornecedor_id: str, texto: str, provedor: str | None = None) -> ResultadoExtracao:
        """Extrai, valida e registra. Itens com pendência NÃO entram."""
        if fornecedor_id not in self.fornecedores:
            raise KeyError(f"fornecedor desconhecido: {fornecedor_id}")
        r = extrair(texto, fornecedor_id, provedor)
        f = self.fornecedores[fornecedor_id]
        if r.frete is not None:
            f.frete_fixo = r.frete
        if r.pedido_minimo is not None:
            f.pedido_minimo = r.pedido_minimo
        atual = {it.insumo: it for it in self.cotacoes.get(fornecedor_id, Cotacao(fornecedor_id, [])).itens}
        for it in r.cotacao.itens:       # cotação nova substitui a antiga item a item
            atual[it.insumo] = it
        self.cotacoes[fornecedor_id] = Cotacao(fornecedor_id, list(atual.values()), origem=r.provedor)
        return r

    # ---------------------------------------------------------------- otimização
    def cotar(self, itens: list[tuple[str, float, int]], obra: str = "obra",
              peso_risco: float = 0.15, permitir_divisao: bool = True,
              max_fornecedores: int | None = None) -> tuple[Resultado, Resultado, dict]:
        """itens = [(código, quantidade, prazo máximo em dias), ...]"""
        req = Requisicao(obra, [ItemRequisicao(c, q, p) for c, q, p in itens])
        # preço suspeito de barato demais (ex.: erro de unidade) fica FORA até alguém confirmar;
        # senão o otimizador "ama" exatamente o erro de digitação.
        suspeitos = {(a.fornecedor, a.insumo): a for a in self.anomalias() if a.razao < 1}
        cot = {fid: Cotacao(fid, [it for it in c.itens if (fid, it.insumo) not in suspeitos], c.validade, c.origem)
               for fid, c in self.cotacoes.items()}
        ot = otimizar(req, self.fornecedores, cot, self.prob_atraso, peso_risco,
                      permitir_divisao, max_fornecedores)
        base = guloso(req, self.fornecedores, cot, self.prob_atraso, peso_risco, permitir_divisao)
        pedidos = {it.insumo for it in req.itens}
        for (fid, cod), a in suspeitos.items():
            if cod in pedidos:
                ot.avisos.append(f"Ignorado preço suspeito de {self.fornecedores[fid].nome} para {cod}: "
                                 f"{brl(a.preco)} ({a.motivo}). Confirme antes de usar.")
        return ot, base, comparar(ot, base)

    @staticmethod
    def catalogo():
        return CATALOGO
