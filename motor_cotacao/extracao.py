"""Camada de LLM: transformar cotação bagunçada (e-mail, WhatsApp, PDF em texto)
em dado estruturado.

Regra de ouro: o LLM LÊ, não DECIDE nem CALCULA.
  - ele só preenche um schema fixo (descrição, unidade, preço, prazo...);
  - o código valida tudo (preço > 0, unidade conhecida, insumo casado);
  - o que não passar na validação vira PENDÊNCIA para revisão humana,
    nunca entra direto no otimizador.

Provedores (o primeiro que tiver chave no ambiente):
  ANTHROPIC_API_KEY -> Claude, com "tool use" forçado (saída sempre no schema)
  GROQ_API_KEY      -> Groq (gratuito), modo JSON
  nenhum            -> extrator por regras (regex), suficiente para a demo
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .catalogo import POR_CODIGO, casar_descricao
from .modelos import Cotacao, ItemCotado
from .unidades import preco_para_base

SCHEMA = {
    "type": "object",
    "properties": {
        "fornecedor": {"type": "string", "description": "nome do fornecedor, se aparecer"},
        "frete": {"type": ["number", "null"], "description": "frete em R$ (0 se grátis, null se não informado)"},
        "pedido_minimo": {"type": ["number", "null"]},
        "prazo_geral_dias": {"type": ["integer", "null"], "description": "prazo que vale para todos os itens"},
        "itens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "descricao": {"type": "string", "description": "exatamente como escrito pelo fornecedor"},
                    "unidade": {"type": "string", "description": "unidade a que o preço se refere: saco, kg, t, m3, un, milheiro, barra"},
                    "preco_unitario": {"type": "number", "description": "R$ por unidade, como número (36.9, não '36,90')"},
                    "prazo_dias": {"type": ["integer", "null"]},
                    "quantidade_disponivel": {"type": ["number", "null"]},
                },
                "required": ["descricao", "unidade", "preco_unitario"],
            },
        },
    },
    "required": ["itens"],
}

INSTRUCOES = (
    "Você extrai dados de cotações de materiais de construção enviadas por fornecedores brasileiros. "
    "Preencha somente o que estiver escrito; não invente preço, prazo nem unidade. "
    "Preços em reais com vírgula decimal (R$ 1.234,56) devem virar número (1234.56). "
    "Se o preço for 'por kg', 'por tonelada', 'o milheiro' etc., registre essa unidade."
)


@dataclass
class Pendencia:
    linha: str
    motivo: str


@dataclass
class ResultadoExtracao:
    cotacao: Cotacao
    frete: float | None
    pedido_minimo: float | None
    pendencias: list[Pendencia] = field(default_factory=list)
    provedor: str = "regras"
    confiancas: dict[str, float] = field(default_factory=dict)   # insumo -> confiança do casamento


# ------------------------------------------------------------------ provedores

def _via_anthropic(texto: str) -> dict:
    import anthropic
    cli = anthropic.Anthropic()
    resp = cli.messages.create(
        model=os.getenv("MOTOR_MODELO_CLAUDE", "claude-haiku-5-5"),
        max_tokens=2000,
        system=INSTRUCOES,
        tools=[{"name": "registrar_cotacao", "description": "Registra a cotação extraída", "input_schema": SCHEMA}],
        tool_choice={"type": "tool", "name": "registrar_cotacao"},
        messages=[{"role": "user", "content": texto}],
    )
    bloco = next(b for b in resp.content if b.type == "tool_use")
    return bloco.input


def _via_groq(texto: str) -> dict:
    import httpx
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": os.getenv("MOTOR_MODELO_GROQ", "openai/gpt-oss-120b"),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": INSTRUCOES + " Responda só com JSON neste schema: " + json.dumps(SCHEMA)},
                {"role": "user", "content": texto},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


_DINHEIRO = r"R\$\s*([\d\.]+(?:,\d{1,2})?)"


def _num(br: str) -> float:
    return float(br.replace(".", "").replace(",", "."))


def _via_regras(texto: str) -> dict:
    """Extrator simples para linhas do tipo
    'Cimento CP II 50kg ..... R$ 36,90/saco - entrega 3 dias'."""
    out = {"itens": [], "frete": None, "pedido_minimo": None, "prazo_geral_dias": None}
    for linha in texto.splitlines():
        l = linha.strip(" -•*\t")
        if not l:
            continue
        low = l.lower()
        m_val = re.search(_DINHEIRO, l)
        if "frete" in low:
            out["frete"] = 0.0 if ("grátis" in low or "gratis" in low or "cif" in low) else (_num(m_val.group(1)) if m_val else None)
            continue
        if "mínimo" in low or "minimo" in low:
            out["pedido_minimo"] = _num(m_val.group(1)) if m_val else None
            continue
        m_prazo = re.search(r"(\d+)\s*dias?", low)
        if m_val is None:
            if m_prazo and ("prazo" in low or "entrega" in low):
                out["prazo_geral_dias"] = int(m_prazo.group(1))
            continue
        m_un = re.search(_DINHEIRO + r"\s*(?:/|por|o|a)\s*([a-zA-Zçãé³0-9]+)", l)
        desc = re.split(r"R\$|[:=]|\.{3,}| - ", l)[0].strip(" .:-")
        out["itens"].append({
            "descricao": desc,
            "unidade": m_un.group(2) if m_un else "",
            "preco_unitario": _num(m_val.group(1)),
            "prazo_dias": int(m_prazo.group(1)) if m_prazo else None,
            "quantidade_disponivel": None,
        })
    return out


# ------------------------------------------------------------------ pipeline

def extrair(texto: str, fornecedor_id: str, provedor: str | None = None) -> ResultadoExtracao:
    if provedor is None:
        provedor = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "groq" if os.getenv("GROQ_API_KEY") else "regras"
    bruto = {"anthropic": _via_anthropic, "groq": _via_groq, "regras": _via_regras}[provedor](texto)

    prazo_geral = bruto.get("prazo_geral_dias")
    itens, pend, conf = [], [], {}
    for li in bruto.get("itens", []):
        desc = str(li.get("descricao", ""))
        preco = li.get("preco_unitario")
        if not isinstance(preco, (int, float)) or preco <= 0:
            pend.append(Pendencia(desc, "preço ausente ou inválido"))
            continue
        cod, score = casar_descricao(desc)
        if cod is None:
            pend.append(Pendencia(desc, f"não reconheci o insumo (confiança {score:.0f}%)"))
            continue
        ins = POR_CODIGO[cod]
        unidade = li.get("unidade") or ins.unidade
        preco_base = preco_para_base(float(preco), unidade, ins)
        if preco_base is None:
            pend.append(Pendencia(desc, f"não sei converter '{unidade}' para {ins.unidade}"))
            continue
        prazo = li.get("prazo_dias") or prazo_geral
        if prazo is None:
            pend.append(Pendencia(desc, "prazo de entrega não informado"))
            continue
        itens.append(ItemCotado(cod, round(preco_base, 4), int(prazo), li.get("quantidade_disponivel")))
        conf[cod] = score

    return ResultadoExtracao(Cotacao(fornecedor_id, itens, origem=provedor),
                             bruto.get("frete"), bruto.get("pedido_minimo"), pend, provedor, conf)
