# Motor de cotação e combinação de fornecedores

Protótipo independente (não mexe no Engenheiro-Copilot). Recebe uma requisição de
materiais da obra, as cotações dos fornecedores, e responde **de quem comprar o quê**
para gastar o mínimo, considerando frete, pedido mínimo, prazo, estoque e risco de atraso.

> **Todos os dados são simulados** (fornecedores, preços, histórico de entregas).
> Os preços de referência imitam a ordem de grandeza do SINAPI, mas não são o SINAPI.

## Rodar (Windows)

```bat
cd motor-cotacao
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python demo.py          :: demonstração ponta a ponta
python -m pytest -q     :: 34 testes
```

Se existir `ANTHROPIC_API_KEY` ou `GROQ_API_KEY` no ambiente, a extração de cotações usa LLM;
sem chave, usa um extrator por regras (`python demo.py --regras` força isso).

## A ideia em uma frase

**O coração é otimização, não IA.** Cada tecnologia do título faz uma coisa só:

| Peça | Faz | Não faz |
|---|---|---|
| **LLM** (`extracao.py`) | Lê cotação bagunçada (WhatsApp, e-mail, PDF) e preenche um schema | Não calcula nem decide |
| **ML** (`risco.py`) | Estima P(atraso) por fornecedor e marca preços anômalos | Não escolhe fornecedor |
| **MILP** (`otimizador.py`) | Escolhe a combinação ótima, com garantia de ótimo | Não lê texto |
| **MCP** (`servidor_mcp.py`) | Expõe tudo como ferramentas que um agente de IA chama | Não tem inteligência própria |

## Fluxo

```
cotação em texto ──LLM──> JSON ──validação──> unidades base ──> cotações
                                   │                                   │
                              pendências                        anomalias (ML)
                         (revisão humana)                   suspeitos ficam fora
                                                                       │
requisição da obra ─────────────────────────────────────> MILP <── P(atraso) (ML)
                                                             │
                                          recomendação + comparação com "mais barato por item"
                                                             │
                                                  explicação (template) ──> humano aprova
```

## O modelo matemático

Itens $i$, fornecedores $j$. Variáveis: $q_{ij} \ge 0$ (quanto comprar) e $y_j \in \{0,1\}$ (fornecedor ativado).

$$\min \sum_{i,j} p_{ij}\,q_{ij}\,(1 + \lambda r_{ij}) + \sum_j F_j\,y_j$$

| Restrição | Significado |
|---|---|
| $\sum_j q_{ij} = Q_i$ | cada item é comprado por inteiro |
| $q_{ij} \le C_{ij}\,y_j$ | só compra de quem foi ativado (é isso que cobra o frete) e respeita o estoque dele |
| $\sum_i p_{ij} q_{ij} \ge M_j\,y_j$ | se ativar, respeita o pedido mínimo |
| $\sum_j y_j \le K$ | opcional: no máximo K fornecedores |
| par $(i,j)$ só existe se $j$ cotou $i$ **e** o prazo cabe | o prazo é filtro, não restrição |

- $r_{ij}$ é a probabilidade de atraso dada pelo ML e $\lambda$ é a fração do valor que se perde se atrasar.
  O termo $\lambda r_{ij} p_{ij} q_{ij}$ é um **custo esperado**: aparece no "custo total", mas não é desembolso.
- Saco, unidade e barra são **inteiros** (não se compra 0,37 saco); m³ é contínuo.
- Com `permitir_divisao=False`, $q_{ij} = Q_i x_{ij}$ com $x_{ij}$ binária (um fornecedor por item).
- Para 6 fornecedores existem $2^6 = 64$ subconjuntos. Os testes conferem o MILP contra **força bruta**
  em instâncias aleatórias. Com 30 fornecedores, a força bruta já não dá conta, e o branch-and-bound do solver (SCIP) dá.
- É uma variante do **problema de localização de facilidades com capacidade** (capacitated facility location).

### Por que não "o mais barato de cada item"?

Porque o frete é um custo da **combinação**, não do item. Na demo, a estratégia gulosa usa 5 fornecedores,
paga R$ 1.800 de frete e ainda viola um pedido mínimo (o fornecedor recusaria). O ótimo usa 2 fornecedores
e sai cerca de 6% mais barato no custo total.

## ML: o que é e o que não é

- **Risco de atraso:** regressão logística sobre (fornecedor, categoria, distância, prazo prometido, valor,
  período chuvoso). A escolha é deliberada: há pouco dado, a regressão é interpretável e devolve
  probabilidade, que é exatamente o que entra num custo esperado. A avaliação é feita num conjunto separado:
  AUC ≈ 0,7 e Brier melhor que chutar a média.
- **Preço anômalo:** compara com a referência e com a mediana dos concorrentes (z robusto com MAD,
  que não se deixa levar por um valor absurdo). Se a razão preço/referência bate com um fator de
  unidade (1/50 no cimento), o sistema avisa que **parece preço por kg lançado como por saco**.
  Preço suspeito de barato **fica fora da otimização**. Sem isso, o otimizador escolheria justamente o erro de digitação.
- **Honestidade:** o histórico é simulado. Com dados reais o código é o mesmo; os números mudam.

## LLM: regras de segurança

1. Saída forçada num schema (tool use no Claude, JSON mode no Groq).
2. Tudo é validado em código: preço > 0, unidade conversível, insumo reconhecido com confiança ≥ 75%, prazo presente.
3. O que falha vira **pendência** e não entra no otimizador. Exemplo da demo: "Tinta acrílica 18L" não está no catálogo e fica pendente.
4. A explicação final é montada por **template** com os números do otimizador, sem risco de valor inventado.

## MCP

```bat
python -m motor_cotacao.servidor_mcp
```

Ferramentas: `listar_insumos`, `listar_fornecedores`, `precos_do_insumo`, `registrar_cotacao_texto`,
`risco_fornecedores`, `otimizar_compra`.

Para usar no Claude Desktop, adicione em `claude_desktop_config.json`, ajustando o caminho:

```json
{
  "mcpServers": {
    "motor-cotacao": {
      "command": "C:\\caminho\\motor-cotacao\\.venv\\Scripts\\python.exe",
      "args": ["-m", "motor_cotacao.servidor_mcp"],
      "cwd": "C:\\caminho\\motor-cotacao"
    }
  }
}
```

Depois é só pedir: *"preciso de 120 sacos de cimento e 12 m³ de areia em até 7 dias, de quem compro?"*.
O agente chama as ferramentas sozinho.

## Arquivos

```
motor_cotacao/
  modelos.py         estruturas de dados (tudo em unidade base)
  unidades.py        R$/kg, R$/t, R$/milheiro -> R$/unidade base
  catalogo.py        insumos + casamento de descrições (rapidfuzz)
  dados_simulados.py gerador com "verdade escondida" para testar o ML
  risco.py           regressão logística de atraso + detecção de anomalias
  otimizador.py      MILP (OR-Tools/SCIP) + linha de base gulosa
  extracao.py        LLM (Claude/Groq) ou regras -> cotação validada
  explicacao.py      texto da recomendação (template)
  motor.py           fachada que junta tudo
  servidor_mcp.py    ferramentas MCP
demo.py              demonstração
tests/test_motor.py  34 testes (inclui MILP x força bruta)
```

## Limitações e próximos passos

- Desconto por volume (faixas de preço) entra no MILP com variáveis binárias por faixa, sem mudar a estrutura.
- Validade da cotação e condição de pagamento (prazo de pagamento tem valor financeiro) ainda não são considerados.
- O frete real depende de volume e peso; aqui ele é fixo por pedido.
- Ligação com o módulo Canteiro: uma `SolicitacaoCompra` (estoque mínimo ou requisição sem estoque) vira a entrada `itens` de `otimizar_compra`.
- Os preços de referência deveriam vir do SINAPI do estado da obra.
