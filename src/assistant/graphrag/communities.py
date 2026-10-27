"""Сообщества графа и их сводки. Требование FR-3."""

import json

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 7
MAX_SUMMARY_WORDS = 150


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Считаем (для алгоритма Лувена) граф ненаправленным. веса встречных рёбер складываются:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1))
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += w
        else:
            undirected.add_edge(u, v, weight=w)

    partition = nx.algorithms.community.louvain_communities(
        undirected, weight="weight", seed=SEED
    )

    # Сортировка сообществ (Нет гарантии порядка сообществ на разных прогонах даже при одном сиде)
    ordered = sorted(partition, key=lambda members: min(members))
    return [
        Community(community_id=index, nodes=sorted(members))
        for index, members in enumerate(ordered)
    ]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    if not community.nodes:
        return ""

    nodes = set(community.nodes)
    subgraph = graph.subgraph(nodes)

    # 1. Состав по типам сущностей.
    by_type: dict[str, int] = {}
    for node in nodes:
        entity_type = graph.nodes[node].get("type", "?")
        by_type[entity_type] = by_type.get(entity_type, 0) + 1

    # 2. Связи: сколько рёбер и какой суммарный вес по каждому типу.
    by_relation: dict[str, dict[str, int]] = {}
    weighted_edges: list[tuple[float, str, str, str]] = []
    for subject, obj, data in subgraph.edges(data=True):
        rel_type = data.get("type", "?")
        weight = int(data.get("weight", 1))
        bucket = by_relation.setdefault(rel_type, {"рёбер": 0, "суммарный_вес": 0})
        bucket["рёбер"] += 1
        bucket["суммарный_вес"] += weight
        weighted_edges.append((weight, subject, obj, rel_type))

    weighted_edges.sort(key=lambda item: (-item[0], item[1], item[2]))

    # 3. Степени внутри сообщества — кто «хаб».
    degrees = sorted(subgraph.degree(), key=lambda kv: (-kv[1], kv[0]))

    def label(node_id: str) -> str:
        attrs = graph.nodes[node_id]
        return f"{attrs.get('name', node_id)} ({attrs.get('type', '?')})"

    facts = {
        "узлов": len(nodes),
        "узлы_по_типам": by_type,
        "связей": len(weighted_edges),
        "связи_по_типам": by_relation,
        "самые_тяжёлые_связи": [
            {"субъект": label(s), "связь": r, "объект": label(o), "вес": w}
            for w, s, o, r in weighted_edges[:5]
        ],
        "самые_связанные_узлы": [
            {"сущность": label(node), "связей": int(degree)}
            for node, degree in degrees[:5]
        ],
    }

    system = (
        "Ты пишешь краткие сводки сообществ графа знаний сервисной службы. "
        "Используй только числа и сущности из данных ниже — ничего не додумывай "
        "и не пересчитывай. "
        f"Длина — не больше {MAX_SUMMARY_WORDS} слов. "
        "Пиши по-русски сплошным текстом, без заголовков и списков."
    )
    user = (
        f"Сообщество №{community.community_id}. Данные:\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
    )

    response = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=400,
    )
    return _limit_words(response.content.strip(), MAX_SUMMARY_WORDS)


def _limit_words(text: str, limit: int) -> str:
    """Гарантирует не больше `limit` слов, обрезая по границе слова."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",;:.!?…") + "…"