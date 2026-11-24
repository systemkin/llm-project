"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient, LLMResponse
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, KnowledgeBase

TYPE_KEY = "type"
NAME_KEY = "name"
SOURCES_KEY = "sources"

PART_TYPE = "Part"
REPLACED_BY_RELATION = "REPLACED_BY"

ROLE_KEY = "role"
CONTENT_KEY = "content"
SYSTEM_ROLE = "system"
USER_ROLE = "user"

MAX_FACTS_CHARS = 8000
REFUSAL_TEXT = "В базе знаний не найдено."
REFUSAL_ANSWER = Answer(REFUSAL_TEXT, refused=True)

def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """

    graph = kb.graph
    anchors = _find_anchors(question, kb)
    # FR-4: Неизвестный код или отсутствие опор означает отказ без вызова модели.
    if len(anchors) == 0:
        return REFUSAL_ANSWER

    groups = _local_fact_groups(graph, anchors)
    facts, edges = _pack_facts(graph, groups)
    # FR-4: Если фактов не нашлось, сразу возвращаем Answer(..., refused=True), без вызова модели.
    if len(facts) == 0:
        return REFUSAL_ANSWER

    context = _build_fact_context(facts)
    response = _ask_local_model(question, context, llm)
    sources = _collect_fact_sources(graph, edges)

    text = response.content.strip()
    if _is_refusal(text):
        return Answer(text or REFUSAL_TEXT, sources=sources, facts=facts, refused=True)
    return Answer(text, sources=sources, facts=facts)

def _find_anchors(question: str, kb: KnowledgeBase) -> list[str]:
    # FR-4: Опорные сущности ищутся кодом: коды через find_codes, названия через find_names.
    graph = kb.graph
    code_nodes = set()
    for code in find_codes(question):
        node_type = code_type(code)
        node_id = f"{node_type}:{code}"
        code_nodes.add(node_id)

    # FR-4: Неизвестный код прерывает поиск опор; local_search вернёт отказ без вызова модели.
    for node_id in code_nodes:
        if node_id not in graph:
            return list()

    anchor_nodes = set(code_nodes)
    for node_type, name in find_names(question, kb.vocabulary):
        node_id = f"{node_type}:{name}"
        if node_id in graph:
            anchor_nodes.add(node_id)
    return sorted(anchor_nodes)


def _build_fact_context(facts: list[str]) -> str:
    # FR-4: Модель получает пронумерованный список отобранных фактов.
    numbered_facts = []
    for number, fact in enumerate(facts, start=1):
        numbered_facts.append(f"{number}. {fact}")
    return "\n".join(numbered_facts)


def _ask_local_model(question: str, context: str, llm: LLMClient) -> LLMResponse:
    # FR-4: Просим отвечать по фактам и называть последнюю деталь цепочки замен.
    return llm.chat(
        [
            {
                ROLE_KEY: SYSTEM_ROLE,
                CONTENT_KEY: (
                    "Ты ассистент сервисной службы. Ответь по-русски только по фактам ниже. "
                    "Сохраняй точные коды и направление связей. REPLACED_BY означает замену старой "
                    "детали новой: если есть цепочка замен, актуальна последняя деталь цепочки. "
                    "При цикле или неоднозначной замене не выбирай деталь наугад. "
                    "Не придумывай цены, остатки и другие отсутствующие факты. "
                    "Если фактов для ответа недостаточно, напиши: «в базе знаний не найдено»."
                ),
            },
            {ROLE_KEY: USER_ROLE, CONTENT_KEY: f"Факты:\n{context}\n\nВопрос: {question}"},
        ]
    )


def _collect_fact_sources(graph: nx.DiGraph, edges: list[tuple[str, str]]) -> list[str]:
    source_documents = set()
    for source_node, target_node in edges:
        edge_data = graph.edges[source_node, target_node]
        for chunk_id in edge_data.get(SOURCES_KEY, []):
            document_id = chunk_id.split("#", 1)[0]
            source_documents.add(document_id)
    return sorted(source_documents)

def _local_fact_groups(graph: nx.DiGraph, anchors: list[str]) -> list[set[tuple[str, str]]]:
    # FR-4: При поиске пути направление связей игнорируется.
    undirected = graph.to_undirected(as_view=True)

    # FR-4: Для одной опорной сущности берётся её окрестность на глубину до двух связей.
    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        neighborhood = graph.subgraph(nearby)
        groups = []
        for edge in neighborhood.edges:
            groups.append(_with_replacements(neighborhood, {edge}))
        return groups

    groups = []
    # FR-4: Для двух и более опор берутся факты на всех путях длиной до трёх связей между ними.
    for source, target in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, source, target, cutoff=3):
            edges = set()
            for first_node, second_node in pairwise(path):
                if graph.has_edge(first_node, second_node):
                    edges.add((first_node, second_node))
                if graph.has_edge(second_node, first_node):
                    edges.add((second_node, first_node))

            # FR-4: К путям добавляются цепочки REPLACED_BY от деталей на этих путях.
            groups.append(_with_replacements(graph, edges))
    return groups


def _with_replacements(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    # Поиск деталей, с которых начинается обход замен
    selected = set(edges)
    pending = set()
    for source_node, target_node in edges:
        for node in (source_node, target_node):
            if graph.nodes[node].get(TYPE_KEY) == PART_TYPE:
                pending.add(node)

    visited = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        for _, target, attrs in graph.out_edges(node, data=True):
            if attrs.get(TYPE_KEY) == REPLACED_BY_RELATION:
                selected.add((node, target))
                if target not in visited:
                    pending.add(target)
    return selected

def _pack_facts(graph: nx.DiGraph, groups: list[set[tuple[str, str]]]) -> tuple[list[str], list[tuple[str, str]]]:
    facts: list[str] = []
    edges: list[tuple[str, str]] = []
    selected: set[tuple[str, str]] = set()
    length = 0

    unique_groups = set()
    for group in groups:
        unique_groups.add(frozenset(group))

    def group_sort_key(group):
        replacement_priority = 1
        for source_node, target_node in group:
            if graph.edges[source_node, target_node].get(TYPE_KEY) == REPLACED_BY_RELATION:
                replacement_priority = 0
                break
        return replacement_priority, len(group), sorted(group)

    # Сначала пути с заменами. Каждый путь с полной цепочкой добавляется целиком.
    ordered = sorted(unique_groups, key=group_sort_key)
    for group in ordered:
        new_edges = []
        for edge in group:
            if edge not in selected:
                new_edges.append(edge)
        new_edges.sort()

        # FR-4: Формат факта: E-17 - INDICATES -> пусковое реле.
        new_facts = []
        for source_node, target_node in new_edges:
            source_name = graph.nodes[source_node][NAME_KEY]
            target_name = graph.nodes[target_node][NAME_KEY]
            relation_type = graph.edges[source_node, target_node][TYPE_KEY]
            new_facts.append(f"{source_name} - {relation_type} -> {target_name}")

        # FR-4: Список фактов ограничен 8000 символами, включая нумерацию и переносы строк.
        extra = 0
        for offset, fact in enumerate(new_facts, start=1):
            number = len(facts) + offset
            numbered_fact = f"{number}. {fact}"
            extra += len(numbered_fact) + 1
        if length + extra > MAX_FACTS_CHARS:
            continue

        facts.extend(new_facts)
        edges.extend(new_edges)
        selected.update(new_edges)
        length += extra
    return facts, edges

def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials, sources = _collect_partials(question, kb, llm)

    # FR-5: Если полезных частичных ответов нет, возвращаем Answer(..., refused=True).
    if len(partials) == 0:
        return REFUSAL_ANSWER

    response = _reduce_partials(question, partials, llm)
    text = response.content.strip()
    return Answer(text or REFUSAL_TEXT, sources=sorted(sources), refused=_is_refusal(text))

def _collect_partials(question: str, kb: KnowledgeBase, llm: LLMClient) -> tuple[list[str], set[str]]:
    partials: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        response = _map_community(question, community, llm)
        # FR-5: Нечитаемый частичный ответ пропускается.
        partial = _parse_partial(response.content)
        if partial is None:
            continue
        answer, score = partial
        partials.append(f"Сообщество {community.community_id}, полезность {score}/10:\n{answer}")
        sources.update(_community_sources(community, kb))
    return partials, sources

def _map_community(question: str, community: Community, llm: LLMClient) -> LLMResponse:
    # FR-5, map: Для каждой сводки модель возвращает частичный ответ по схеме MAP_SCHEMA.
    return llm.chat(
        [
            {
                ROLE_KEY: SYSTEM_ROLE,
                CONTENT_KEY: (
                    "Ответь на вопрос только по сводке сообщества ниже. "
                    "Верни JSON с полями answer (частичный ответ) и score "
                    "(целая оценка полезности от 0 до 10). "
                    "Если сводка не содержит ответа, верни пустой answer и score 0. "
                    "Не добавляй факты и числа, которых нет в сводке."
                ),
            },
            {
                ROLE_KEY: USER_ROLE,
                CONTENT_KEY: (
                    f"Сводка сообщества {community.community_id}:\n{community.summary}\n\nВопрос: {question}"
                ),
            },
        ],
        # FR-5: JSON имеет вид {"answer": "...", "score": 7}.
        json_schema=MAP_SCHEMA,
    )

def _reduce_partials(question: str, partials: list[str], llm: LLMClient) -> LLMResponse:
    # FR-5, reduce: Частичные ответы с оценкой выше 0 объединяются одним вызовом модели.
    return llm.chat(
        [
            {
                ROLE_KEY: SYSTEM_ROLE,
                CONTENT_KEY: (
                    "Объедини полезные частичные ответы в ответ на вопрос. "
                    "Используй только факты из частичных ответов, не добавляй свои знания. "
                    "C  охраняй точные коды и числа. "
                    "Если данных для ответа нет, напиши: \"в базе знаний не найдено\"."
                ),
            },
            {ROLE_KEY: USER_ROLE, CONTENT_KEY: "\n\n".join(partials) + f"\n\nВопрос: {question}"},
        ]
    )

def _parse_partial(content: str) -> tuple[str, int] | None:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    answer, score = data.get("answer"), data.get("score")
    if not isinstance(answer, str) or type(score) is not int or not 0 <= score <= 10:
        return None
    answer = answer.strip()
    # FR-5: Для reduce оставляем только частичные ответы с оценкой выше 0.
    return (answer, score) if answer and score > 0 else None

def _community_sources(community: Community, kb: KnowledgeBase) -> set[str]:
    subgraph = kb.graph.subgraph(community.nodes)
    chunk_ids: set[str] = set()
    for _, attrs in subgraph.nodes(data=True):
        chunk_ids.update(attrs.get(SOURCES_KEY, []))
    for _, _, attrs in subgraph.edges(data=True):
        chunk_ids.update(attrs.get(SOURCES_KEY, []))
    documents = set()
    for chunk_id in chunk_ids:
        document_id = chunk_id.split("#", 1)[0]
        documents.add(document_id)
    return documents

def _is_refusal(text: str) -> bool:
    if text == "":
        return True

    normalized_text = " ".join(text.casefold().split())
    return "в базе знаний не найдено" in normalized_text
