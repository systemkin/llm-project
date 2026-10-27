"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from dataclasses import dataclass, field

import networkx as nx

from assistant.infra.schemas import ExtractionResult


@dataclass
class NodeAccumulator:
    """Накопитель данных по узлу"""
    type: str
    name: str
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RelationKey:
    """Ключ ребра: пара узлов + тип связи."""
    subject: str
    type: str
    object: str

    def reversed(self) -> "RelationKey":
        return RelationKey(self.object, self.type, self.subject)


@dataclass
class EdgeAccumulator:
    """Накопитель данных по ребру"""
    sources: set[str] = field(default_factory=set)


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # 1. Накопление.
    #   nodes: node_id -> NodeAccumulator
    #   edges: RelationKey -> EdgeAccumulator  (тип связи уже внутри ключа)
    nodes: dict[str, NodeAccumulator] = {}
    edges: dict[RelationKey, EdgeAccumulator] = {}

    for result in results:
        for entity in result.entities:
            node = nodes.setdefault(
                entity.node_id,
                NodeAccumulator(type=entity.type, name=entity.name),
            )
            node.sources.add(result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue
            key = RelationKey(
                subject=relation.subject.node_id,
                type=relation.type,
                object=relation.object.node_id,
            )
            edges.setdefault(key, EdgeAccumulator()).sources.add(result.chunk_id)

    # 2. Построение графа.
    graph = nx.DiGraph()
    for node_id, node in nodes.items():
        graph.add_node(
            node_id,
            type=node.type,
            name=node.name,
            sources=sorted(node.sources),
        )

    for key, edge in edges.items():
        graph.add_edge(
            key.subject,
            key.object,
            type=key.type,
            weight=len(edge.sources),
            sources=sorted(edge.sources),
        )

    # 3. Разбор встречных REPLACED_BY.
    conflicts = 0
    to_remove: list[RelationKey] = []
    for key in edges:
        if key.type != "REPLACED_BY" or key.subject >= key.object:
            continue
        reverse_key = key.reversed()
        if reverse_key not in edges:
            continue

        conflicts += 1
        w_forward = len(edges[key].sources)
        w_backward = len(edges[reverse_key].sources)
        if w_forward > w_backward:
            to_remove.append(reverse_key)
        elif w_backward > w_forward:
            to_remove.append(key)
        else:
            to_remove.extend((key, reverse_key))

    for key in to_remove:
        if graph.has_edge(key.subject, key.object):
            graph.remove_edge(key.subject, key.object)

    graph.graph["conflicts"] = conflicts
    return graph