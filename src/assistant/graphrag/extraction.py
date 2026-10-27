"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

_SYSTEM = (
    "Ты извлекаешь сущности и связи из фрагмента документации сервисной службы.\n"
    "Верни только JSON без пояснений.\n\n"
    "Типы сущностей: ProductLine, Model, Component, Part, ErrorCode, Procedure, Supplier.\n"
    "Коды моделей, деталей, ошибок и регламентов записывай канонически: AR-210, P-1042B, E-17, SP-104.\n"
    "Названия линеек, узлов и поставщиков бери только из словаря ниже.\n\n"
    "Допустимые связи и их направление:\n"
    "Model -BELONGS_TO-> ProductLine\n"
    "Model -HAS_PART-> Part\n"
    "Part -IS_A-> Component\n"
    "ErrorCode -INDICATES-> Component\n"
    "ErrorCode -FIXED_BY-> Procedure\n"
    "Part -REPLACED_BY-> Part\n"
    "Part -SUPPLIED_BY-> Supplier\n"
    "Model -FAILED-> Component\n\n"
    "Пример ответа:\n"
    '{"entities":[{"type":"Model","name":"AR-210"},'
    '{"type":"ProductLine","name":"Арктика"},'
    '{"type":"Part","name":"P-1042B"}],'
    '"relations":['
    '{"subject_type":"Model","subject":"AR-210","type":"BELONGS_TO",'
    '"object_type":"ProductLine","object":"Арктика"},'
    '{"subject_type":"Model","subject":"AR-210","type":"HAS_PART",'
    '"object_type":"Part","object":"P-1042B"}]}\n\n'
)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через ``llm.chat(..., json_schema=EXTRACTION_SCHEMA)``.
    Всё, что не подтверждается текстом чанка и словарём, отбрасывается и
    считается в ``dropped``. При неразбираемом ответе делается одна повторная
    попытка, затем возвращается пустой результат.
    """
    vocab = vocabulary()
    messages = [
        {
            "role": "system",
            "content": _SYSTEM + "Словарь названий:\n" + json.dumps(vocab, ensure_ascii=False),
        },
        {"role": "user", "content": f"Текст чанка:\n{chunk.text}"},
    ]

    for _ in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2048)
        data = _parse_response(response.content)
        if data is not None:
            return _build_result(chunk, data, vocab)

    return ExtractionResult(chunk.chunk_id, chunk.doc_id)


def _parse_response(content: str) -> dict | None:
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("entities"), list):
        return None
    if not isinstance(data.get("relations"), list):
        return None
    return data


def _build_result(
    chunk: Chunk,
    data: dict,
    vocab: dict[str, list[str]],
) -> ExtractionResult:
    codes_in_chunk = set(find_codes(chunk.text))
    text_lower = chunk.text.lower()
    dropped = 0

    entities: dict[str, Entity] = {}

    for raw in data.get("entities", []):
        if not isinstance(raw, dict):
            dropped += 1
            continue
        entity_type = raw.get("type")
        name = raw.get("name")
        if not isinstance(entity_type, str) or not isinstance(name, str):
            dropped += 1
            continue

        canonical = canonical_name(entity_type, name, vocab)
        if canonical is None:
            dropped += 1
            continue

        if entity_type in CODE_TYPES:
            confirmed = canonical in codes_in_chunk
        else:
            confirmed = canonical.lower() in text_lower
        if not confirmed:
            dropped += 1
            continue

        entity = Entity(entity_type, canonical)
        entities.setdefault(entity.node_id, entity)

    relations: list[Relation] = []
    seen_relations: set[tuple[str, str, str]] = set()

    for raw in data.get("relations", []):
        if not isinstance(raw, dict):
            dropped += 1
            continue

        subject_type = raw.get("subject_type")
        subject_name = raw.get("subject")
        relation_type = raw.get("type")
        object_type = raw.get("object_type")
        object_name = raw.get("object")

        if not all(
            isinstance(value, str)
            for value in (subject_type, subject_name, relation_type, object_type, object_name)
        ):
            dropped += 1
            continue

        subject = canonical_name(subject_type, subject_name, vocab)
        obj = canonical_name(object_type, object_name, vocab)
        if subject is None or obj is None:
            dropped += 1
            continue

        subject_entity = Entity(subject_type, subject)
        object_entity = Entity(object_type, obj)

        if (
            subject_entity.node_id not in entities
            or object_entity.node_id not in entities
        ):
            dropped += 1
            continue

        relation = Relation(subject_entity, relation_type, object_entity)
        if not relation.matches_schema():
            dropped += 1
            continue

        key = (relation.subject.node_id, relation.type, relation.object.node_id)
        if key not in seen_relations:
            seen_relations.add(key)
            relations.append(relation)

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(entities.values()),
        relations=relations,
        dropped=dropped,
    )