"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_ID_PATTERN = r"^P-[0-9]{4}[A-Z]?$"

PART_ID_SCHEMA = {
    "type": "string",
    "pattern": PART_ID_PATTERN,
    "description": "Точный номер детали, например P-1042 или P-1042B.",
}
WAREHOUSE_SCHEMA = {"type": "string", "enum": WAREHOUSES, "description": "Название склада."}
QUESTION_SCHEMA = {"type": "string", "minLength": 1, "description": "Вопрос к базе знаний."}

# FR-6: Пять инструментов в формате function calling с обязательными параметрами.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Найти факты о конкретных моделях, деталях, ошибках и регламентах в графе знаний.",
            "parameters": {
                "type": "object",
                "properties": {"question": QUESTION_SCHEMA},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Подготовить обзор по сводкам сообществ: общие закономерности и частые отказы.",
            "parameters": {
                "type": "object",
                "properties": {"question": QUESTION_SCHEMA},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Получить из ERP остаток детали на выбранном складе и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": PART_ID_SCHEMA, "warehouse": WAREHOUSE_SCHEMA},
                "required": ["part_id", "warehouse"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить из ERP цену детали в рублях и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": PART_ID_SCHEMA},
                "required": ["part_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать сервисный заказ на деталь для указанной модели и вернуть номер заказа.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": PART_ID_SCHEMA,
                    "warehouse": WAREHOUSE_SCHEMA,
                    "qty": {"type": "integer", "minimum": 1, "description": "Количество деталей в заказе."},
                    "model": {"type": "string", "minLength": 1, "description": "Код модели, например AR-210."},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        error = _validate_arguments([(_validate_question, question)])
        if error is not None:
            return error

        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        error = _validate_arguments([(_validate_question, question)])
        if error is not None:
            return error

        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        # FR-6: Все параметры проверяются до обращения к ERP.
        error = _validate_arguments([
            (_validate_part_id, part_id),
            (_validate_warehouse, warehouse),
        ])
        if error is not None:
            return error

        return _call_erp(erp.get_stock, part_id, warehouse)

    def get_price(part_id: str) -> dict:
        # FR-6: Все параметры проверяются до обращения к ERP.
        error = _validate_arguments([(_validate_part_id, part_id)])
        if error is not None:
            return error

        return _call_erp(erp.get_price, part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        # FR-6: Все параметры проверяются до обращения к ERP.
        error = _validate_arguments([
            (_validate_part_id, part_id),
            (_validate_warehouse, warehouse),
            (_validate_qty, qty),
            (_validate_model, model),
        ])
        if error is not None:
            return error

        return _call_erp(erp.create_order, part_id, warehouse, qty, model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


def _validate_arguments(validations: list) -> dict | None:
    for validator, value in validations:
        error = validator(value)
        if error is not None:
            return {"error": error}
    return None


def _validate_question(question: str) -> str | None:
    if not _is_nonempty_string(question):
        return "Вопрос должен быть непустой строкой."
    return None


def _is_nonempty_string(value: str) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _validate_part_id(part_id: str) -> str | None:
    if not isinstance(part_id, str) or re.fullmatch(PART_ID_PATTERN, part_id) is None:
        return f"Номер детали должен соответствовать шаблону {PART_ID_PATTERN}."
    return None


def _validate_warehouse(warehouse: str) -> str | None:
    if not isinstance(warehouse, str) or warehouse not in WAREHOUSES:
        return "Неизвестный склад. Допустимые склады: " + ", ".join(WAREHOUSES) + "."
    return None


def _validate_qty(qty: int) -> str | None:
    if type(qty) is not int or qty <= 0:
        return "Количество должно быть целым числом больше нуля."
    return None


def _validate_model(model: str) -> str | None:
    if not _is_nonempty_string(model):
        return "Модель должна быть непустой строкой, например AR-210."
    return None


def _call_erp(method: Callable[..., dict], *args) -> dict:
    try:
        return method(*args)
    except ErpNotFound as exc:
        return {"error": f"Данных в ERP нет: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}
