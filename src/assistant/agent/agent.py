"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from decimal import Decimal
from inspect import signature

import httpx

from assistant.agent import tools as tool_module
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM_PROMPT = (
    "Ты ассистент сервисной службы «ПолярТех». Отвечай по-русски, опирайся на вопрос "
    "и результаты инструментов. Для поиска детали по ошибке или модели используй search_knowledge, "
    "для обзора по базе знаний — summarize_topic. Цены и наличие проверяй в ERP. "
    "Различай нулевой остаток, отсутствие данных о детали, отклонение запроса и недоступность ERP. "
    "Для заказа используй только количество, указанное пользователем. "
    "Если данных в базе знаний нет, так и скажи: «В базе знаний не найдено»."
)
_NO_DATA = "В базе знаний не найдено."
_ERP_FAILURE = "Данные ERP получить не удалось."
_STEP_LIMIT = "Работа прервана: достигнут лимит вызовов инструментов."
_ASK_QUANTITY = "Сколько штук нужно заказать?"
_ORDER_UNCERTAIN = "ERP не подтвердила результат. Проверьте статус заказа перед повторной отправкой."
_WAREHOUSES = {
    "Москва": r"москв(?:[аеуы]|ой)",
    "Санкт-Петербург": r"санкт-петербург(?:[еау]|ом)?",
    "Казань": r"казан(?:ь|и|ью)",
    "Новосибирск": r"новосибирск(?:[еау]|ом)?",
}
_NUMBER_WORDS = {
    "ноль": 0,
    "одну": 1,
    "одна": 1,
    "один": 1,
    "две": 2,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}
_NUMBER = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
_QUANTITY_UNIT = r"(?:шт\b\.?|штук(?:а|и|у)?\b|единиц(?:а|ы|у)?\b)"
_SIGNED_NUMBER = rf"[+-]?{_NUMBER}(?:[.,]\d+)?"
_PRICE_LABEL = r"цена|стоит|стоимость|обойд[её]тся в"
_STOCK_LABEL = r"остаток|в наличии|осталось|на складе(?: в [\w-]+)? есть"
_ORDER_COUNT_LABEL = r"заказано|заказали|оформлено"
_ORDER_REQUEST = re.compile(r"\b(?:закаж\w*|заказать|заказ\w*|оформ\w*|куп\w*)\b", re.IGNORECASE)
_ORDER_PROHIBITION = re.compile(
    r"\bне\s+(?:(?:нужно|надо|следует|хочу|стоит)\s+)?(?:закаж\w*|заказ\w*|оформ\w*|покуп\w*|куп\w*)"
    r"|\b(?:заказ\w*|оформ\w*)\s+не\s+(?:нужно|надо|следует|стоит)\b",
    re.IGNORECASE,
)
_ORDER_WORD = rf"(?:{'|'.join(_NUMBER_WORDS)}|тысяч[аиу]?|миллион\w*)\b"
_WORD_AMOUNT = rf"(?:минус\s+)?{_ORDER_WORD}(?:\s+{_ORDER_WORD})*"
_ORDER_AMOUNT = rf"{_NUMBER}|{_WORD_AMOUNT}"
_FACT_AMOUNT = rf"{_SIGNED_NUMBER}|{_WORD_AMOUNT}"
_AMOUNT_WITH_UNIT = re.compile(
    rf"(?<![\w-])(?<!\d[.,])({_FACT_AMOUNT})\s*(руб(?:лей|ля|ль)?\b\.?|р\b\.?|₽|{_QUANTITY_UNIT})",
    re.IGNORECASE,
)
_AMOUNT_WITH_LABEL = re.compile(
    rf"\b({_PRICE_LABEL}|{_STOCK_LABEL}|{_ORDER_COUNT_LABEL})"
    rf"(?:\s*[:=–—]\s*|\s+)({_FACT_AMOUNT})(?![\w-]|[.,]\d)",
    re.IGNORECASE,
)
_ORDER_QUANTITY = re.compile(
    rf"(?<![\w−-])(?<!\d[.,])({_ORDER_AMOUNT})\s*{_QUANTITY_UNIT}"
    rf"|\b(?:закажи(?:те)?|заказать|заказ на|количество\s*[:=])\s+"
    rf"({_ORDER_AMOUNT})(?![\w-]|[.,]\d)",
    re.IGNORECASE,
)
_SENTENCE_END = r"(?<!\bшт)(?<!\bруб)[.!?]+(?:\s+|$)"
_CLAUSE_END = rf"{_SENTENCE_END}|[\n;]|,(?!\d)|\s+и\s+"
_STOCK_CONTEXT = re.compile(r"\b(?:остат\w*|остал\w*|в наличии|на складе есть)\b", re.IGNORECASE)
_PRICE_QUERY = re.compile(r"\b(?:цен[ауые]|ценой|стоит|стоят|стоимост\w*|руб\w*|поч[её]м)\b|₽", re.IGNORECASE)
_STOCK_QUERY = re.compile(r"\b(?:остат\w*|остал\w*|наличи\w*|штук\w*)\b", re.IGNORECASE)
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_STATUS_QUERY = re.compile(r"\b(?:что с|статус\w*|состояни\w*|(?:создан|оформлен|принят) ли)\b", re.IGNORECASE)
_ORDER_SUBJECT = r"\b(?:заказ\w*|SO-\d+)\b"
_CREATED_STATUS = (
    r"(?<!\bне )\b(?:(?:создан|оформлен|зарегистрирован|принят)[аоы]?"
    r"|(?:созда|оформи|зарегистрирова|приня)л[аои]?)\b"
)
_ORDER_CREATION = re.compile(
    rf"{_ORDER_SUBJECT}.*{_CREATED_STATUS}|{_CREATED_STATUS}.*{_ORDER_SUBJECT}|(?<!\bне )\bзаказано\b",
    re.IGNORECASE,
)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью и проверяет вызовы перед передачей инструментам."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": question}]
    result = AgentResult(answer="")
    known_codes = set(find_codes(question))
    quantities = _requested_quantities(question)
    cached: dict[tuple[str, str], dict] = {}

    while len(result.steps) < max_steps:
        reply = llm.chat(messages, tools=tool_module.TOOL_SCHEMAS)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            return _finish(result, reply.content, messages, llm, question, known_codes)

        for call in reply.tool_calls:
            if len(result.steps) >= max_steps:
                messages.append(tool_message(call, {"error": _STEP_LIMIT}))
                continue

            tool = tools.get(call.name)
            error = _call_error(call, tool, known_codes)
            if error is None and call.name == "create_service_order":
                if _ORDER_PROHIBITION.search(question):
                    result.answer = "Заказ не отправлен."
                    result.steps.append(ToolStep(call.name, call.arguments, error=result.answer))
                    return result
                args = call.arguments
                models = {code for code in find_codes(question) if code_type(code) == "Model"}
                warehouses = _warehouses_in(question)
                missing = []
                if len(models) != 1:
                    missing.append("модель оборудования")
                if len(warehouses) != 1:
                    missing.append("склад")
                if missing:
                    result.answer = f"Уточните {' и '.join(missing)} для заказа."
                    result.steps.append(ToolStep(call.name, args, error=result.answer))
                    return result
                if args["model"] not in models or args["warehouse"] not in warehouses:
                    error = "Используйте модель и склад, указанные пользователем."
                elif type(args.get("qty")) is not int or args["qty"] <= 0:
                    error = "Количество должно быть положительным целым числом."
                elif quantities.get(normalize_code(args["part_id"])) != {args["qty"]} and (
                    quantities.get(None) != {args["qty"]} or len(quantities) != 1
                ):
                    result.steps.append(ToolStep(call.name, args, error=_ASK_QUANTITY))
                    result.answer = _ASK_QUANTITY
                    return result
                elif not confirm:
                    result.steps.append(ToolStep(call.name, args, error="Требуется подтверждение."))
                    result.needs_confirmation = True
                    result.pending_action = {"tool": call.name, "args": args}
                    result.answer = (
                        f"Подтвердите заказ детали {args['part_id']} "
                        f"для {args['model']} в количестве {args['qty']} шт. для склада {args['warehouse']}."
                    )
                    return result

            if error is None:
                key = (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))
                if key not in cached:
                    cached[key] = _execute(call, tool)
                observation = cached[key]
            else:
                observation = {"error": error}

            result.steps.append(
                ToolStep(call.name, call.arguments, observation if error is None else None, observation.get("error"))
            )
            if observation.get("error") == _ORDER_UNCERTAIN:
                result.answer = _ORDER_UNCERTAIN
                return result
            if "error" not in observation and not observation.get("refused"):
                for source in observation.get("sources", []):
                    if source not in result.sources:
                        result.sources.append(source)
                if call.name == "create_service_order":
                    order_id = observation["order_id"]
                    status = observation.get("status", "created")
                    if status == "created":
                        result.answer = f"Заказ {order_id} создан."
                        if "qty" in observation:
                            result.answer += f" Заказано {observation['qty']} шт."
                    else:
                        result.answer = f"Заказ {order_id}. Статус ERP: {status}."
                    if any(step.error == _ERP_FAILURE for step in result.steps):
                        result.answer += f"\n{_ERP_FAILURE}"
                    return result
                known_codes.update(find_codes(json.dumps(observation, ensure_ascii=False)))
            messages.append(tool_message(call, observation))

    messages.append({"role": "user", "content": f"{_STEP_LIMIT} Подведи итог по полученным данным."})
    reply = llm.chat(messages)
    messages.append({"role": "assistant", "content": reply.content})
    result = _finish(result, reply.content, messages, llm, question, known_codes)
    result.answer = f"{result.answer}\n{_STEP_LIMIT}"
    return result


def _call_error(call: ToolCall, tool: Callable[..., dict] | None, known_codes: set[str]) -> str | None:
    if tool is None:
        return "Неизвестный инструмент."
    args = call.arguments
    if not isinstance(args, dict) or "_raw" in args:
        return "Аргументы инструмента должны быть объектом JSON."
    try:
        signature(tool).bind(**args)
    except TypeError:
        return "Неверный набор аргументов инструмента."

    if call.name in {"get_price", "check_stock", "create_service_order"} or "part_id" in args:
        part = args.get("part_id")
        code = normalize_code(part) if isinstance(part, str) else None
        if not code or not code.startswith("P-") or code not in known_codes:
            return "Номер детали не подтверждён. Уточните его у пользователя или вызовите search_knowledge."

    if call.name == "create_service_order":
        for name in ("warehouse", "model"):
            if not isinstance(args.get(name), str) or not args[name].strip():
                return "Для заказа укажите модель и склад."
        if args["warehouse"] not in _WAREHOUSES:
            return f"Неизвестный склад. Допустимые склады: {', '.join(_WAREHOUSES)}."
    return None


def _number_in_words(raw: str) -> int | None:
    words = raw.lower().split()
    sign = -1 if words[0] == "минус" else 1
    if sign == -1:
        words = words[1:]
    if not words or any(word not in _NUMBER_WORDS for word in words):
        return None
    values = [_NUMBER_WORDS[word] for word in words]
    hundreds = values.pop(0) if values[0] >= 100 else 0
    if not values:
        return sign * hundreds
    if len(values) == 1 and values[0] < 100:
        return sign * (hundreds + values[0])
    if len(values) == 2 and 20 <= values[0] < 100 and 1 <= values[1] <= 9:
        return sign * (hundreds + sum(values))
    return None


def _requested_quantities(question: str) -> dict[str | None, set[int]]:
    parts = {code for code in find_codes(question) if code.startswith("P-")}
    default_part = next(iter(parts)) if len(parts) == 1 else None
    quantities: dict[str | None, set[int]] = {}
    for sentence in re.split(rf"{_SENTENCE_END}|[;\n]", question):
        request = _ORDER_REQUEST.search(sentence)
        if request is None:
            continue
        start = request.start() if _STOCK_CONTEXT.search(sentence, 0, request.start()) else 0
        following_stock = _STOCK_CONTEXT.search(sentence, request.end())
        order = sentence[start : following_stock.start() if following_stock else len(sentence)]
        for clause in re.split(_CLAUSE_END, order) if len(parts) > 1 else [order]:
            part, _ = _scope(clause)
            if part is None and len(parts) > 1:
                continue
            for first, second in _ORDER_QUANTITY.findall(clause):
                raw = first or second
                digits = re.sub(r"\s", "", raw)
                value = int(digits) if digits.isdecimal() else _number_in_words(raw)
                if value is not None:
                    quantities.setdefault(part or default_part, set()).add(value)
    return quantities


def _execute(call: ToolCall, tool: Callable[..., dict]) -> dict:
    for _ in range(3):
        try:
            observation = tool(**call.arguments)
            if call.name == "create_service_order" and "error" not in observation and not observation.get("order_id"):
                return {"error": _ORDER_UNCERTAIN}
            return observation
        except ErpUnavailable as exc:
            # The bundled ERP injects 503 before dispatching to the order handler.
            before_dispatch = isinstance(exc.__cause__, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            if call.name == "create_service_order" and not (before_dispatch or str(exc) == "ERP недоступна: код 503"):
                return {"error": _ORDER_UNCERTAIN}
    return {"error": _ERP_FAILURE}


def _warehouses_in(text: str) -> set[str]:
    return {name for name, pattern in _WAREHOUSES.items() if re.search(rf"\b{pattern}\b", text, re.IGNORECASE)}


def _scope(text: str) -> tuple[str | None, str | None]:
    parts = [code for code in find_codes(text) if code.startswith("P-")]
    warehouses = _warehouses_in(text)
    return (parts[0] if len(parts) == 1 else None, next(iter(warehouses)) if len(warehouses) == 1 else None)


def _amount_value(number: str) -> Decimal | None:
    if any(char.isalpha() for char in number):
        value = _number_in_words(number)
        return Decimal(value) if value is not None else None
    return Decimal(re.sub(r"\s", "", number).replace(",", "."))


def _amounts(text: str) -> set[tuple[str, Decimal | None, str | None, str | None]]:
    whole_part, whole_warehouse = _scope(text)
    amounts = set()
    for clause in re.split(_CLAUSE_END, text):
        part, warehouse = _scope(clause)
        quantity_kind = "order_qty" if _ORDER_REQUEST.search(clause) and not _STOCK_CONTEXT.search(clause) else "stock"
        numbers = [
            (number, "price" if unit.lower().startswith(("р", "₽")) else quantity_kind)
            for number, unit in _AMOUNT_WITH_UNIT.findall(clause)
        ]
        for label, number in _AMOUNT_WITH_LABEL.findall(clause):
            if re.fullmatch(_PRICE_LABEL, label, re.IGNORECASE):
                kind = "price"
            elif re.fullmatch(_ORDER_COUNT_LABEL, label, re.IGNORECASE):
                kind = "order_qty"
            else:
                kind = "stock"
            numbers.append((number, kind))
        for number, kind in numbers:
            location = (warehouse or whole_warehouse) if kind != "price" else None
            amounts.add((kind, _amount_value(number), part or whole_part, location))
    return amounts


def _finish(
    result: AgentResult,
    answer: str,
    messages: list[dict],
    llm: LLMClient,
    question: str,
    known_codes: set[str],
) -> AgentResult:
    observations = [step for step in result.steps if step.result is not None and step.error is None]
    evidence = [step for step in observations if not step.result.get("refused")]
    allowed_amounts = _amounts(question)
    for step in evidence:
        data = step.result
        if step.tool in {"search_knowledge", "summarize_topic"}:
            allowed_amounts.update(_amounts(data.get("answer", "")))
        if step.tool == "get_price" and type(data.get("price_rub")) is int:
            part = normalize_code(data.get("part_id", step.args.get("part_id", "")))
            allowed_amounts.add(("price", Decimal(data["price_rub"]), part, None))
        if step.tool == "check_stock" and type(data.get("qty")) is int:
            part = normalize_code(data.get("part_id", step.args.get("part_id", "")))
            warehouse = data.get("warehouse", step.args.get("warehouse"))
            allowed_amounts.add(("stock", Decimal(data["qty"]), part, warehouse))

    missing_order_status = bool(_ORDER_ID.search(question) and _STATUS_QUERY.search(question))

    def is_grounded(text: str) -> bool:
        claims = _amounts(text)
        number = text.strip().removesuffix(".")
        if re.fullmatch(_FACT_AMOUNT, number, re.IGNORECASE):
            kinds = {
                kind for kind, query in (("price", _PRICE_QUERY), ("stock", _STOCK_QUERY)) if query.search(question)
            }
            if not kinds and re.search(r"\bсклад\w*\b", question, re.IGNORECASE):
                kinds.add("stock")
            if not kinds:
                kinds = {kind for kind, _, _, _ in allowed_amounts if kind in {"price", "stock"}}
            if len(kinds) > 1:
                return False  # Для неоднозначного ответа модель должна назвать единицу измерения.
            if kinds:
                kind = kinds.pop()
                part, warehouse = _scope(question)
                claims.add((kind, _amount_value(number), part, warehouse if kind == "stock" else None))
        return (
            set(find_codes(text)) <= known_codes
            and all(
                value is not None
                and any(
                    (kind, value) == (source_kind, source_value)
                    and (part is None or part == source_part)
                    and (warehouse is None or warehouse == source_warehouse)
                    for source_kind, source_value, source_part, source_warehouse in allowed_amounts
                )
                for kind, value, part, warehouse in claims
            )
            and not _ORDER_ID.search(text)
            and not any(_ORDER_CREATION.search(sentence) for sentence in re.split(_SENTENCE_END, text))
        )

    if not evidence and result.steps and all(step.error for step in result.steps):
        error = result.steps[-1].error
        if error.startswith("Номер детали не подтверждён"):
            answer = "Номер детали мне неизвестен."
        elif result.steps[-1].result is None:
            answer = "Не получилось."
        else:
            answer = error
        result.refused = True
        if any(step.error == _ERP_FAILURE for step in result.steps) and _ERP_FAILURE not in answer:
            answer += f"\n{_ERP_FAILURE}"
        result.answer = answer
        return result

    if not is_grounded(answer):
        messages.append(
            {
                "role": "user",
                "content": "Исправь ответ: в нём есть коды, цены или остатки без подтверждения. "
                "Используй только вопрос и результаты выполненных инструментов. Запиши количества и цены цифрами. "
                "Сообщай о создании заказа только при наличии номера из ERP.",
            }
        )
        answer = llm.chat(messages).content

    if (
        not evidence
        or not answer.strip()
        or not is_grounded(answer)
        or missing_order_status
    ):
        answer = _NO_DATA
        result.refused = True
    elif "в базе знаний не найдено" in answer.lower():
        result.refused = True

    lines = [answer] if answer else []
    if any(step.error == _ERP_FAILURE for step in result.steps) and _ERP_FAILURE.casefold() not in answer.casefold():
        lines.append(_ERP_FAILURE)

    result.answer = "\n".join(lines)
    return result
