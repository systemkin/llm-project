import httpx
import pytest

from assistant.agent.agent import run_agent
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

ORDER = dict(part_id="P-1042B", model="AR-210", warehouse="Казань", qty=2)
QUESTION = "Закажи 2 шт. P-1042B для AR-210 в Казани."


def call(name, args):
    return LLMResponse("", [ToolCall("1", name, args)])


@pytest.mark.parametrize("confirm", [False, True])
def test_explicit_order_prohibition(confirm):
    llm = FakeLLM([call("create_service_order", ORDER)])
    result = run_agent("Не заказывай 2 шт. P-1042B для AR-210 в Казани.", llm,
                       {"create_service_order": lambda **kw: pytest.fail("unexpected order")}, confirm=confirm)
    assert result.pending_action is None and not result.needs_confirmation


def test_lost_order_response():
    attempts = []

    def create(**kw):
        attempts.append(kw)
        if len(attempts) == 1:
            raise ErpUnavailable("timeout") from httpx.ReadTimeout("response lost")
        return {"order_id": "SO-2"}

    result = run_agent(QUESTION, FakeLLM([call("create_service_order", ORDER)]),
                       {"create_service_order": create}, confirm=True)
    assert len(attempts) == 1 and "Проверьте статус" in result.answer


def test_price_abbreviation():
    llm = FakeLLM([call("get_price", {"part_id": "P-1042B"}), "К оплате 9999 р.", "К оплате 2630 р."])
    result = run_agent("Сколько стоит P-1042B?", llm, {"get_price": lambda **kw: {"price_rub": 2630}})
    assert result.answer == "К оплате 2630 р." and len(llm.requests) == 3


def test_no_basis_for_repair_advice():
    result = run_agent("Как починить компрессор?", FakeLLM(["Замените пусковое реле."]), {})
    assert result.refused and result.answer == "В базе знаний не найдено."


def test_order_retries_after_erp_rejects_before_dispatch():
    attempts = []

    def create(**kw):
        attempts.append(kw)
        if len(attempts) < 3:
            raise ErpUnavailable("ERP недоступна: код 503")
        return {"order_id": "SO-1"}

    result = run_agent(QUESTION, FakeLLM([call("create_service_order", ORDER)]),
                       {"create_service_order": create}, confirm=True)
    assert len(attempts) == 3 and "SO-1" in result.answer
