"""The tasks `adder trial` runs: small repositories with a check that cannot be argued with.

A saving is only a saving if the work still gets done, and "the work got done"
has to be decided by something other than the model that did it. Each task is
a package with failing tests, a prompt (or a sequence of prompts, for the
multi-part tasks that let context build up the way a long session does), and a
set of hidden tests copied in only after the agent has finished, so passing
cannot be achieved by editing the tests it was shown.

Every task also carries a reference solution. `tests/evaluate/replay/
test_trial_tasks.py` applies it and runs the checks: a task whose own solution
fails, or whose starting state already passes, would score noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    files: dict[str, str]
    parts: tuple[str, ...]
    hidden: dict[str, str]
    solution: dict[str, str] = field(default_factory=dict)

    @property
    def multi(self) -> bool:
        return len(self.parts) > 1


_LEDGER_MONEY = '''"""Money amounts."""


class Money:
    def __init__(self, amount):
        self.amount = float(amount)

    def __add__(self, other):
        return Money(self.amount + other.amount)

    def __sub__(self, other):
        return Money(self.amount - other.amount)

    def __eq__(self, other):
        return isinstance(other, Money) and self.amount == other.amount

    def __lt__(self, other):
        return self.amount < other.amount

    def __repr__(self):
        return f"Money({self.amount})"

    def __str__(self):
        return f"{self.amount:.2f}"
'''

_LEDGER_ACCOUNT = '''"""Accounts."""

from ledger.money import Money


class Account:
    def __init__(self, name, opening="0"):
        self.name = name
        self.balance = Money(opening)
        self.entries = []          # (date, description, Money) in posting order

    def post(self, date, description, amount):
        m = Money(amount)
        self.balance = self.balance + m
        self.entries.append((date, description, m))
'''

LEDGER = Task(
    id="ledger",
    files={
        "ledger/__init__.py": "",
        "ledger/money.py": _LEDGER_MONEY,
        "ledger/account.py": _LEDGER_ACCOUNT,
        "ledger/report.py": '"""Statements."""\n',
        "tests/test_money.py": '''from ledger.money import Money


def test_cents_do_not_drift():
    total = Money("0")
    for _ in range(10):
        total = total + Money("0.10")
    assert total == Money("1.00")


def test_rounds_half_up_to_cents():
    assert str(Money("2.675")) == "2.68"
    assert str(Money("-2.675")) == "-2.68"
''',
    },
    parts=(
        "Balances in this ledger drift by fractions of a cent because Money uses "
        "floats. Make Money exact: decimal arithmetic, amounts rounded half-up "
        "(away from zero) to whole cents. tests/test_money.py must pass.",
        "Add Account.transfer(to, amount, date) in ledger/account.py. It moves "
        "`amount` from this account to `to`, posting 'transfer to <name>' here and "
        "'transfer from <name>' there. If this account's balance would go below "
        "zero it must raise InsufficientFunds (define it in ledger/account.py as a "
        "subclass of ValueError) and change neither account. Add tests for it in "
        "tests/test_account.py.",
        "Implement statement(account) in ledger/report.py. It returns a list of "
        "lines, one per entry in posting order, formatted "
        "'<date>  <description>  <amount>  <running balance>' with two spaces "
        "between fields and amounts as Money prints them. The running balance "
        "starts from the balance the account was opened with. Add tests in "
        "tests/test_report.py.",
    ),
    hidden={
        "tests_hidden/test_ledger_hidden.py": '''import pytest

from ledger.account import Account, InsufficientFunds
from ledger.money import Money
from ledger.report import statement


def test_money_exact_and_rounded():
    assert Money("0.1") + Money("0.2") == Money("0.3")
    assert str(Money("1.005")) == "1.01"
    assert str(Money("-0.005")) == "-0.01"
    assert str(Money("7")) == "7.00"


def test_transfer_moves_money():
    a, b = Account("a", "10"), Account("b")
    a.transfer(b, "3.50", "2026-01-02")
    assert a.balance == Money("6.50") and b.balance == Money("3.50")
    assert a.entries[-1][1] == "transfer to b"
    assert b.entries[-1][1] == "transfer from a"


def test_overdraft_changes_nothing():
    a, b = Account("a", "1"), Account("b", "5")
    with pytest.raises(InsufficientFunds):
        a.transfer(b, "1.01", "2026-01-02")
    assert issubclass(InsufficientFunds, ValueError)
    assert a.balance == Money("1") and b.balance == Money("5")
    assert a.entries == [] and b.entries == []


def test_statement_runs_from_the_opening_balance():
    a = Account("a", "100")
    a.post("2026-01-01", "coffee", "-3.20")
    a.post("2026-01-02", "refund", "1.10")
    assert statement(a) == [
        "2026-01-01  coffee  -3.20  96.80",
        "2026-01-02  refund  1.10  97.90",
    ]
''',
    },
    solution={
        "ledger/money.py": '''"""Money amounts."""

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


class Money:
    def __init__(self, amount):
        self.amount = Decimal(str(amount)).quantize(CENT, rounding=ROUND_HALF_UP)

    def __add__(self, other):
        return Money(self.amount + other.amount)

    def __sub__(self, other):
        return Money(self.amount - other.amount)

    def __eq__(self, other):
        return isinstance(other, Money) and self.amount == other.amount

    def __lt__(self, other):
        return self.amount < other.amount

    def __repr__(self):
        return f"Money({self.amount})"

    def __str__(self):
        return f"{self.amount:.2f}"
''',
        "ledger/account.py": '''"""Accounts."""

from ledger.money import Money


class InsufficientFunds(ValueError):
    pass


class Account:
    def __init__(self, name, opening="0"):
        self.name = name
        self.opening = Money(opening)
        self.balance = Money(opening)
        self.entries = []

    def post(self, date, description, amount):
        m = Money(amount)
        self.balance = self.balance + m
        self.entries.append((date, description, m))

    def transfer(self, to, amount, date):
        m = Money(amount)
        if self.balance - m < Money("0"):
            raise InsufficientFunds(self.name)
        self.post(date, f"transfer to {to.name}", str(-m.amount))
        to.post(date, f"transfer from {self.name}", str(m.amount))
''',
        "ledger/report.py": '''"""Statements."""


def statement(account):
    running = account.opening
    out = []
    for date, description, m in account.entries:
        running = running + m
        out.append(f"{date}  {description}  {m}  {running}")
    return out
''',
    },
)


_INV_STORE = '''"""An inventory of stock-keeping units."""


class Inventory:
    def __init__(self):
        self._items = {}            # sku -> (name, qty, unit_price_cents)

    def add(self, sku, name, qty, unit_price_cents):
        self._items[sku] = (name, qty, unit_price_cents)

    def get(self, sku):
        return self._items[sku]

    def page(self, n, size=10):
        """Page `n`, counting from 1, of items sorted by sku."""
        skus = sorted(self._items)
        start = n * size
        return [self._items[s] for s in skus[start:start + size]]

    def value_cents(self):
        return sum(q * p for _, q, p in self._items.values()) // 100
'''

INVENTORY = Task(
    id="inventory",
    files={
        "inventory/__init__.py": "",
        "inventory/store.py": _INV_STORE,
        "tests/test_store.py": '''from inventory.store import Inventory


def _inv(n):
    inv = Inventory()
    for i in range(n):
        inv.add(f"SKU{i:03d}", f"item {i}", 2, 150)
    return inv


def test_first_page_is_page_one():
    assert _inv(25).page(1)[0][0] == "item 0"


def test_lookup_ignores_case():
    inv = _inv(1)
    assert inv.get("sku000")[0] == "item 0"


def test_value_is_in_cents():
    assert _inv(3).value_cents() == 900
''',
    },
    parts=(
        "The inventory package has three bugs that tests/test_store.py exposes: "
        "pagination is off by one, SKU lookup should be case-insensitive, and the "
        "stock value is reported in the wrong unit. Fix all three without changing "
        "the public method names, and make the tests pass.",
    ),
    hidden={
        "tests_hidden/test_store_hidden.py": '''import pytest

from inventory.store import Inventory


def _inv(n):
    inv = Inventory()
    for i in range(n):
        inv.add(f"SKU{i:03d}", f"item {i}", 1, 99)
    return inv


def test_last_page_is_partial():
    assert [x[0] for x in _inv(25).page(3)] == ["item 20", "item 21", "item 22",
                                                "item 23", "item 24"]


def test_page_past_the_end_is_empty():
    assert _inv(5).page(2) == []


def test_mixed_case_sku_on_add_and_get():
    inv = Inventory()
    inv.add("AbC", "x", 1, 1)
    assert inv.get("aBc")[0] == "x"
    with pytest.raises(KeyError):
        inv.get("abd")


def test_value_sums_exactly():
    assert _inv(3).value_cents() == 297
''',
    },
    solution={
        "inventory/store.py": '''"""An inventory of stock-keeping units."""


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, sku, name, qty, unit_price_cents):
        self._items[sku.upper()] = (name, qty, unit_price_cents)

    def get(self, sku):
        return self._items[sku.upper()]

    def page(self, n, size=10):
        skus = sorted(self._items)
        start = (n - 1) * size
        return [self._items[s] for s in skus[start:start + size]]

    def value_cents(self):
        return sum(q * p for _, q, p in self._items.values())
''',
    },
)


_DEPS = '''"""Resolve an install order from a dependency map."""


def install_order(deps):
    """`deps` maps a package to the packages it needs. Dependencies first."""
    order = []
    for pkg in deps:
        for d in deps[pkg]:
            order.append(d)
        order.append(pkg)
    return order
'''

DEPS = Task(
    id="deps",
    files={
        "deps/__init__.py": "",
        "deps/resolve.py": _DEPS,
        "tests/test_resolve.py": '''import pytest

from deps.resolve import CycleError, install_order


def test_dependencies_come_first_and_once():
    order = install_order({"app": ["web", "db"], "web": ["http"], "db": [],
                           "http": []})
    assert sorted(order) == ["app", "db", "http", "web"]
    assert order.index("http") < order.index("web") < order.index("app")


def test_a_cycle_is_an_error():
    with pytest.raises(CycleError):
        install_order({"a": ["b"], "b": ["a"]})
''',
    },
    parts=(
        "deps/resolve.py's install_order is wrong: it repeats packages and does not "
        "order transitive dependencies. Rewrite it as a proper topological sort "
        "that lists every package exactly once, dependencies before dependents, "
        "and raises CycleError (define it, a subclass of ValueError) naming the "
        "packages in the cycle. A dependency that is not a key in the map is a "
        "package with no dependencies. Ties are broken alphabetically so the "
        "order is deterministic. Make tests/test_resolve.py pass.",
    ),
    hidden={
        "tests_hidden/test_resolve_hidden.py": '''import pytest

from deps.resolve import CycleError, install_order


def test_unlisted_dependency_is_a_leaf():
    assert install_order({"a": ["z"]}) == ["z", "a"]


def test_deterministic_tie_break():
    assert install_order({"c": [], "b": [], "a": []}) == ["a", "b", "c"]


def test_diamond():
    order = install_order({"top": ["l", "r"], "l": ["base"], "r": ["base"],
                           "base": []})
    assert order[0] == "base" and order[-1] == "top" and len(order) == 4


def test_self_cycle_and_message():
    with pytest.raises(CycleError) as e:
        install_order({"a": ["a"]})
    assert "a" in str(e.value)
    assert issubclass(CycleError, ValueError)
''',
    },
    solution={
        "deps/resolve.py": '''"""Resolve an install order from a dependency map."""


class CycleError(ValueError):
    pass


def install_order(deps):
    nodes = set(deps)
    for v in deps.values():
        nodes.update(v)
    state, order = {}, []

    def visit(n, path):
        if state.get(n) == "done":
            return
        if state.get(n) == "active":
            raise CycleError(" -> ".join(path + [n]))
        state[n] = "active"
        for d in sorted(deps.get(n, [])):
            visit(d, path + [n])
        state[n] = "done"
        order.append(n)

    for n in sorted(nodes):
        visit(n, [])
    return order
''',
    },
)


_TEXT = '''"""Text utilities."""


def slugify(title):
    return title.lower().replace(" ", "-")
'''

TEXTKIT = Task(
    id="textkit",
    files={
        "textkit/__init__.py": "",
        "textkit/text.py": _TEXT,
        "tests/test_text.py": '''from textkit.text import slugify


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"
''',
    },
    parts=(
        "Harden textkit.text.slugify: collapse any run of characters that are not "
        "ASCII letters or digits into a single '-', strip leading and trailing '-', "
        "transliterate accented Latin letters to their base letter (é -> e), and "
        "return 'untitled' for input with no letters or digits. Add tests to "
        "tests/test_text.py.",
        "Add wrap(text, width) to textkit/text.py: greedy word wrap returning a "
        "list of lines, none longer than `width`, splitting a single word longer "
        "than `width` into width-sized chunks, collapsing runs of whitespace, and "
        "returning [] for empty or whitespace-only text. width < 1 raises "
        "ValueError. Add tests.",
        "Add truncate(text, limit, ellipsis='...') to textkit/text.py: returns text "
        "unchanged if it fits in `limit` characters, otherwise cuts at the last "
        "word boundary that leaves room for the ellipsis and appends it; if no "
        "word boundary fits, cut mid-word. The result, ellipsis included, is never "
        "longer than `limit`; limit smaller than the ellipsis raises ValueError. "
        "Add tests.",
    ),
    hidden={
        "tests_hidden/test_text_hidden.py": '''import pytest

from textkit.text import slugify, truncate, wrap


def test_slugify_edges():
    assert slugify("  Crème Brûlée!! 2026 ") == "creme-brulee-2026"
    assert slugify("a--b__c") == "a-b-c"
    assert slugify("!!!") == "untitled"
    assert slugify("") == "untitled"


def test_wrap():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]
    assert wrap("abcdefghij", 4) == ["abcd", "efgh", "ij"]
    assert wrap("  a   b  ", 5) == ["a b"]
    assert wrap("   ", 3) == []
    with pytest.raises(ValueError):
        wrap("x", 0)


def test_truncate():
    assert truncate("short", 10) == "short"
    assert truncate("the quick brown fox", 12) == "the quick..."
    assert truncate("abcdefghijkl", 8) == "abcde..."
    assert len(truncate("the quick brown fox", 12)) <= 12
    with pytest.raises(ValueError):
        truncate("abc", 2)
''',
    },
    solution={
        "textkit/text.py": '''"""Text utilities."""

import re
import unicodedata


def slugify(title):
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.lower()).strip("-")
    return s or "untitled"


def wrap(text, width):
    if width < 1:
        raise ValueError(width)
    lines, cur = [], ""
    for word in text.split():
        while len(word) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def truncate(text, limit, ellipsis="..."):
    if limit < len(ellipsis):
        raise ValueError(limit)
    if len(text) <= limit:
        return text
    room = limit - len(ellipsis)
    cut = text.rfind(" ", 0, room + 1)
    head = text[:cut] if cut > 0 else text[:room]
    return head.rstrip() + ellipsis
''',
    },
)


_CACHE = '''"""A small cache."""


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        self.data[key] = value
'''

CACHEKIT = Task(
    id="cachekit",
    files={
        "cachekit/__init__.py": "",
        "cachekit/lru.py": _CACHE,
        "tests/test_lru.py": '''from cachekit.lru import LRUCache


def test_evicts_least_recently_used():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
''',
    },
    parts=(
        "cachekit.lru.LRUCache never evicts. Make it a real LRU: capacity is a "
        "positive int (otherwise ValueError), get and put both count as use, "
        "putting an existing key updates it and refreshes it, and len(cache) "
        "reports the number of entries. Make tests/test_lru.py pass.",
        "Add optional expiry: LRUCache(capacity, ttl=None, clock=time.monotonic). "
        "With a ttl in seconds an entry older than ttl since it was last put is "
        "treated as absent by get, `in` and len, and is removed when seen. "
        "`clock` is injectable so tests control time. Add tests with a fake clock.",
        "Add hit and miss counters: cache.stats() returns a dict with 'hits', "
        "'misses' and 'evictions' (capacity evictions only, not expiries), and "
        "cache.clear() empties the cache without resetting the counters. Add tests.",
    ),
    hidden={
        "tests_hidden/test_lru_hidden.py": '''import pytest

from cachekit.lru import LRUCache


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_capacity_validated():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_update_refreshes():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)
    c.put("c", 3)
    assert c.get("a") == 10 and c.get("b") is None and len(c) == 2


def test_ttl_with_fake_clock():
    clk = Clock()
    c = LRUCache(3, ttl=5, clock=clk)
    c.put("a", 1)
    clk.t = 4.9
    assert c.get("a") == 1
    clk.t = 5.1
    assert c.get("a") is None and "a" not in c and len(c) == 0


def test_stats():
    c = LRUCache(1)
    c.put("a", 1)
    c.get("a")
    c.get("zz")
    c.put("b", 2)
    assert c.stats() == {"hits": 1, "misses": 1, "evictions": 1}
    c.clear()
    assert len(c) == 0 and c.stats()["hits"] == 1
''',
    },
    solution={
        "cachekit/lru.py": '''"""A small cache."""

import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity, ttl=None, clock=time.monotonic):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(capacity)
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.data = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _live(self, key):
        if key not in self.data:
            return False
        if self.ttl is not None and self.clock() - self.data[key][1] > self.ttl:
            del self.data[key]
            return False
        return True

    def get(self, key, default=None):
        if not self._live(key):
            self._stats["misses"] += 1
            return default
        self.data.move_to_end(key)
        self._stats["hits"] += 1
        return self.data[key][0]

    def put(self, key, value):
        self.data[key] = (value, self.clock())
        self.data.move_to_end(key)
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
            self._stats["evictions"] += 1

    def __contains__(self, key):
        return self._live(key)

    def __len__(self):
        for k in list(self.data):
            self._live(k)
        return len(self.data)

    def stats(self):
        return dict(self._stats)

    def clear(self):
        self.data.clear()
''',
    },
)


TASKLOG = Task(
    id="tasklog",
    files={
        "tasklog/__init__.py": "",
        "tasklog/core.py": '"""A small task tracker."""\n',
        "tests/test_core.py": """from tasklog.core import TaskList


def test_ids_count_from_one():
    tl = TaskList()
    assert tl.add("write report") == 1
    assert tl.add("send it") == 2
    assert tl.get(2).title == "send it"
""",
    },
    parts=(
        "In tasklog/core.py create a Task class (fields id, title, done, due, tags, "
        "priority) and a TaskList with add(title) -> int id (ids count from 1) and "
        "get(id) -> Task, raising KeyError for an unknown id. Make tests/test_core.py "
        "pass.",
        "Add TaskList.complete(id) and pending(), which returns the tasks not yet "
        "done, in id order. Add tests.",
        "Let add take tags=(): tags are stored lower-cased and stripped, with "
        "duplicates removed keeping first occurrence. Add by_tag(tag), "
        "case-insensitive, returning matching tasks in id order. Add tests.",
        "Let add take due=None as a 'YYYY-MM-DD' string, stored as a datetime.date; "
        "an invalid date raises ValueError. Add overdue(today), taking a date and "
        "returning pending tasks due strictly before it, ordered by due date then "
        "id. Add tests.",
        "Add save(path), writing the list as JSON, and a classmethod "
        "TaskList.load(path) that restores it exactly, including done flags, due "
        "dates and tags; ids added after a load continue from the highest id "
        "loaded. Add tests.",
        "Let add take priority='normal', one of 'low', 'normal', 'high' (anything "
        "else raises ValueError). Add next_up(n) returning up to n pending tasks: "
        "high before normal before low, then earliest due (tasks with no due date "
        "last), then id. Add tests.",
        "Add remove(id) and undo(), which reverts the most recent add, complete or "
        "remove (removing the added task, marking the completed one pending again, "
        "or restoring the removed one with its id). undo() with nothing to undo "
        "raises IndexError. Add tests.",
        "Add summary(today) returning a string like '3 pending, 1 done, 1 overdue' "
        "(overdue as defined by overdue(today)). Add tests, and make sure the whole "
        "test suite passes.",
    ),
    hidden={
        "tests_hidden/test_tasklog_hidden.py": """import datetime as dt

import pytest

from tasklog.core import TaskList

D = dt.date


def test_parts_one_and_two():
    tl = TaskList()
    a, b, c = tl.add("a"), tl.add("b"), tl.add("c")
    with pytest.raises(KeyError):
        tl.get(99)
    tl.complete(b)
    assert [t.id for t in tl.pending()] == [a, c]


def test_tags():
    tl = TaskList()
    i = tl.add("x", tags=(" Work", "work", "Home "))
    assert tl.get(i).tags == ["work", "home"] or tl.get(i).tags == ("work", "home")
    tl.add("y", tags=("home",))
    assert [t.title for t in tl.by_tag("HOME")] == ["x", "y"]


def test_due_and_overdue():
    tl = TaskList()
    tl.add("late", due="2026-01-05")
    tl.add("later", due="2026-01-03")
    tl.add("fine", due="2026-02-01")
    tl.add("none")
    assert tl.get(1).due == D(2026, 1, 5)
    assert [t.title for t in tl.overdue(D(2026, 1, 10))] == ["later", "late"]
    with pytest.raises(ValueError):
        tl.add("bad", due="2026-13-40")


def test_round_trip(tmp_path):
    tl = TaskList()
    tl.add("a", due="2026-01-01", tags=("t",), priority="high")
    tl.add("b")
    tl.complete(2)
    p = tmp_path / "tl.json"
    tl.save(p)
    back = TaskList.load(p)
    assert back.get(1).due == D(2026, 1, 1) and back.get(2).done
    assert list(back.get(1).tags) == ["t"] and back.get(1).priority == "high"
    assert back.add("c") == 3


def test_priority_and_next_up():
    tl = TaskList()
    tl.add("low", priority="low")
    tl.add("n-nodue")
    tl.add("n-due", due="2026-03-01")
    tl.add("high", priority="high")
    assert [t.title for t in tl.next_up(3)] == ["high", "n-due", "n-nodue"]
    with pytest.raises(ValueError):
        tl.add("x", priority="urgent")


def test_undo():
    tl = TaskList()
    tl.add("a")
    tl.add("b")
    tl.complete(1)
    tl.remove(2)
    tl.undo()
    assert tl.get(2).title == "b"
    tl.undo()
    assert not tl.get(1).done
    tl.undo()
    with pytest.raises(KeyError):
        tl.get(2)
    tl.undo()
    with pytest.raises(IndexError):
        tl.undo()


def test_summary():
    tl = TaskList()
    tl.add("a", due="2026-01-01")
    tl.add("b")
    tl.add("c")
    tl.complete(3)
    assert tl.summary(D(2026, 6, 1)) == "2 pending, 1 done, 1 overdue"
""",
    },
    solution={
        "tasklog/core.py": """# A small task tracker.

import datetime as dt
import json
from dataclasses import dataclass, field

PRIORITIES = ("high", "normal", "low")


@dataclass
class Task:
    id: int
    title: str
    done: bool = False
    due: dt.date | None = None
    tags: list = field(default_factory=list)
    priority: str = "normal"


class TaskList:
    def __init__(self):
        self.tasks = {}
        self.next_id = 1
        self.history = []

    def add(self, title, due=None, tags=(), priority="normal"):
        if priority not in PRIORITIES:
            raise ValueError(priority)
        when = dt.date.fromisoformat(due) if due is not None else None
        clean = []
        for t in tags:
            t = t.strip().lower()
            if t not in clean:
                clean.append(t)
        i = self.next_id
        self.next_id += 1
        self.tasks[i] = Task(i, title, False, when, clean, priority)
        self.history.append(("add", i))
        return i

    def get(self, i):
        return self.tasks[i]

    def complete(self, i):
        self.tasks[i].done = True
        self.history.append(("complete", i))

    def remove(self, i):
        self.history.append(("remove", self.tasks.pop(i)))

    def undo(self):
        if not self.history:
            raise IndexError("nothing to undo")
        kind, x = self.history.pop()
        if kind == "add":
            del self.tasks[x]
        elif kind == "complete":
            self.tasks[x].done = False
        else:
            self.tasks[x.id] = x

    def pending(self):
        return [t for _, t in sorted(self.tasks.items()) if not t.done]

    def by_tag(self, tag):
        tag = tag.strip().lower()
        return [t for _, t in sorted(self.tasks.items()) if tag in t.tags]

    def overdue(self, today):
        late = [t for t in self.pending() if t.due is not None and t.due < today]
        return sorted(late, key=lambda t: (t.due, t.id))

    def next_up(self, n):
        def key(t):
            return (PRIORITIES.index(t.priority), t.due is None,
                    t.due or dt.date.max, t.id)
        return sorted(self.pending(), key=key)[:n]

    def summary(self, today):
        done = sum(t.done for t in self.tasks.values())
        return (f"{len(self.pending())} pending, {done} done, "
                f"{len(self.overdue(today))} overdue")

    def save(self, path):
        rows = [{"id": t.id, "title": t.title, "done": t.done,
                 "due": t.due.isoformat() if t.due else None, "tags": list(t.tags),
                 "priority": t.priority} for t in self.tasks.values()]
        with open(path, "w") as fh:
            json.dump({"next_id": self.next_id, "tasks": rows}, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            blob = json.load(fh)
        tl = cls()
        for r in blob["tasks"]:
            due = dt.date.fromisoformat(r["due"]) if r["due"] else None
            tl.tasks[r["id"]] = Task(r["id"], r["title"], r["done"], due, r["tags"],
                                     r["priority"])
        tl.next_id = max([blob.get("next_id", 1)] + [i + 1 for i in tl.tasks])
        return tl
""",
    },
)


TASKS: tuple[Task, ...] = (LEDGER, INVENTORY, DEPS, TEXTKIT, CACHEKIT)
# Long enough that context accumulates the way a real session's does: eight
# parts, each building on the last. Not in the default suite, because one
# baseline run of it costs what the rest of the suite does.
LONG: tuple[Task, ...] = (TASKLOG,)
BY_ID: dict[str, Task] = {t.id: t for t in TASKS + LONG}
