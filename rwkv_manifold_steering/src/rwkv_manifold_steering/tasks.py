from __future__ import annotations

from dataclasses import dataclass


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
NUMBERS = [
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
]
MONTH_NUMBERS = [
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
]


@dataclass(frozen=True)
class WeekdayExample:
    entity: str
    number_word: str
    number: int
    result: str
    result_index: int
    prompt: str


@dataclass(frozen=True)
class CyclicTask:
    name: str
    labels: list[str]
    examples: list[WeekdayExample]
    unit: str


def make_weekday_prompt(entity: str, number_word: str) -> str:
    return f"User: What day comes {number_word} days after {entity}?\n\nAssistant:"


def make_weekday_examples() -> list[WeekdayExample]:
    examples: list[WeekdayExample] = []
    for entity_index, entity in enumerate(WEEKDAYS):
        for number_word, number in NUMBERS:
            result_index = (entity_index + number) % len(WEEKDAYS)
            result = WEEKDAYS[result_index]
            examples.append(
                WeekdayExample(
                    entity=entity,
                    number_word=number_word,
                    number=number,
                    result=result,
                    result_index=result_index,
                    prompt=make_weekday_prompt(entity, number_word),
                )
            )
    return examples


def make_month_prompt(entity: str, number_word: str) -> str:
    return f"User: What month comes {number_word} months after {entity}?\n\nAssistant:"


def make_month_examples() -> list[WeekdayExample]:
    examples: list[WeekdayExample] = []
    for entity_index, entity in enumerate(MONTHS):
        for number_word, number in MONTH_NUMBERS:
            result_index = (entity_index + number) % len(MONTHS)
            result = MONTHS[result_index]
            examples.append(
                WeekdayExample(
                    entity=entity,
                    number_word=number_word,
                    number=number,
                    result=result,
                    result_index=result_index,
                    prompt=make_month_prompt(entity, number_word),
                )
            )
    return examples


def get_cyclic_task(name: str) -> CyclicTask:
    if name == "weekday":
        return CyclicTask(
            name="weekday",
            labels=WEEKDAYS,
            examples=make_weekday_examples(),
            unit="days",
        )
    if name == "month":
        return CyclicTask(
            name="month",
            labels=MONTHS,
            examples=make_month_examples(),
            unit="months",
        )
    raise ValueError(f"unknown cyclic task: {name}")
