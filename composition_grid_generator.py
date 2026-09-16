from decimal import Decimal
from itertools import product


def make_oxide_values(start, end, step, oxide):
    start_value = Decimal(str(start))
    end_value = Decimal(str(end))
    step_value = Decimal(str(step))

    if start_value == end_value:
        return [float(start_value)]

    if step_value == 0:
        raise ValueError(f"{oxide} varies, so its step cannot be zero")

    direction = Decimal("1") if end_value > start_value else Decimal("-1")

    if step_value * direction <= 0:
        raise ValueError(
            f"{oxide} has an inconsistent start, end, and step direction"
        )

    interval_count = (end_value - start_value) / step_value

    if interval_count != interval_count.to_integral_value():
        raise ValueError(
            f"{oxide} range is not exactly divisible by its step. "
            f"Start={start}, end={end}, step={step}"
        )

    return [
        float(start_value + index * step_value)
        for index in range(int(interval_count) + 1)
    ]


def make_composition_grid(composition_ranges):
    oxide_names = list(composition_ranges.keys())
    oxide_value_lists = []

    for oxide in oxide_names:
        start, end, step = composition_ranges[oxide]
        oxide_value_lists.append(make_oxide_values(start, end, step, oxide))

    for values in product(*oxide_value_lists):
        yield dict(zip(oxide_names, values))


def composition_grid_size(composition_ranges):
    total = 1

    for oxide, values in composition_ranges.items():
        start, end, step = values
        total *= len(make_oxide_values(start, end, step, oxide))

    return total
