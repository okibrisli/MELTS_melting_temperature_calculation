from copy import deepcopy


def linear_composition_path(start_composition, end_composition, point_count):
    if point_count < 2:
        raise ValueError("point_count must be at least 2")

    oxide_names = list(dict.fromkeys(start_composition.keys() | end_composition.keys()))
    compositions = []

    for index in range(point_count):
        fraction = index / (point_count - 1)
        composition = {}

        for oxide in oxide_names:
            start_value = float(start_composition.get(oxide, 0.0))
            end_value = float(end_composition.get(oxide, 0.0))
            value = start_value + fraction * (end_value - start_value)

            if value < 0.0:
                raise ValueError(f"Negative value generated for {oxide}: {value}")

            composition[oxide] = value

        compositions.append(composition)

    return compositions


def copy_composition(composition):
    return deepcopy(composition)
