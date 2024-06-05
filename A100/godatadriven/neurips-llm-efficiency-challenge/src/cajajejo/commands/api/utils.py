# ruff: noqa: E741
import re
import typing


MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}
MAP_REVERSE = {v: k for k, v in MAP.items()}


class FastApiPlaceholder(object):
    def __init__(self):
        ...

    def post(self, endpoint):
        def wrapper(f):
            return f

        return wrapper

    def get(self, endpoint):
        def wrapper(f):
            return f

        return wrapper


def remove_singletons(l):
    out = []
    for i in l:
        if i == "A":
            out.append(i)
            continue
        i_p = MAP[MAP_REVERSE[i] - 1]
        if i_p not in l:
            continue
        else:
            out.append(i)
    return out


def remove_from_text(x):
    return re.sub(r"[A-Z]\.\s[A-Z]\.", "", x)


def extract_mc_options(x) -> typing.Optional[typing.List[str]]:
    # This ridiculous regex matches:
    #  - NOT a period, number, letter, space, or dash
    #  - Capital letter, followed by colon or period and space
    #  - Anything, greedy (except for question mark)
    #  - Until a whitespace
    # And then we're pretty sure that we're dealing with a multiple-choice question
    regex = r"(?<!\.|[0-9]|[A-Za-z]| |-)[A-Z][\.:]\s.*?(?<!\?)\n"
    result = re.findall(regex, str(x) + "\n")
    if len(result) == 0:
        return None
    else:
        return [r.strip("\n") for r in result]


def check_statements_from_identifiers(l):
    if "I" in l and "H" not in l:
        return True
    else:
        return False


def check_if_qa(l):
    if "Q" in l and "P" not in l:
        return True
    else:
        return False


def get_identifier(l):
    # Retrieve identifiers (e.g. 'A', 'B' for possible MC options)
    return [re.split("\.|:", i)[0] for i in l]


def identifier_to_int(l):
    # Map identifiers to integers
    return [MAP_REVERSE[c] for c in l]


def sequential(l):
    # Check if a list of integer identifiers is sequential
    return l == [*range(min(l), max(l) + 1)]


def min_length(l):
    # Check if a list of integer identifiers is at least length 2
    return len(l) > 1


def is_multiple_choice(p):
    p = remove_from_text(p)
    mc_opts = extract_mc_options(p)
    if mc_opts is None:
        return False
    identifiers = sorted(set(get_identifier(mc_opts)))
    is_statement = check_statements_from_identifiers(identifiers)
    if is_statement:
        identifiers = [i for i in identifiers if i != "I"]
    is_qa = check_if_qa(identifiers)
    if is_qa:
        identifiers = [i for i in identifiers if i != "Q"]
    identifiers = remove_singletons(identifiers)
    if not min_length(identifiers):
        return False
    identifiers_int = identifier_to_int(identifiers)
    return sequential(identifiers_int)
