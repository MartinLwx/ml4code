import ast
import tokenize
from datasets import load_dataset
from io import BytesIO


def get_code(partition: str, language: str = "python") -> list[str]:
    """
    Get data from the datasets library from huggingface.
    Only keep the `whole_func_string` column

    Arg
    ---
        `partition`: train/validation/test
        `language`: see details in the CodeSearchNet

    Return
    -----
        return a list of python functions
    """
    raw_datasets = load_dataset("code_search_net", language)

    return raw_datasets[partition]["whole_func_string"]


def get_token_stream(code: str, language: str = "python") -> list[str]:
    """
    Tokenize the source code and return a token stream

    Note that the following token type will be removed:
    - COMMENT
    - NEWLINE
    - NL
    - INDENT
    - DEDENT
    - ENCODING
    - STRING
    """
    assert language == "python", "Only support python for now"

    # see https://docs.python.org/3/library/token.html
    useless_token_type = {
        tokenize.COMMENT,
        tokenize.NEWLINE,
        tokenize.NL,  # non-terminating newline
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.STRING,
    }
    parse_tree = ast.parse(code)
    origin_tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
    token_as_strlist = [
        token.string for token in origin_tokens if token.type not in useless_token_type
    ]

    return token_as_strlist


def create_vocab(
    tokenized_corpus: list[list[str]],
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Arg
    ---
        `tokenized_corpus`: each document in this corpus is tokenized
    Return
    ------
        token2id
        id2token
    """
    unique_words = set(sum(tokenized_corpus, start=[]))
    token2id = {}
    for token in sorted(unique_words):
        token2id[token] = len(token2id)
    id2token = {v: k for k, v in token2id.items()}

    return token2id, id2token
