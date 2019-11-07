import ast
import numpy as np


def get_text_as_list(text):
    if text is None or text is np.nan:
        return []
    else:
        return ast.literal_eval(text)

