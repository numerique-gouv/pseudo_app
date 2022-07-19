import copy
import glob
import itertools
import re
import subprocess
from pathlib import Path
from string import ascii_uppercase
from hashlib import md5
from typing import Tuple

import dash_html_components as html
import pandas as pd
import textract
from flair.data import Token
from typing import List
from flair.datasets import ColumnDataset
import requests
import xml.etree.ElementTree as ET


def prepare_upload_tab_html(
    tags:str,
    pseudo: str,
):
    """
    Build Dash frontend to display tags

    Args:
        tags (str): _description_
        pseudo (str): _description_
        original_text_lines (str): _description_
    """

    def generate_upload_tab_html_components(tagged_text: str):
        html_components = []
        if tagged_text:
            root = ET.fromstring(tagged_text)
            for child in root: # normaly every of these "first-degree-children" must be sentences
                if child.tag == "sentence":
                    if not len(child.tag): # = no grandchildren. It should not happen with API, as "not entity" text is wrapped in tag <a>
                        html_components.append(html.P(child.text))
                    else:
                        marked_content = []
                        for grandchild in child:
                            if grandchild.tag == "a":
                                marked_content.append(grandchild.text)
                            elif grandchild.tag:
                                marked_content.append(
                                    html.Mark(children=grandchild.text, **{"data-entity": ENTITIES[grandchild.tag], "data-index": ""})
                                )
                        html_components.append(html.P(marked_content))
        return html_components

    html_components_pseudo =  html.P(pseudo)
    html_components_tagged = generate_upload_tab_html_components(tagged_text=tags)
    return html_components_tagged, html_components_pseudo

def request_pseudo_api(text: str, pseudo_api_url: str) -> str:
    try:
        r = requests.post(pseudo_api_url, {"text": text}).json()
        if r["success"]:
            return r["pseudo"]
    except Exception as e:
        raise e

def request_tags_api(text: str, pseudo_api_url: str) -> Tuple[str, str]:
    try:
        r = requests.post(pseudo_api_url, {"text": text}).json()
        if r["success"]:
            return r["tags"], r["pseudo"]
    except Exception as e:
        raise e

def request_stats_api(pseudo_api_url: str):
    if not pseudo_api_url:
        return
    r = requests.get(pseudo_api_url).json()
    if r["success"]:
        return r["stats_info"]


def create_upload_tab_html_output(text:str, pseudo_api_url:str):
    tags, pseudo = request_tags_api(text=text, pseudo_api_url=pseudo_api_url+"tags/")
    html_tagged, html_pseudoynmized = prepare_upload_tab_html(
        tags=tags,
        pseudo=pseudo
    )
    return html_tagged, html_pseudoynmized 


def file2txt(doc_path: str) -> str:
    if doc_path.endswith("doc"):
        result = subprocess.run(['antiword', '-w', '0', doc_path], stdout=subprocess.PIPE)
        result = result.stdout.decode("utf-8").replace("|", "\t")
    else:
        result = textract.process(doc_path, encoding='utf-8').decode("utf8").replace("|", "\t")
    return result


def load_text(doc_path: Path) -> str:
    return file2txt(doc_path.as_posix())


ENTITIES = {"PER_PRENOM": "PRENOM", "PER_NOM": "NOM", "LOC": "ADRESSE", "PER": "PERSONNE", "ORG": "ORGANISATION"}

def sent_tokenizer(text):
    return text.split("\n")



def add_span_positions_to_dataset(dataset: ColumnDataset):
    for i_sent, sentence in enumerate(dataset.sentences):
        for i_tok, token in enumerate(sentence.tokens):
            token: Token = token
            if i_tok == 0:
                token.start_pos = 0
            else:
                prev_token = sentence.tokens[i_tok - 1]
                # if comma, dot do increment counter (there is no space between them and prev token)
                if (len(token.text) == 1 and re.match(r'[.,]', token.text)) or re.match(r"\w?[('°]$", token.text):
                    token.start_pos = prev_token.end_pos
                else:
                    token.start_pos = prev_token.end_pos + 1

            token.end_pos = token.start_pos + len(token.text)


def prepare_error_decisions(decisions_path: Path):
    error_files = glob.glob(decisions_path.as_posix() + "/*.txt")
    dict_df = {}
    dict_stats = {}
    for error_file in error_files:
        df_error = pd.read_csv(error_file, sep="\t", engine="python", skip_blank_lines=False,
                               names=["token", "true_tag", "pred_tag"]).fillna("")
        df_no_spaces = df_error[df_error["token"] != ""]

        under_pseudonymization = df_no_spaces[(df_no_spaces["true_tag"] != df_no_spaces["pred_tag"])
                                              & (df_no_spaces["true_tag"] != "O")]
        miss_pseudonymization = df_no_spaces[(df_no_spaces["true_tag"] != df_no_spaces["pred_tag"])
                                             & (df_no_spaces["true_tag"] != "O")
                                             & (df_no_spaces["pred_tag"] != "O")]
        over_pseudonymization = df_no_spaces[(df_no_spaces["true_tag"] != df_no_spaces["pred_tag"])
                                             & (df_no_spaces["true_tag"] == "O")]
        correct_pseudonymization = df_no_spaces[(df_no_spaces["true_tag"] == df_no_spaces["pred_tag"])
                                                & (df_no_spaces["true_tag"] != "O")]

        df_error["display_col"] = "O"
        if not correct_pseudonymization.empty:
            df_error.loc[correct_pseudonymization.index, "display_col"] = correct_pseudonymization['pred_tag'] + "_C"
        if not under_pseudonymization.empty:
            df_error.loc[under_pseudonymization.index, "display_col"] = under_pseudonymization["pred_tag"] + "_E"
        if not miss_pseudonymization.empty:
            df_error.loc[miss_pseudonymization.index, "display_col"] = miss_pseudonymization["pred_tag"] + "_E"
        if not over_pseudonymization.empty:
            df_error.loc[over_pseudonymization.index, "display_col"] = over_pseudonymization["pred_tag"] + "_E"
        df_error.loc[df_error["token"] == "", "display_col"] = ""

        # Get simple stats
        nb_noms = len(df_error[df_error["true_tag"].str.startswith("B-PER_NOM")])
        nb_prenoms = len(df_error[df_error["true_tag"].str.startswith("B-PER_PRENOM")])
        nb_loc = len(df_error[df_error["true_tag"].str.startswith("B-LOC")])

        serie_stats = pd.Series({"nb_noms": nb_noms, "nb_prenoms": nb_prenoms, "nb_loc": nb_loc,
                                 "under_classifications": len(under_pseudonymization),
                                 "over_classifications": len(over_pseudonymization),
                                 "miss_classifications": len(miss_pseudonymization),
                                 "correct_classifications": len(correct_pseudonymization)})

        dict_df[error_file.split("/")[-1]] = df_error.loc[:, ["token", "display_col"]]
        dict_stats[error_file.split("/")[-1]] = serie_stats

    return dict_df, dict_stats
