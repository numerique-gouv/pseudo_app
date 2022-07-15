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
                            marked_content.append(children=grandchild.text, **{"data-entity": ENTITIES[grandchild.tag], "data-index": ""})
                    html_components.append(html.P(marked_content))
        return html_components

    html_components_pseudo =  html.P(pseudo)
    html_components_tagged = generate_upload_tab_html_components(tagged_text=tags)
    return html_components_tagged, html_components_pseudo


def prepare_upload_tab_html_2(sentences_tagged:str, original_text_lines):
    singles = [f"{letter}..." for letter in ascii_uppercase]
    doubles = [f"{a}{b}..." for a, b in list(itertools.combinations(ascii_uppercase, 2))]
    pseudos = singles + doubles
    pseudo_entity_dict = {}
    sentences_pseudonymized = copy.deepcopy(sentences_tagged)

    def generate_upload_tab_html_components(sentences, original_text):
        html_components = []
        for i_sent, sent in enumerate(sentences):
            sent_span = sent.get_spans("ner")
            if not sent_span:
                html_components.append(html.P(sent.to_original_text()))
            else:
                temp_list = []
                index = 0
                for span in sent_span:
                    start = span.start_pos
                    end = span.end_pos
                    temp_list.append(original_text[i_sent][index:start])
                    index = end
                    temp_list.append(
                        html.Mark(children=span.text, **{"data-entity": ENTITIES[span.tag], "data-index": ""}))
                temp_list.append(original_text[i_sent][index:])
                html_components.append(html.P(temp_list))
        return html_components
    for id_sn, sent in enumerate(sentences_pseudonymized):
        for sent_span in sent.get_spans("ner"):
            if "LOC" in sent_span.tag:
                for id_tok in range(len(sent_span.tokens)):
                    sent_span.tokens[id_tok].text = "..."
            else:
                for id_tok, token in enumerate(sent_span.tokens):
                    replacement = pseudo_entity_dict.get(token.text.lower(), pseudos.pop(0))
                    pseudo_entity_dict[token.text.lower()] = replacement
                    sent_span.tokens[id_tok].text = replacement

    html_components_anonym = generate_upload_tab_html_components(sentences=sentences_pseudonymized,
                                                                 original_text=original_text_lines)
    html_components_tagged = generate_upload_tab_html_components(sentences=sentences_tagged,
                                                                 original_text=original_text_lines)
    return html_components_anonym, html_components_tagged


def create_flair_corpus(conll_tagged: str):
    text_id = md5(conll_tagged.encode("utf-8")).hexdigest()
    temp_conll_file = Path(f"/tmp/{text_id}")
    try:
        with open(temp_conll_file, "w") as temp_file:
            temp_file.write(conll_tagged)

        flair_corpus = ColumnDataset(path_to_column_file=temp_conll_file,
                                     column_name_map={0: 'text', 1: 'ner',
                                                      2: 'start_pos', 3: 'end_pos'})
        for sentence in flair_corpus.sentences:
            for (token, start_pos_span, end_pos_span) in zip(sentence.tokens, sentence.get_spans("start_pos"),
                                                             sentence.get_spans("end_pos")):
                token.start_pos = int(start_pos_span.tag)
                token.end_pos = int(end_pos_span.tag)

        return flair_corpus
    finally:
        temp_conll_file.unlink()


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
    tags, pseudo = request_pseudo_api(text=text, pseudo_api_url=pseudo_api_url+"tags/")
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


def retokenize_conll(dataset: ColumnDataset):
    for s in dataset.sentences:
        sent_tokens = [t.text for t in s.tokens]
        sent_text = detokenizer_fr.detokenize(sent_tokens)
        span_tokens = tokenizer_fr.span_tokenize(sent_text)
        if not len(sent_tokens) == len(span_tokens):
            return
        for i, t in enumerate(s.tokens):
            t.start_pos = span_tokens[i][1][0]
            t.end_pos = span_tokens[i][1][1]

    return dataset


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
