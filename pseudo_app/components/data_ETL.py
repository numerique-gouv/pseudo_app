import re
import subprocess
from pathlib import Path
from string import ascii_uppercase
from hashlib import md5
from typing import Tuple

import dash_html_components as html
import pandas as pd
import textract
from typing import List
import requests
import xml.etree.ElementTree as ET


def prepare_upload_tab_html(
    tags:str,
    pseudo: str,
):
    """
    Build Dash frontend to display tags

    Args:
        tags (str): the response of pseudo API - text where recognized entities are highligthed in xml tags
        pseudo (str): the response of pseudo API - text where recognized entities are replaced with random characters
    """

    def generate_upload_tab_html_components(tagged_text: str):
        html_components = []
        if tagged_text:
            root = ET.fromstring(tagged_text) # interpret the input string as XML
            for child in root: # normaly every of these "first-degree-children" must be sentences
                if child.tag == "sentence":
                    if not len(child.tag): # = no grandchildren. It should not happen with API, as "not entity" text is wrapped in tag <a>
                        html_components.append(html.P(child.text))
                    else:
                        marked_content = []
                        for grandchild in child:
                            if grandchild.tag == "a": # = this span does not contain recognized entities
                                marked_content.append(grandchild.text)
                            elif grandchild.tag: # = this span contain recognized entities
                                marked_content.append(
                                    html.Mark(children=grandchild.text, **{"data-entity": ENTITIES[grandchild.tag], "data-index": ""})
                                )
                        html_components.append(html.P(marked_content))
        return html_components

    html_components_pseudo =  [html.P(pseudo_sentence) for pseudo_sentence in re.split("\?|\.|\n|\!", pseudo)]
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
