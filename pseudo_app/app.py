import os
from typing import Dict

import dash

from components.page_layout import app_page_layout
from components.tab_about import tab_about_content
from components.tab_upload import tab_upload_content, pane_upload_content

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

def layout():
    """
    Returns page layout. Modifications of page structure should be passed here
    :return:  Layout
    """
    div = html.Div(id='seq-view-body', className='app-body', children=[
        dcc.Store(id='session-store', storage_type='session'),
        html.Div(id='seq-view-control-tabs',
                 className="six columns div-user-controls",
                 children=dbc.Container(
                     dbc.Tabs(id='main-tabs', children=[
                         tab_about_content,
                         tab_upload_content,
                     ], active_tab="tab-about"),
                 )),
        dbc.Container(className="five columns", fluid=True,
                      children=dcc.Loading(id="right-pane",
                                           type="default",
                                           fullscreen=False,
                                           className="six columns",
                                           )
                      )
    ])
    return div


def callbacks(_app):
    """ Define callbacks to be executed on page change"""

    @_app.callback([Output("error-pane", 'children'),
                    Output("alert-stats", 'children')],
                   [Input('error-slider', 'value')])

    @_app.callback([Output('right-pane', 'children'),
                    Output('session-store', 'data')],
                   [Input('upload-data', 'contents'),
                    Input('upload-data', 'filename'),
                    Input('upload-div', 'n_clicks'),
                    Input("main-tabs", "active_tab"),
                    Input("example-text", "n_clicks")],
                   [State('session-store', 'data')])
    def pseudo_pane_update(contents, file_name: str, n_clicks_upload, tab_is_at, n_clicks_example, data: Dict):
        data = data or {"previous_tab": tab_is_at}
        if tab_is_at == "tab-about":
            data["previous_tab"] = tab_is_at
            return None, data
        elif tab_is_at == "tab-errors":
            data["previous_tab"] = tab_is_at
            return pane_errors_content, data
        elif tab_is_at == "tab-upload":
            if data["previous_tab"] != "tab-upload":
                if "previous_content" in data and data["previous_content"]:
                    children = data["previous_content"]
                    data["previous_tab"] = tab_is_at
                    return children, data
            children, data = pane_upload_content(contents, file_name, n_clicks_example, data)
            data["previous_tab"] = tab_is_at
            return children, data


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname="/pseudo/")
server = app.server
app.title = "Pseudo"
app_title = "Démo Pseudo"
app.config['suppress_callback_exceptions'] = True
# Assign layout
app.layout = app_page_layout(
    page_layout=layout(),
    app_title=app_title,
)
# Register all callbacks
callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
