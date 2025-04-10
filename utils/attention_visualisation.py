import json
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import plotly.colors as pc
from scipy.interpolate import interp1d
from matplotlib.colors import to_rgb
import IPython.display as display
from IPython.display import IFrame
import dash
from threading import Thread

def _get_distinct_colors(n, colormap='Viridis'):
    cmap = pc.sequential.__dict__[colormap]
    cmap_rgb = [to_rgb(c) for c in cmap]
    xs = np.linspace(0, 1, len(cmap_rgb))
    interp = interp1d(xs, cmap_rgb, axis=0)
    sampled_colors = [f'rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})' for r, g, b in interp(np.linspace(0, 1, n))]
    return sampled_colors

def _get_dulled_colors(colors):
    return [color.replace('rgb', 'rgba').replace(')', ', 0.3)') if 'rgb' in color 
            else f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)" 
            for color in colors]


def _plot_attention_heads(attention_matrix, tokens_in, tokens_out, layer_idx, head_colors, subset_heads=None, figsize=(800, 600)):
    fig = go.Figure()
    if figsize == (800, 600):
        fig.add_shape(
            type="rect",
            x0=-0.1, y0=-0.5,
            x1=1.1, y1=max(len(tokens_in), len(tokens_out)) + 0.5,
            line=dict(color="white", width=1),
            fillcolor="black",
            opacity=0.1
        )
    
    for h in range(attention_matrix.shape[1]):
        if subset_heads is None or h in subset_heads:
            for i in range(len(tokens_in)):
                for j in range(len(tokens_out)):
                    weight = attention_matrix[layer_idx, h, i, j]
                    if weight > 0.05:
                        fig.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[i, j],
                            mode='lines',
                            line=dict(width=2*weight, color=head_colors[h]),
                            hoverinfo='text',
                            hovertext=f"Attention: {weight:.2f}"
                        ))
    
    for i, token in enumerate(tokens_in):
        fig.add_trace(go.Scatter(
            x=[0],
            y=[i],
            mode='text',
            text=token,
            textposition="middle left",
            textfont=dict(color="white", size=10),
            hoverinfo='text',
            hovertext=f"Token: '{token}'",
        ))
    
    for j, token in enumerate(tokens_out):
        fig.add_trace(go.Scatter(
            x=[1],
            y=[j],
            mode='text',
            text=token,
            textposition="middle right",
            textfont=dict(color="white", size=10),
            hoverinfo='text',
            hovertext=f"Token: '{token}'",
        ))
    
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=20, r=40, t=40, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.3, 1.3]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, max(len(tokens_in), len(tokens_out)) + 0.5]
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        showlegend=False,
    )
    return fig


def run_attention_dashboard(attention_vals, tokens_in, tokens_out, 
                              port=8050, colormap='Magma', height=800):
    app = Dash("head_view")
    dark_bg = 'black'
    text_color = 'white'
    with open('../utils/assets/individual.css', 'r') as f:
        css_content = f.read()
    app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {css_content}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''
    
    # Get dimensions and colors
    num_heads = attention_vals.shape[1]
    num_layers = attention_vals.shape[0]
    color_list = _get_distinct_colors(num_heads, colormap=colormap)
    dulled_colors = _get_dulled_colors(color_list)
    
    # Define layout
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Layer:", style={'color': text_color, 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='layer-dropdown',
                        options=[{'label': str(i), 'value': i} for i in range(num_layers)],
                        value=0,
                        style={'width': '80px'},
                        className='dark-dropdown'
                    ),
                ], className='control-group'),
                
                html.Div([
                    html.Div("Toggle attention heads:", className='head-toggle-label', style={'color': text_color}),
                    html.Div(id='color-toggles', children=[
                        html.Div(
                            style={
                                'backgroundColor': color_list[i],
                                'width': '40px',
                                'height': '30px',
                                'display': 'inline-block',
                                'margin': '3px',
                                'cursor': 'pointer',
                                'border': '2px solid white',
                                'textAlign': 'center',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'lineHeight': '30px',
                                'borderRadius': '3px'
                            },
                            id=f'color-box-{i}',
                            children=f"{i}",
                            n_clicks=0
                        ) for i in range(len(color_list))
                    ], className='head-toggles')
                ], className='control-group'),
            ], className='controls'),
            
            html.Div([
                dcc.Graph(id='attention-graph')
            ], className='graph-container'),
            
            html.Div(id='selected-heads', style={'display': 'none'}, children=','.join(str(i) for i in range(num_heads)))
        ], className='container')
    ])
    
    # Define callbacks
    @app.callback(
        [Output(f'color-box-{i}', 'style') for i in range(len(color_list))],
        [Input('selected-heads', 'children')]
    )
    def update_toggles(selected_heads_str):
        selected_heads = [int(h) for h in selected_heads_str.split(',') if h]
        styles = []
        for i in range(len(color_list)):
            if i in selected_heads:
                styles.append({
                    'backgroundColor': color_list[i],
                    'width': '40px',
                    'height': '30px',
                    'display': 'inline-block',
                    'margin': '3px',
                    'cursor': 'pointer',
                    'border': '2px solid white',
                    'textAlign': 'center',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'lineHeight': '30px',
                    'borderRadius': '3px'
                })
            else:
                styles.append({
                    'backgroundColor': dulled_colors[i],
                    'width': '40px',
                    'height': '30px',
                    'display': 'inline-block',
                    'margin': '3px',
                    'cursor': 'pointer',
                    'border': '2px dashed gray',
                    'textAlign': 'center',
                    'color': 'white',
                    'fontWeight': 'normal',
                    'lineHeight': '30px',
                    'borderRadius': '3px'
                })
        
        return styles
    
    @app.callback(
        Output('attention-graph', 'figure'),
        Output('selected-heads', 'children'),
        [Input('layer-dropdown', 'value')] +
        [Input(f'color-box-{i}', 'n_clicks') for i in range(len(color_list))],
        [State('selected-heads', 'children')]
    )
    def update_graph(layer, *args):
        color_clicks = args[:-1]
        selected_heads_str = args[-1]
        selected_heads = [int(h) for h in selected_heads_str.split(',') if h]
        ctx = callback_context
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id.startswith('color-box-'):
                head_idx = int(trigger_id.split('-')[-1])
                if head_idx in selected_heads:
                    selected_heads.remove(head_idx)
                else:
                    selected_heads.append(head_idx)
        
        # Choose the appropriate token sets based on layer type
        if layer < 2:  # encoder self attention
            token1, token2 = tokens_in, tokens_in
        elif layer % 2 == 0:  # decoder self attention
            token1, token2 = tokens_out, tokens_out
        else:  # cross attention
            token1, token2 = tokens_out, tokens_in
        
        fig = _plot_attention_heads(attention_vals, token1, token2, layer, 
                                  head_colors=color_list, subset_heads=selected_heads)
        
        fig.update_layout(
            paper_bgcolor=dark_bg,
            plot_bgcolor=dark_bg,
            font=dict(color=text_color)
        )
        
        selected_heads_str = ','.join(map(str, selected_heads)) if selected_heads else 'none'
        return fig, selected_heads_str

    def run_app():
        app.run(port=port, debug=False)
    
    thread = Thread(target=run_app)
    thread.daemon = True
    thread.start()
    

########################################
###### Second Visualisation ############
########################################

def _create_attention_pattern(attention_weights, tokens_in, tokens_out, layer_idx, head_idx, head_color):
    attn = attention_weights[layer_idx, head_idx]
    fig = go.Figure()
    for i in range(len(tokens_in)):
        for j in range(len(tokens_out)):
            weight = attn[i, j]
            if weight > 0.05: 
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[i, j],
                    mode='lines',
                    line=dict(width=weight*3, color=head_color),
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        width=120,
        height=120,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.2, len(tokens_in) + 0.2]
        ),
        plot_bgcolor='black'
    )
    
    return fig


def run_model_dashboard(attention_data, tokens_in, tokens_out, port=8051, colormap='Viridis', height=800):
    app = dash.Dash("model_view", suppress_callback_exceptions=True)
    head_colors = _get_distinct_colors(attention_data.shape[1], colormap=colormap)
    with open('../utils/assets/styles.css', 'r') as f:
        css_content = f.read()
    
    app.index_string = f'''
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>{{%title%}}</title>
                {{%favicon%}}
                {{%css%}}
                <style>
                    {css_content}
                </style>
            </head>
            <body>
                {{%app_entry%}}
                <footer>
                    {{%config%}}
                    {{%scripts%}}
                    {{%renderer%}}
                </footer>
            </body>
        </html>
        '''
    
    def create_attention_grid():
        num_layers = attention_data.shape[0]
        num_heads = attention_data.shape[1]
        
        rows = []
        
        header_cells = [html.Th("Layers")] + [html.Th(f"Layer {h}") for h in range(num_layers)]
        rows.append(html.Tr(header_cells))
        
        for l in range(num_heads):
            cells = [html.Td(f"Head {l}", style={'color': 'white', 'textAlign': 'center'})]
            for h in range(num_layers):
                fig = _create_attention_pattern(attention_data, tokens_in, tokens_out, h, l, head_colors[l])
                
                pattern_div = html.Div([
                    dcc.Graph(
                        id=f'pattern-{l}-{h}',
                        figure=fig,
                        config={'displayModeBar': False}
                    ),
                    html.Button(
                        id=f'btn-{l}-{h}',
                        style={
                            'position': 'absolute',
                            'width': '100%',
                            'height': '100%',
                            'top': '0',
                            'left': '0',
                            'opacity': '0',
                            'cursor': 'pointer'
                        }
                    )
                ], className='attention-cell', style={'position': 'relative'})
                
                cells.append(html.Td(pattern_div))
            
            rows.append(html.Tr(cells))
        table = html.Table(rows, id='attention-grid')
        modal = html.Div(
            html.Div(
                [
                    html.Div(
                        html.Button("×", id="close-modal", className="close-button"),
                        className="modal-header"
                    ),
                    html.Div(
                        dcc.Graph(id='token-detail-view', config={'displayModeBar': False}),
                        className="modal-content"
                    ),
                ],
                className="modal-container"
            ),
            id="detail-modal",
            className="modal",
            style={'display': 'none'}
        )
        
        return html.Div([
            html.Div(table, className='grid-container'),
            modal,
            # Store for selected layer/head
            dcc.Store(id='selected-attention', data={'layer': 0, 'head': 0})
        ])
    
    app.layout = create_attention_grid()
    
    # Callback to handle clicking on attention cells
    @app.callback(
        Output('selected-attention', 'data'),
        Output('detail-modal', 'style'),
        [Input(f'btn-{l}-{h}', 'n_clicks') for h in range(attention_data.shape[0]) for l in range(attention_data.shape[1])],
        prevent_initial_call=True
    )
    def on_pattern_click(*args):
        """Handle clicks on any attention cell button"""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update
        
        # Get the ID of the clicked button
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Extract layer and head indices from button ID
        # Format is 'btn-{layer}-{head}'
        parts = button_id.split('-')
        if len(parts) == 3:
            head = int(parts[1])
            layer = int(parts[2])
            
            # Store the selected layer and head
            selected = {'layer': layer, 'head': head}
            
            # Show the modal
            return selected, {'display': 'block'}
        
        return dash.no_update, dash.no_update
    
    @app.callback(
        Output('token-detail-view', 'figure'),
        Input('selected-attention', 'data'),
        prevent_initial_call=True
    )
    def update_detail_view(selected):
        if not selected:
            return go.Figure()
        
        layer = selected['layer']
        head = selected['head']
        if layer < 2:  # encoder self attention
            token1, token2 = tokens_in, tokens_in
        elif layer % 2 == 0:  # decoder self attention
            token1, token2 = tokens_out, tokens_out
        else:  # cross attention
            token1, token2 = tokens_out, tokens_in
        # Create detailed view
        fig = _plot_attention_heads(
            attention_data, token1, token2, 
            layer, head_colors, [head], figsize=(400, 400)
        )
        fig.update_layout(
            title=dict(
                text=f"Layer {layer}, Head {head}",
                font=dict(color='white', size=20),
                # x=0.5,
                # xanchor='center'
            )
        )
        
        return fig
    
    @app.callback(
        Output('detail-modal', 'style', allow_duplicate=True),
        Input('close-modal', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_modal(n_clicks):
        return {'display': 'none'}
    
    def run_app():
        app.run(port=port, debug=False)
    
    thread = Thread(target=run_app)
    thread.daemon = True
    thread.start()
    

########################################
##### Third Visualisation ##############
########################################
custom_colorscale = [
    [0, 'rgb(165,0,38)'],    # Deep red
    [0.25, 'rgb(215,48,39)'], # Red
    [0.5, 'rgb(0,0,0)'],     # Black
    [0.75, 'rgb(8,69,148)'],     # Deep blue
    [1, 'rgb(49,130,189)'], # Blue
]


def _create_heatmap(vals, length, figsize=(240, 40), margins=dict(l=20, r=20, t=5, b=5), zmin=-2, zmax=2):
  fig = go.Figure(
    data=go.Heatmap(
        x = np.arange(length), 
        y = [1] * length, 
        z = vals, 
        type='heatmap',
        showscale=False, 
        hoverinfo='skip',
        zmax=zmax, 
        zmin=zmin
    )
  )
  fig.update_layout(
        width=figsize[0], 
        height=figsize[1],   
        margin=margins,
        xaxis=dict(
            showticklabels=False,
            showgrid=False, 
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
  
  fig.update_traces(colorscale=custom_colorscale, zmid=0)
  return fig
    

def _create_empty_figure(width=240, height=40):
    fig = go.Figure()
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, t=0, b=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def plot_query_key(raw_attention_data, normalised_attention, query_data, key_data, layer, head, selected_token, tokens_in ,tokens_out):
    rows = []
    # Create header row
    header_cells = [
        html.Th("", style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace',}),
        html.Th("Query q", style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace'}),
        html.Th("Key k", style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace'}),
        html.Th("q ⋅ k", style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace'}),
        html.Th("Softmax", style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace'})
    ]

    rows.append(html.Tr(header_cells))
    big_seq = max(len(tokens_in), len(tokens_out))
    raw_attention_data_full = raw_attention_data[layer, head, selected_token, :] / np.max(np.abs(raw_attention_data[layer, head, selected_token, :])) * 10

    for l in range(big_seq - 1, -1, -1):
        if l < len(tokens_in):
            query_word = tokens_in[l]
            query_fig =  _create_heatmap(query_data[layer, head, l, :], query_data.shape[-1])
        else:
            query_word = ""
            token_button = ""
            query_fig = _create_empty_figure()
        
        if l <= len(tokens_out):
            key_fig = _create_heatmap(key_data[layer, head, l, :], key_data.shape[-1])
            dp_fig = _create_heatmap([raw_attention_data_full[l]], 1, figsize=(40, 40), 
                             margins=dict(l=0, t=0, r=0, b=0), zmin=-2*5, zmax=2*5) 
            opacity = np.clip(normalised_attention[layer, head, selected_token, l] * 2, 0, 1)
            bg_color = f'rgba(49, 130, 189, {opacity}'
            text_color = 'white'
            token_div = html.Div(
                tokens_out[l],
                style={
                    'padding': '8px 20px',
                    'margin': '2px',
                    'backgroundColor': bg_color,
                    'color': text_color,
                    'borderRadius': '4px',
                    'textAlign': 'center',
                    'fontFamily': 'monospace',
                    'display': 'inline-block',
                    'width': 60,
                    'height' : 40 - 15
                    }
                )
        
        else:
            key_fig = _create_heatmap([0], 1)
            dp_fig = _create_empty_figure(width=40, height=20)
            token_div =  _create_empty_figure(width=60, height=40)
            opacity = 0.0
        if opacity > 0.2:
            border_color = f'3px solid rgba(74, 255, 255, {np.clip(opacity, 0.0, 0.6)})'
        else:
            border_color = '1px solid #555'

        cells = [
            html.Td(query_word,style={'color': 'white', 'textAlign': 'center', 'fontFamily': 'monospace', 
                                       'border': '3px solid rgba(74, 255, 255, 0.5)' if l == selected_token else 'rgba(0,0,0,0)', 'borderRadius': '4px', 'overflow': 'hidden'}), 
            html.Td(html.Div([
            dcc.Graph(
                id=f'pattern-{l}-query',
                figure=query_fig,
                config={'displayModeBar': False}
            )
        ], className='attention-cell', 
        style={'position': 'relative', 'border': f'{border_color}', 'borderRadius': '4px', 'overflow': 'hidden'})),
            html.Td(html.Div([
            dcc.Graph(
                id=f'pattern-{l}-key',
                figure=key_fig,
                config={'displayModeBar': False}
            )
        ], className='attention-cell', 
        style={'position': 'relative', 'border': f'{border_color}', 'borderRadius': '4px', 'overflow': 'hidden'})),            
            html.Td(html.Div([
            dcc.Graph(
                id=f'pattern-{l}-dp',
                figure=dp_fig,
                config={'displayModeBar': False}
            )
        ], className='attention-cell', 
        style={'position': 'relative', 'border': '1px solid #555', 'borderRadius': '4px', 'overflow': 'hidden'})),            
        html.Td(token_div) 
        ]
        rows.append(html.Tr(cells))
    table = html.Table(rows, id='attention-grid', style={'borderSpacing': '40px 5px'})

    return table

def _create_token_buttons(tokens_in, tokens_out, current_layer=0, is_expanded=False, selected_token_idx=0):
    
    if current_layer < 2:  # encoder self attention
        tokens = tokens_in
        token_type = 'input'
    else:  # decoder self-attention or cross-attention
        tokens = tokens_out
        token_type = 'output'
    
    buttons = []
    
    for display_idx, token in enumerate(reversed(tokens)):
        actual_idx = len(tokens) - 1 - display_idx
        is_selected = is_expanded and actual_idx == selected_token_idx
        
        style = {
            'background': 'rgba(33,150,243,0.3)' if is_selected else 'transparent',
            'padding': '5px',
            'margin': '3px 0',
            'borderRadius': '3px',
            'width': '100%',
            'cursor': 'pointer',
            'border': 'none',
            'color': 'white',
            'textAlign': 'left',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap'
        }
        label = token + "          "
        label += '−' if is_selected else '+'
        
        buttons.append(html.Button(
            label, 
            id={'type': 'token-button', 'index': actual_idx},
            style=style
        ))
    
    # Return container with buttons (now in reversed visual order)
    return html.Div(buttons, id='token-buttons-container', style={'width': '100%'})

def run_neuron_dashboard(raw_attention, normalised_attention, query_data, key_data, tokens_in, tokens_out, port=8052):
    num_heads = normalised_attention.shape[1]
    num_layers = normalised_attention.shape[0]
    head_colors = ['rgb(10,235,255)'] * num_heads
    app = Dash("neuron_view", suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        dcc.Store(id='view-state', data={'expanded': False, 'token_idx': 0}),
        
        # Control panel for layer and head selection
        html.Div([
            html.Div([
                # Layer dropdown
                html.Div([
                    html.Label("Layer:", style={'color': 'white', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='layer-dropdown',
                        options=[{'label': str(i), 'value': i} for i in range(num_layers)],
                        value=0,
                        style={'width': '80px'},
                        className='dark-dropdown'
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px'}),
                
                # Head dropdown
                html.Div([
                    html.Label("Head:", style={'color': 'white', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='head-dropdown',
                        options=[{'label': str(i), 'value': i} for i in range(num_heads)],
                        value=0,
                        style={'width': '80px'},
                        className='dark-dropdown'
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                
                # Layer type indicator
                html.Div(id='layer-type-indicator', style={
                    'marginLeft': '20px', 
                    'color': 'rgba(255,255,255,0.8)',
                    'display': 'flex',
                    'alignItems': 'center'
                }),
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'})
        ], style={
            'backgroundColor': '#222',
            'padding': '10px',
            'marginBottom': '15px',
            'borderRadius': '4px'
        }),
        
        # Main content with token list and visualization
        html.Div([
            # Left panel: token buttons
            html.Div(
                id='left-panel',
                children=[_create_token_buttons(tokens_in, tokens_out)],  # Initial state
                style={
                    'width': '150px',
                    'backgroundColor': '#222',
                    'padding': '10px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'borderRadius': '4px',
                    'marginRight': '15px',
                    'overflowY': 'auto',
                    'maxHeight': 'calc(100vh - 120px)'  # Allow scrolling for many tokens
                }
            ),
            
            # Right panel: visualization area
            html.Div(
                id='right-panel',
                style={
                    'flex': 1, 
                    'backgroundColor': '#1E1E1E',
                    'padding': '15px',
                    'borderRadius': '4px',
                    'overflowX': 'auto'  # Allow horizontal scrolling if needed
                }
            ),
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': 'calc(100vh - 120px)'})
    ], style={'padding': '15px', 'backgroundColor': 'black', 'height': '100vh', 'boxSizing': 'border-box'})
    
    
    @app.callback(
        Output('layer-type-indicator', 'children'),
        Input('layer-dropdown', 'value')
    )
    def update_layer_type_indicator(layer_value):
        if layer_value is None:
            return ""
        
        if layer_value < 2:  # encoder self attention
            return html.Div([
                html.Span("Type: ", style={'marginRight': '5px'}),
                html.Span("Encoder Self-Attention", 
                         style={'backgroundColor': 'rgba(76, 175, 80, 0.3)', 'padding': '3px 8px', 'borderRadius': '4px'})
            ])
        elif layer_value % 2 == 0:  # decoder self attention
            return html.Div([
                html.Span("Type: ", style={'marginRight': '5px'}),
                html.Span("Decoder Self-Attention", 
                         style={'backgroundColor': 'rgba(33, 150, 243, 0.3)', 'padding': '3px 8px', 'borderRadius': '4px'})
            ])
        else:  # cross attention
            return html.Div([
                html.Span("Type: ", style={'marginRight': '5px'}),
                html.Span("Cross-Attention", 
                         style={'backgroundColor': 'rgba(255, 152, 0, 0.3)', 'padding': '3px 8px', 'borderRadius': '4px'})
            ])
    
    @app.callback(
        Output('left-panel', 'children'),
        Input('layer-dropdown', 'value'),
        Input('view-state', 'data')
    )
    def update_left_panel(layer_value, view_state):
        if layer_value is None:
            layer_value = 0
            
        is_expanded = view_state['expanded']
        selected_token_idx = view_state['token_idx']
            
        return _create_token_buttons(
            tokens_in, 
            tokens_out, 
            current_layer=layer_value, 
            is_expanded=is_expanded, 
            selected_token_idx=selected_token_idx
        )
    
    @app.callback(
        Output('right-panel', 'children'),
        Input('view-state', 'data'),
        Input('layer-dropdown', 'value'),
        Input('head-dropdown', 'value')
    )
    def update_visualization(view_state, layer_value, head_value):
        if layer_value is None or head_value is None:
            layer_value = 0
            head_value = 0
        
        # Determine which tokens to use based on layer type
        if layer_value < 2:  # encoder self attention
            token1, token2 = tokens_in, tokens_in
        elif layer_value % 2 == 0:  # decoder self attention
            token1, token2 = tokens_out, tokens_out
        else:  # cross attention
            token1, token2 = tokens_out, tokens_in
    
        if not view_state['expanded']:
            # Regular view: show attention patterns
            fig = _plot_attention_heads(
                normalised_attention, 
                token1, 
                token2,
                layer_idx=layer_value, 
                subset_heads=[head_value], 
                head_colors=head_colors
            )
            return dcc.Graph(id='attention-graph', figure=fig, config={'displayModeBar': False})
        else:
            token_idx = view_state['token_idx']
            max_idx = len(token1) - 1
            if token_idx > max_idx:
                token_idx = 0  # Default to first token if out of range
            
            return plot_query_key(
                raw_attention, 
                normalised_attention, 
                query_data, 
                key_data, 
                layer_value, 
                head_value, 
                token_idx, 
                token1, 
                token2
            )
    
    @app.callback(
        Output('view-state', 'data'),
        [
            Input({'type': 'token-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
            Input('layer-dropdown', 'value'),
            Input('head-dropdown', 'value')
        ],
        [State('view-state', 'data')]
    )
    def update_view_state(button_clicks, layer_value, head_value, current_state):
        ctx = callback_context
        if not ctx.triggered:
            return current_state
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id in ['layer-dropdown.value', 'head-dropdown.value'] and current_state['expanded']:
            return {'expanded': True, 'token_idx': 0}
        
        # If a token button was clicked
        if 'token-button' in triggered_id:
            try:
                button_id = json.loads(triggered_id)
                token_idx = button_id['index']
                
                # Toggle expanded state or switch to a different token
                if not current_state['expanded']:
                    return {'expanded': True, 'token_idx': token_idx}
                else:
                    if token_idx == current_state['token_idx']:
                        # Clicking the same token collapses the view
                        return {'expanded': False, 'token_idx': 0}
                    else:
                        # Clicking a different token switches to that token
                        return {'expanded': True, 'token_idx': token_idx}
            except Exception as e:
                print(f"Error parsing button ID: {e}")
                return current_state
        
        return current_state
    
    def run_app():
        app.run(port=port, debug=False)
    
    thread = Thread(target=run_app)
    thread.daemon = True
    thread.start()