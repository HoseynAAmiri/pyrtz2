from dash import dcc

import plotly.graph_objects as go


def render(id: str, title: str, xaxis: str) -> dcc.Graph:
    fig = make_fig(title, xaxis)

    if 'contact' in id:
        fig.add_vline(x=0, line=dict(color='red', width=1.2))
        fig.add_hline(y=0, line=dict(color='red', width=1.2))

    return dcc.Graph(
        figure=fig,
        id=id,
        mathjax=True,
        style={
            'width': '50%',
        }
    )


def make_fig(title: str, xaxis: str) -> go.Figure:
    fig = go.Figure()

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False,
        transition={'duration': 500},
        height=400
    )

    fig.update_annotations(yshift=10)

    fig.update_xaxes(
        showgrid=False,
        ticks='outside',
        tickwidth=2,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        title_text=xaxis,
        tickprefix=r"$",
        ticksuffix=r"$",
    )

    fig.update_yaxes(
        showgrid=False,
        ticks='outside',
        tickwidth=2,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        title_text=r"$Force \text{ (N)}$",
        tickprefix=r"$",
        ticksuffix=r"$",
    )

    return fig


def update_fig(fig: go.Figure, x, y, mode: str, color: str, hover: bool = False, y2: bool = False) -> go.Figure:
    hover_texts = None
    if hover:
        hover_texts = [f'Index: {i}' for i in range(len(x))]

    if not fig['data']:
        if mode == 'lines':
            color_arg = {'line': {'color': color}}
        elif mode == 'markers':
            color_arg = {'marker': {'color': color}}
        else:
            color_arg = {}

        trace = go.Scattergl(
            x=x,
            y=y,
            mode=mode,
            text=hover_texts,
            hoverinfo='text',
            hoverlabel={'bgcolor': 'red'},
            **color_arg,
        )

        fig.add_trace(trace)
    else:
        fig.data[0].x = x
        fig.data[0].y = y
        fig.data[0]['text'] = hover_texts

    return fig


def update_contact_line(cp: int, fig: dict | go.Figure) -> go.Figure:
    if isinstance(fig, dict):
        fig = go.Figure(fig)
    data_x = fig.data[0].x
    data_y = fig.data[0].y
    fig.layout['shapes'][0]['x0'] = data_x[cp]
    fig.layout['shapes'][0]['x1'] = data_x[cp]
    fig.layout['shapes'][1]['y0'] = data_y[cp]
    fig.layout['shapes'][1]['y1'] = data_y[cp]
    fig.layout['title']['text'] = fr"$\text{{Selected Contact Point: {cp}}}$"

    return fig
