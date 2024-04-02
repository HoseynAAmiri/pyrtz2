import plotly.graph_objects as go


def make_fig(title: str, xaxis: str) -> go.Figure:
    layout = go.Layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    fig = go.Figure(
        data=[],
        layout=layout,
    )

    fig.update_annotations(yshift=10)

    tick_Coll = {
        "xaxis": {},
        "xaxis": {},
    }
    tick_Sett = {
        "tickprefix": r"$",
        "ticksuffix": r"$",
    }
    tick_Coll["xaxis"] = tick_Sett.copy()
    tick_Coll["yaxis"] = tick_Sett.copy()
    tick_Coll["xaxis2"] = tick_Sett.copy()
    tick_Coll["yaxis2"] = tick_Sett.copy()
    fig.update_layout(
        tick_Coll,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False,
        # clickmode='event+select'
    )

    fig.update_xaxes(
        showgrid=False,
        ticks='outside',
        tickwidth=2,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        title_text=xaxis,
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
    )

    return fig
