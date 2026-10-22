import plotly.express as px


def time_series_figure(frame, x="date", y="Appliances"):
    return px.line(
        frame,
        x=x,
        y=y,
        title="Household appliance energy use over time",
        labels={x: "Timestamp", y: "Appliances (Wh per 10 min)"},
    )
