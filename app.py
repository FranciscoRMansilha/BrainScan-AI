import dash
import dash_bootstrap_components as dbc

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions=True)

server = app.server

# Custom CSS for hamburger menu and other elements
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''
# Navbar with 3-bar icon for page navigation
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Brain Tumor Detection", className="ms-2"),
            dbc.Button(
                "🧠",  
                id="navbar-toggler",
                className="ms-auto",
                color="primary",
                style={"font-size": "24px", "border": "none", "background": "none", "color": "white"},
            ),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavLink("Home", href="/", active="exact"),
                        dbc.NavLink("Image Manipulation", href="/image-manipulation", active="exact"),
                        dbc.NavLink("Tumor Detection", href="/tumor-detection", active="exact")
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            )
        ]
    ),
    color="info",
    dark=True,
    className="mb-5"
)
