from dash import dcc, html
from dash.dependencies import Input, Output
from app import app, navbar
from pages import home, image_manipulation, tumor_detection

# Main layout that includes the navbar and a location component for navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Callback to render the correct page
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/image-manipulation':
        return image_manipulation.layout
    elif pathname == '/tumor-detection':
        return tumor_detection.layout
    else:
        return home.layout  # Default to home page

if __name__ == '__main__':
    app.run_server(debug=True)
