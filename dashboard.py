import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# Load data
data = pd.read_csv("data/winequality-red.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Drop duplicate rows
data.drop_duplicates(keep='first', inplace=True)

# Label quality into Good(1) and Bad(0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)

# Drop the target variable
X = data.drop('quality', axis=1)

# Set the target variable as label
y = data['quality']

scaler = StandardScaler()

# Split the data into training and testing sets (20% testing and 80% training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create an object of the logistic regression model
logreg_model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = logreg_model.predict(X_test)

# Create the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Compute the precision of the model
precision = precision_score(y_test, y_pred)

# Compute the recall of the model
recall = recall_score(y_test, y_pred)

# Compute the F1 score of the model
f1 = f1_score(y_test, y_pred)

# Compute the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

# Create the Dash app
app = dash.Dash(__name__)

# Add CSS styles
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css'
})

# Define available parameter options for comparison dropdowns
param_options = [
    {'label': 'Fixed Acidity', 'value': 'fixed acidity'},
    {'label': 'Volatile Acidity', 'value': 'volatile acidity'},
    {'label': 'Citric Acid', 'value': 'citric acid'},
    {'label': 'Residual Sugar', 'value': 'residual sugar'},
    {'label': 'Chlorides', 'value': 'chlorides'},
    {'label': 'Free Sulfur Dioxide', 'value': 'free sulfur dioxide'},
    {'label': 'Total Sulfur Dioxide', 'value': 'total sulfur dioxide'},
    {'label': 'Density', 'value': 'density'},
    {'label': 'pH', 'value': 'pH'},
    {'label': 'Sulphates', 'value': 'sulphates'},
    {'label': 'Alcohol', 'value': 'alcohol'}
]

# Define the layout of the dashboard
app.layout = html.Div(
    className='container',
    children=[
        html.H1('Wine Quality Prediction Dashboard', style={'text-align': 'center'}),

        html.H2('Model Evaluation Metrics'),

        html.Div(
            className='row',
            children=[
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Confusion Matrix'),
                        html.Table(
                            className='u-full-width',
                            children=[
                                html.Tr(children=[
                                    html.Th(''),
                                    html.Th('Predicted 0'),
                                    html.Th('Predicted 1'),
                                ]),
                                html.Tr(children=[
                                    html.Th('Actual 0'),
                                    html.Td(f'True Negative: {confusion_mat[0, 0]}'),
                                    html.Td(f'False Positive: {confusion_mat[0, 1]}'),
                                ]),
                                html.Tr(children=[
                                    html.Th('Actual 1'),
                                    html.Td(f'False Negative: {confusion_mat[1, 0]}'),
                                    html.Td(f'True Positive: {confusion_mat[1, 1]}'),
                                ]),
                            ],
                            style={'margin': '20px'}
                        ),
                    ]
                ),
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Model Performance Metrics'),
                        html.Table(
                            className='u-full-width',
                            children=[
                                html.Tr(children=[
                                    html.Th('Accuracy'),
                                    html.Td(f'{accuracy:.2f}'),
                                ]),
                                html.Tr(children=[
                                    html.Th('Precision'),
                                    html.Td(f'{precision:.2f}'),
                                ]),
                                html.Tr(children=[
                                    html.Th('Recall'),
                                    html.Td(f'{recall:.2f}'),
                                ]),
                                html.Tr(children=[
                                    html.Th('F1 Score'),
                                    html.Td(f'{f1:.2f}'),
                                ]),
                                html.Tr(children=[
                                    html.Th('AUC Score'),
                                    html.Td(f'{auc_score:.2f}'),
                                ]),
                            ],
                            style={'margin': '20px'}
                        ),
                    ]
                ),
            ]
        ),

        html.H2('Receiver Operating Characteristic (ROC) Curve'),

        html.Div(
            className='row',
            children=[
                html.Div(
                    className='six columns',
                    children=[
                        html.H4('ROC Curve'),
                        dcc.Graph(
                            id='roc-curve',
                            figure={
                                'data': [
                                    {'x': fpr, 'y': tpr, 'type': 'line', 'name': 'ROC curve'},
                                    {'x': [0, 1], 'y': [0, 1], 'type': 'line', 'name': 'Random'},
                                ],
                                'layout': {
                                    'title': 'Receiver Operating Characteristic (ROC) Curve',
                                    'xaxis': {'title': 'False Positive Rate'},
                                    'yaxis': {'title': 'True Positive Rate'},
                                    'margin': {'t': 30, 'b': 30, 'l': 30, 'r': 30},
                                }
                            }
                        ),
                    ]
                ),
                html.Div(
                    className='six columns',
                    children=[
                        html.H4('Compare Parameter Effects'),
                        dcc.Dropdown(
                            id='x-param-dropdown',
                            options=param_options,
                            value='fixed acidity',
                            clearable=False
                        ),
                        dcc.Dropdown(
                            id='y-param-dropdown',
                            options=param_options,
                            value='volatile acidity',
                            clearable=False
                        ),
                        dcc.Graph(
                            id='param-effects',
                        ),
                    ]
                ),
            ]
        ),
    ]
)


# Callback function to update the parameter effects graph based on dropdown selections
@app.callback(
    Output('param-effects', 'figure'),
    Input('x-param-dropdown', 'value'),
    Input('y-param-dropdown', 'value')
)
def update_param_effects(x_param, y_param):
    fig = px.scatter(
        data,
        x=x_param,
        y=y_param,
        color='quality',
        title=f'Effect of {x_param.capitalize()} and {y_param.capitalize()} on Wine Quality',
        labels={'quality': 'Wine Quality'},
        hover_name='quality',
        hover_data={'quality': True}
    )
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
