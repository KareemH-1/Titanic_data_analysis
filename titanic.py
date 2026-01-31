import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic.csv')
df.head()


sns.set_theme(style="darkgrid")
sns.set_palette("bright")


sns.pairplot(df)

numSurvived = df[df['Survived'] == 1].shape[0]
numNotSurvived = df[df['Survived'] == 0].shape[0]
plot = sns.barplot(x=['Survived', 'Not Survived'], y=[numSurvived, numNotSurvived])

plot.set_title('Titanic Survival Counts')
plot.set_xlabel('Status')
plot.set_ylabel('Number of Passengers')
plot.legend(['Survival Status'])

plt.show()

print("Number of Survivors:", numSurvived)
print("Number of Non-Survivors:", numNotSurvived)
print("number of survivors is less than non survivors , almost by half")


sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()
print("the age group with heighest survival rate appears to be children then the elderly")
print("which could be because they were given priority during evacuation or received more assistance")



#fare vs pClass
sns.barplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()
print("As the passenger class increases, the fare also increases")


sns.countplot(x='Embarked', data=df)
plt.title('Number of Passengers')
plt.xlabel('Place of embark')
plt.ylabel('Number of Passengers')
plt.show()

print("Most passengers embarked from S, then C then Q")


sns.boxplot(x="Survived", y="Fare", data=df)
plt.title('Fare vs Survived')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

print("Passengers who survived paid higher fares on average compared to those who did not survive.")


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print("There is strong correlation betwwen survived and fare paid")
print("also a strong correlation between age and fare")

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Scatterplot of Age vs Fare by Survival')
plt.show()
print("younger passengers who paid higher fares had a higher survival rate compared to older passengers who paid lower fares.")

sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Boxplot of Age by Passenger Class')
plt.show()
print("the graph shows that the heigher the class the heigher the age on average")


sns.barplot(x='Embarked', y='Fare', data=df)
plt.title('Barplot of Fare by embark')
plt.show()
print("location C has the highest fare on average then S then Q")


import dash
from dash import dcc, html
import plotly.express as px

app = dash.Dash(__name__)

figSurvival = px.bar(
    x=['Survived', 'Not Survived'],
    y=[numSurvived, numNotSurvived],
    labels={'x': 'Status', 'y': 'Number of Passengers'},
    title='Titanic Survival Counts'
)

figAgeHist = px.histogram(
    df, x='Age', color='Survived', barmode='stack', nbins=30,
    labels={'Survived': 'Survived', 'Age': 'Age'},
    title='Age Distribution by Survival Status'
)

figFarePclass = px.bar(
    df.groupby('Pclass')['Fare'].mean().reset_index(),
    x='Pclass', y='Fare',
    labels={'Pclass': 'Passenger Class', 'Fare': 'Average Fare'},
    title='Average Fare by Passenger Class'
)

embarkedCounts = df['Embarked'].value_counts().reset_index()
embarkedCounts.columns = ['Embarkation Port', 'Number of Passengers']
figEmbarked = px.bar(
    embarkedCounts,
    x='Embarkation Port', y='Number of Passengers',
    labels={'Embarkation Port': 'Embarkation Port', 'Number of Passengers': 'Number of Passengers'},
    title='Number of Passengers by Embarkation Port'
)

figFareSurvived = px.box(
    df, x='Survived', y='Fare',
    labels={'Survived': 'Survived', 'Fare': 'Fare'},
    title='Fare vs Survived'
)

figCorr = px.imshow(
    df.corr(numeric_only=True),
    text_auto=True,
    color_continuous_scale='RdBu',
    title='Correlation Heatmap'
)

figScatter = px.scatter(
    df, x='Age', y='Fare', color='Survived',
    labels={'Age': 'Age', 'Fare': 'Fare', 'Survived': 'Survived'},
    title='Scatterplot of Age vs Fare by Survival'
)

figAgePclass = px.box(
    df, x='Pclass', y='Age',
    labels={'Pclass': 'Passenger Class', 'Age': 'Age'},
    title='Boxplot of Age by Passenger Class'
)

figFareEmbarked = px.bar(
    df.groupby('Embarked')['Fare'].mean().reset_index(),
    x='Embarked', y='Fare',
    labels={'Embarked': 'Embarkation Port', 'Fare': 'Average Fare'},
    title='Average Fare by Embarkation Port'
)

app.layout = html.Div([
    html.H1('Titanic Data Dashboard'),
    dcc.Graph(figure=figSurvival),
    dcc.Graph(figure=figAgeHist),
    dcc.Graph(figure=figFarePclass),
    dcc.Graph(figure=figEmbarked),
    dcc.Graph(figure=figFareSurvived),
    dcc.Graph(figure=figCorr),
    dcc.Graph(figure=figScatter),
    dcc.Graph(figure=figAgePclass),
    dcc.Graph(figure=figFareEmbarked),
])



