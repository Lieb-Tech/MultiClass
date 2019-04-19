# MultiClass
ML.Net based project to load and "learn" items processed by my LiebFeed processing application.

A quick application generate models from my CosmosB news feed database. The Features are Title and Description, and Label is source. To generate the training datset, 15,000 items are extracted and saved in a TSV format. 

This program uses 3 MultiClass learners to geneate the models:
- StochasticDualCoordinateAscent 
- Logistic Regression
- Naive Bayes

These models are loaded into the LiebTechReact wep application (http://LiebTechReact.azurewebsites.net/mlnet) - so as to compare their predictions to the actual item source.

When the Naive Bayes was used in the web application, the predictions were almost always incorrect (same source every time), so I am not using it there.
