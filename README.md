# MultiClass
Machine Learing, using ML.Net to generate models from news items processed by my LiebFeed processing application. This is my "Hello World" application for ML.

A quick application generate models from my CosmosB news feed database. The Features are Title and Description, and the Label is source. To generate the training datset, news items are extracted and saved in a TSV format. These text files are then submitted to ML.Net so as to generate models. 

This program uses thes MultiClass learners to geneate the models:
- StochasticDualCoordinateAscent 
- Logistic Regression

These models are loaded into the LiebTechReact wep application (http://LiebTechReact.azurewebsites.net/mlnet) - so as to compare their predictions to the actual item source.

The application extracts the TSV files in 5k increments. Once enough data is loaded into the database to hit the next plateau, MultiClass will extract the training file, generate the model, run evaluations and run some Predictions to compare it's accuracy to previous models.
