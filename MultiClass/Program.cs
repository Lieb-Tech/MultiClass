using Microsoft.Azure.Documents.Client;
using Microsoft.Azure.Documents.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MultiClass
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static MLContext _mlContext;    
        /*
         *https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification
         */
         
        private static CosmosDB client = new CosmosDB();

        static void Main(string[] args)
        {
            
            client.OpenConnection();            
            _mlContext = new MLContext(seed: 0);

            var basePath = Path.Combine(_appPath, "..", "..", "..");

            var mb = new FileBuilder(client);
            mb.BuildModels(basePath).Wait();

            var mt = new ModelTrainer(_mlContext, basePath);
            mt.BuildAndTrain();

            var pt = new PredictionTest(_mlContext, basePath);
            pt.RunPredictions();
        }        
    }
}
