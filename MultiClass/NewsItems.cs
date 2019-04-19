using Microsoft.ML.Data;

namespace MultiClass
{
    public class NewsItem
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Feed { get; set; }
        [LoadColumn(2)]
        public string SiteSection { get; set; }
        [LoadColumn(3)]
        public string Title { get; set; }
        [LoadColumn(4)]
        public string Description { get; set; }
    }

    public class SectionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Section;
    }

    public class CosmosNewsItem
    {
        public string title { get; set; }
        public string description { get; set; }
        public string partionKey { get; set; }
        public string siteSection { get; set; }
        public string id { get; set; }
    }
}
