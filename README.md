# freesearch

Building hotel room search with self-querying retrieval
In this example we'll walk through how to build and iterate on a hotel room search service that leverages an LLM to generate structured filter queries that can then be passed to a vector store.

For an introduction to self-querying retrieval check out the docs.

Imports and data prep
In this example we use ChatOpenAI for the model and ElasticsearchStore for the vector store, but these can be swapped out with an LLM/ChatModel and any VectorStore that support self-querying.

Download data from: https://www.kaggle.com/datasets/keshavramaiah/hotel-recommendation