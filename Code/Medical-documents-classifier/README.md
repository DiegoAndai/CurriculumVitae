# Medical documents classifier

On this project, I've worked together with:

Vicente Valencia, PUC student and research partner. [Click here to go to Vicente's github](https://github.com/Vince-Valence).

The goal was to construct a document classifier using word embeddings. The embeddings were obtained using the SkipGram model and two different datasets, one from wikipedia and another from medline. The classifier is mean to separate between two classes of medical papers: Primary Studies and Systematic Reviews. For this purpose every document was represented by a vector which was the result of a max-pooling process over a matrix constructed with the document's abstract's words embeddings. A KNN classifier was used, this because we didn't want a more sophisticated classifier biasing the results, the KNN offers a simple method that relies entirely on the representation obtained from the embeddings, so this way we can accurately analyze the performance of our representation method. Many months are invested in this projects, which was born and migrated from [this github repository](https://github.com/DiegoAndai/Deep-learning-framework-research) to [this one](https://github.com/Vince-Valence/medical-documents-classifier).

As this is a large project there isn't many code that I haven't contributed to in some way or another, so instead of listing the things that I did, the piece of code that I didn't do is `full_classification_pipeline.py`, I leave it here because is essential to the project, as this uses all the other code and puts it together to obtain results. This is a work in progress so the official results aren't out yet, still I provided some results (on the Some_results folder). Bellow I explain briefly some files and folders:

- `full_classification_pipeline.py`: Uses other files as `pipeline_utils.py` and `document_space.py` to basically open the documents, process them and classify them.

- `document_space.py`: Implements the KNN classifier and the vector construction.

- `pipeline_utils.py`: Implements tools to obtain and analyze results from the classification.

- `WordAppearanceAnalysis`: This folder contains code to analyze the word's density over documents, specifically obtaining ratios which told what word appeared more on which class of documents. Here `probability_Study.py` implements the ratio calculation using counts obtained from `word_Probability.py`.

- `Some_results`: Some results, here:

    - `Best_model_results.txt`: Metrics for the best model at this moment

    - `BOW`: Metric for classifying the same documents but with BOW method.

    - `TSNE`: TSNE visualizations for documents using different methods, the ones using our method are labelled as w2v, the span indicates how many words were used from the abstracts. None means all words were used.
