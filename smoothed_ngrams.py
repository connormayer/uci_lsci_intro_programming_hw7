from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

def fit_ngram_model(training_data):
    """
    Assume that training_data will be a list of lists of strings in the same
    format we've seen in class.
    """
    ngram_sents, vocab = padded_everygram_pipeline(2, training_data)
    model = MLE(2)
    model.fit(ngram_sents, vocab)
    return model

if __name__ == "__main__":
    # You can write code to test your function here
    pass
