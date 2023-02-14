from transformers import pipeline
summarizer=pipeline('summarization')

def abs_summary(article):
    l = len(list(article.split()))
    return summarizer(article,max_length=l,min_length=int(l/2),do_sample=False)[0]['summary_text']