from transformers import pipeline
summarizer=pipeline('summarization')

def abs_summary(article):
    l = len(list(article.split()))
    abs_summ = summarizer(article,max_length=l,min_length=int(l/2),do_sample=False)[0]['summary_text']
    res= {
        "abs_summ":abs_summ
    }

    return res