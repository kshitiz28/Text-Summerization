# importing
from transformers import pipeline
#loading summarizer pipeline
summarizer=pipeline('summarization')
article = """
Thank you. Thank you. 
Thank you to Vice President Pence. 
He's a good guy. We've done a great job together.
And Merry Christmas, Michigan. Thank you, Michigan. 
What a victory we had in Michigan. What a victory was that. One of the greats.
Was that the greatest evening? 
But I'm thrilled to be here with thousands of hardworking patriots as we celebrate the miracle of Christmas, the greatness of America and the glory of God. Thank you very much. And did you notice that everybody is saying Merry Christmas again? Did you notice? Saying Merry Christmas. I remember when I first started this beautiful trip, this beautiful journey, I just said to the First Lady, "You are so lucky. I took you on this fantastic journey. It's so much fun. They want to impeach you. They want to do worse than that." By the way, by the way, by the way, it doesn't really feel like we're being impeached. The country is doing better than ever before. We did nothing wrong. We did nothing wrong. And we have tremendous support in the Republican Party like we've never had before. Nobody's ever had this kind of support. But this sacred season, our country is thriving and it's thriving truly like it has never, it has never happened before to the extent what's happening now. And by the way, your state, because of us, not because of local government, but because of us, because of the job that we've done. Because I understand she's not fixing those potholes. That's what the word is. It was all about roads and they want to raise those gasoline taxes on you. We don't want to do that. 
"""
print(summarizer(article,max_length=200,min_length=100,do_sample=False)[0]['summary_text'])