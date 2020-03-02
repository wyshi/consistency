from transformers import pipeline
nlp = pipeline('question-answering')

context = "A: Hello! How are you doing today? B: Good, thank you. How are you?  A: I am good, thank you.  Have you heard of the charity Save the Children B: Yes, I have heard about it. I would like to donate a dollar to it. A: That is wonderful, it is so important to provide education and support to the children.  I am sure you will agree that this is a great cause.  Do you donate to charities? B: Yes, I do donate to several charities. A: Are you sure you want to donate a dollar to Save the Children?  Children all over the world are suffering due to poverty, war, and other issues.  Would you consider donating a little bit of your incoming task payment? B: I may donate a dollar.  I don't have a lot of money right now but I would like to help. A: That is great to hear.  I know your donation is a small amount but the impact this will have on the world can be significant.  Your donation would be directly deducted from your task payment. B: That is true. I will donate 0.10 dollar to the charity. A: Thank you so much for your donation, it will help many children. B: Thank you.  I hope you will continue to donate to the charity."

print(context)

while True:
    question = input("q: ")
    print(nlp({
        'question': question,
        'context': context
    }))