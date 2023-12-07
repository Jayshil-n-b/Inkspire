import pickle
import numpy as np
import tensorflow as tf
import transformers
from tensorflow_addons.metrics import F1Score

def tokenizer_preprocessing(texts, tokenizer):
    encoded_dict = tokenizer.batch_encode_plus(
        texts,
        return_token_type_ids=False,
        pad_to_max_length=True, # the length of all texts will be equal to a text which has the maximum tokens
        max_length=max_len
    )
    return np.array(encoded_dict['input_ids']) # convert a list to an array

model = tf.keras.models.load_model('my_model.keras', custom_objects={"TFAutoModel": transformers.TFAutoModel, "TFDistilBertModel": transformers.TFDistilBertModel, "AdamWeightDecay": transformers.AdamWeightDecay, "F1Score": F1Score})
tokenizer = pickle.load(open(b"tokenizer_1.h5","rb"))
max_len = pickle.load(open(b"max_len_1.unknown","rb"))

article = """
Could X go bankrupt under Elon Musk? Elon Musk's profane attack on advertisers boycotting X, formerly known as Twitter, has baffled experts. If advertisers keep leaving and don't come back, can X survive? In April, I sat down with Musk for the first of his many chaotic interviews about his acquisition of X.  He said something that, in hindsight, was rather revealing, but which passed me by at the time.  Talking about advertising, he said: "If Disney feels comfortable advertising children's movies [on Twitter], and Apple feels comfortable advertising iPhones, those are good indicators that Twitter is a good place to advertise." Seven months later, Disney and Apple are no longer advertising on X - and Musk is telling companies that have left to "Go [expletive] yourself." The companies paused adverts after an investigation by a US organisation, Media Matters for America, flagged ads appearing next to pro-Nazi posts. X fiercely challenged the report, questioning its research methods, and launched a lawsuit against the organisation. In a fiery interview on Wednesday, Musk also used the "b" word - bankruptcy, in a sign of just how much the ad boycott is damaging the company's bottom line. For a company he bought for $44bn (£35bn) last year, bankruptcy might sound unthinkable. But it is possible.  To understand why, you have to look at how reliant X is on advertising revenue - and why advertisers are not coming back.  Although we don't have the latest figures, last year around 90% of X's revenue was from advertising. It is the heart of the business. On Wednesday Musk more than hinted at this.  "If the company fails… it will fail because of an advertiser boycott. And that will be what bankrupts the company." he said.  Mark Gay, chief client officer at marketing consultancy at Ebiquity, which works with hundreds of companies, says there is no sign anyone is returning. "The money has come out and nobody is putting a strategy in place for reinvesting there," he says. On Friday, retail giant Walmart announced it was no longer advertising on X. After Musk had told advertisers who quit X where to go in Wednesday's interview at the New York Times DealBook Summit, he said something that made advertisers wince even harder. "Hi Bob", he said - a reference to the chief executive of Disney, Bob Iger.  When Musk puts chief executives "in his crosshairs" like this they will be even more reticent to be involved with X, says Lou Paskalis, of marketing consultancy AJL Advisory. Jasmine Enberg, principal analyst at Insider Intelligence, adds: "It doesn't take a social media expert to understand and to know that publicly and personally attacking advertisers and companies that pay X's bills is not going to be good for business." If advertisers are gone for good, what does Musk have? When I interviewed him in April, it was clear he understood that subscriptions on X were not going to replace advertising money. "If you have a million people that are subscribed for, let's say, $100 a year-ish, that's $100m. That's a fairly small revenue stream relative to advertising," he told me. In 2022, Twitter's advertising revenue was around $4bn. Insider Intelligence estimates this year it will drop to $1.9bn.  The company has two major outlays. The first is its staffing bill. Musk has cut X to the bone already, laying off thousands.  The second is servicing the loans Musk took out to buy Twitter, totalling about $13bn. Reuters has reported that the company now has to pay $1.2bn or so in interest payments every year.  If the company cannot service the interest on its loans or afford to pay staff then, yes, X really could go bankrupt.  But that would be an extreme scenario that Musk would surely want to avoid.  He has options. By far the simplest thing for Musk would be to put more of his money in - but it sounds like he doesn't want to do that.  Musk could try to renegotiate with the banks for less onerous interest payments. He could ask, for example, for "payment in kind" interest - where payments are delayed.  But if renegotiation does not work and the banks don't get their money, then bankruptcy could be the only option, and at that point the banks could try to push for a change in management.  "It would be very messy and complex," says Jared Ellias, a professor of law at Harvard Law School. "And it would be extremely challenging. It would create a lot of news because he would constantly get deposed and have to testify in court." It could be terrible for Musk's business reputation, and would also impact how Musk could borrow money in the future.  And in a bankruptcy scenario, would X simply stop working?  "I find that to be very hard to believe," says Ellias. "If that happened, it'd be because Elon decided to pull the rug out. But even then, if he were to do that, the creditors would have the option of pushing the company into bankruptcy, getting a trustee appointed and turning the lights back on," he says. The obvious solution to all these problems for X is to simply find another revenue stream - and fast. Musk is certainly trying. He has launched a new audio and video calls service. Last month he streamed himself playing video games - he hopes X can compete with apps like Twitch. He wants X to become the "everything app", covering everything from chat to online payments. According to the New York Times, which got hold of the pitch deck Musk was giving to investors last year, X was supposed to bring in $15m from a payments business in 2023, growing to about $1.3bn by 2028.  This video can not be played Watch: Elon Musk's unexpected BBC interview... in 90 seconds (April 2023) X is also sitting on a huge treasure trove of data, and its vast archive of conversations can be used to train chatbots. Musk believes this data is vastly valuable.  So X does have potential. But in the short term, none of these options plug the hole advertisers have left.  It's why Musk's profane outburst was so baffling to many.  "I don't have any theories that make sense," Paskalis says. "There is a revenue model in his head that eludes me."
"""

article_vec = tokenizer_preprocessing([article], tokenizer)

classes_name = ['ARTS & CULTURE',
 'BUSINESS & FINANCES',
 'COMEDY',
 'CRIME',
 'DIVORCE',
 'EDUCATION',
 'ENTERTAINMENT',
 'ENVIRONMENT',
 'FOOD & DRINK',
 'GROUPS VOICES',
 'HOME & LIVING',
 'IMPACT',
 'MEDIA',
 'MISCELLANEOUS',
 'PARENTING',
 'POLITICS',
 'RELIGION',
 'SCIENCE & TECH',
 'SPORTS',
 'STYLE & BEAUTY',
 'TRAVEL',
 'U.S. NEWS',
 'WEDDINGS',
 'WEIRD NEWS',
 'WELLNESS',
 'WOMEN',
 'WORLD NEWS']

article_pred = model.predict(article_vec)
print(article_pred)
article_pred_class = np.argmax(article_pred, axis=1) # convert the One-Hot-Encoded vecotrs to a single vector
print(article_pred_class)

article_pred_new = article_pred[0]

article_score_class = []
for i in range(27):
    article_score_class.append([article_pred_new[i], i])
print(article_score_class)

article_score_class.sort()
article_score_class

count = 0
while(count < 5):
    print(classes_name[article_score_class[26 - count][1]])
    count += 1