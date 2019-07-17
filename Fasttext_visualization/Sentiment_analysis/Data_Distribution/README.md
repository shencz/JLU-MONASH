# Data_Distribution

## statistical graph

mean.jpg include the mean distributions of correct and incorrect data predictions
![image](https://github.com/shencz/JLU-MONASH/blob/Shencz/Fasttext_visualization/Sentiment_analysis/Data_Distribution/mean.jpg)

var.jpe include the var distributions of correct and incorrect data predictions
![image](https://github.com/shencz/JLU-MONASH/blob/Shencz/Fasttext_visualization/Sentiment_analysis/Data_Distribution/var.jpg)

std.jpe include the std distributions of correct and incorrect data predictions
![image](https://github.com/shencz/JLU-MONASH/blob/Shencz/Fasttext_visualization/Sentiment_analysis/Data_Distribution/std.jpg)

sigmoid_predict_n.jpg includes the sigmoid distributions of simples predicted negative. Where sigmoid_n_n means predict negative, sigmoid_p_n means predict positive.

![image](https://github.com/shencz/JLU-MONASH/blob/Shencz/Fasttext_visualization/Sentiment_analysis/Data_Distribution/sigmoid_predict_n.jpg)

sigmoid_predict_p.jpg includes the sigmoid distributions of simples predicted positive. Where sigmoid_p_p means predict positive, sigmoid_n_p means predict negative.

![image](https://github.com/shencz/JLU-MONASH/blob/Shencz/Fasttext_visualization/Sentiment_analysis/Data_Distribution/sigmoid_predict_p.jpg)

## model comparison


- model acc

| model   |fasttext |fasttext_bigram|fasttext_trgram|   CNN   |   HAN   |
|:-----   | :-----: | :-----------: | :-----------: | :-----: | :-----: |
| val_acc |  90.28  |     91.83     |     91.60     |  89.32  |   89.7  |

- error statistics

4333 test sample in total

for fasttext_bigram, there are 354 wrong predict samples.

Of the data that the fasttext_bigram model predicted incorrectly, 136 were correctly predicted by the CNN model, accounting for 38.4%

Of the data that the fasttext_bigram model predicted incorrectly, 110 were correctly predicted by the HAN model, accounting for 31.1%

The forecast details are in the .txt file.







