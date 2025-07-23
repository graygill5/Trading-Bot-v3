# Trading-Bot-v3
Attempting to make a trading bot for the US treasury dataset too maximize returns with a high risk using pandas and python

HOW TO USE: just run python progs/main.py

Summary:

-used original signals by using basic metrics to indicate sells and buys or to hold postition
-originally overfit like crazy becuase i was looking at the entire set and abusing look ahead bias 

-next used soem basic ML intergrations and got poor results
-decided to do an ensemble model to combine for the best results and was able to crack above random chances at around 55%

-issues and future: rioght now the bot is not wining alot of trades but when it wins its winning big in the future i might want to look at some lasso regression or things to tune down the data a little bit to avoid so much noise
-takeaways: learned lots of finance metrics that I've never heard of before and tought me the pwoer of compounding a investing, looking forward to workign more with finicial data