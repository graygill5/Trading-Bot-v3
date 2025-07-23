# Trading-Bot-v3
Attempting to make a trading bot for the US treasury dataset to maximize returns with a high risk using pandas and python

HOW TO USE: just run python progs/main.py

Summary:

-used original signals by using basic metrics to indicate sells and buys or to hold position
-originally overfit like crazy because i was looking at the entire set and abusing look-ahead bias 

-Next used some basic ML integrations and got poor results
-decided to do an ensemble model to combine for the best results ,and was able to crack above random chances at around 55%

-issues and future: right now the bot is not winning alot of trades, but when it wins its winning big. In the future, I might want to look at some lasso regression or things to tune down the data a little bit to avoid so much noise
-takeaways: learned lots of finance metrics that I've never heard of before and taught me the power of compounding a investing, looking forward to working more with financial data
