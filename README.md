# cm-hate-speech-detection


Based on https://medium.com/@vladkens/how-to-still-scrape-millions-of-tweets-in-2023-using-twscrape-97f5d3881434

https://github.com/vladkens/twscrape


### Steps to scrape

1. Run the command `pip install twscrape` in a new environment
2. Create a new twitter account which you don't mind getting blocked, use a new email and password
3. Create a file called accounts.txt in the same folder as this file. This newly created file should have one line per account in the following format. You can add multiple accounts if you want to fasten the scraping process
twitter_username:twitter_password:email_username:email_password
4. Run the command `twscrape add_accounts ./accounts.txt username:password:email:email_password`
5. Run the command `twscrape login_accounts`
6. Open `scrapy.py`, and select the portion of the dataframe you want to fetch tweet for in the second line
7. Run `scrapy.py`
