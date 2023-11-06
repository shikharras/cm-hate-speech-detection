import asyncio
import time

import twscrape
import pandas as pd
import json
from tqdm import tqdm

def tweet_to_dict(tweet, query):
    tweet_dict = tweet.__dict__
    if tweet.user:
        user_dict = tweet.user.__dict__
        for key, value in user_dict.items():
            tweet_dict['user_' + key] = value
    tweet_dict.pop('user', None)
    tweet_dict["query"] = query
    return tweet_dict

async def worker(queue: asyncio.Queue, api: twscrape.API, df_list):
    while True:
        query = await queue.get()

        try:
            tweets = await twscrape.gather(api.search(query, limit=1))
            print(f"{query} - {len(tweets)} - {int(time.time())}")
            for tweet in tweets:
                # Convert tweet to dict and append to df_list
                df_list.append(tweet_to_dict(tweet, query))
        except Exception as e:
            print(f"Error on {query} - {type(e)}")
        finally:
            queue.task_done()

async def main():
    api = twscrape.API()
    await api.pool.add_account("user1", "pass1", "u1@example.com", "mail_pass1")
    await api.pool.login_all()

    queries = ["BSDK since:2023-01-01 :("]
    queue = asyncio.Queue()
    workers_count = 2
    df_list = []  # List to hold tweet dicts

    # Start workers
    workers = [asyncio.create_task(worker(queue, api, df_list)) for _ in range(workers_count)]
    for q in queries:
        queue.put_nowait(q)

    await queue.join()

    # Cancel workers after the queue is empty
    for worker_task in workers:
        worker_task.cancel()
    
    # Save the tweets to a CSV file
    df = pd.DataFrame(df_list)
    df.to_csv('tweets.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())