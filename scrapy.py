import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
import pandas as pd
import json
from tqdm import tqdm

df = pd.read_csv('cyberbullying_10k.csv', dtype={'tweet_id': 'string'})
df = df[:2500]
print(df.head())

ids = df['tweet_id'].values

print(len(ids))

tweet_op = []
tweet_full_json = []

kv = {
	"referrer": "tweet",  # tweet, profile
	"with_rux_injections": False,
	"includePromotedContent": False,
	"withCommunity": False,
	"withQuickPromoteEligibilityTweetFields": False,
	"withBirdwatchNotes": False,
	"withVoice": False,
	"withV2Timeline": False,
	"withDownvotePerspective": False,
	"withReactionsMetadata": False,
	"withReactionsPerspective": False,
	"withSuperFollowsTweetFields": False,
	"withSuperFollowsUserFields": False
}

async def main():
	api = API()

	file = open('tweet_texts.txt','w+')
	for idx,t_id in tqdm(enumerate(ids)):
		try:
			kv["focalTweetId"] = t_id
			rep = await api.tweet_details(int(t_id), kv)
			# rep is httpx.Response object
			# tweet_full_json.append(rep)
			tweet_op.append(rep.rawContent)
			file.write(rep.rawContent+"<<EOT>>\n")
		except:
			print("Error for ID: ", t_id)
			tweet_op.append("")
			file.write("<<EOT>>\n")
	file.close()

	df['tweet_text'] = tweet_op
	df.to_csv('filled_10k.csv')



if __name__ == "__main__":
    asyncio.run(main())