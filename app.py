import json
from analysis import classify_tweet
from flask import Flask, render_template, request, flash, Response
import tweepy as tw

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# take in keyword
# return number positive, number negative
@app.route("/tweets")
def tweets():
    keyword = request.args.get('keyword')

    num_positive, num_negative = scrape(keyword)

    return Response(json.dumps({'positives': num_positive, 'negatives': num_negative}), mimetype = 'application/json')

#take text
# return -1 if negative, 1 if positive
@app.route("/text")
def text():
    text = request.args.get('text')

    status = classify_tweet(text)

    return_status = 0
    if status == 'Positive':
        return_status =1
    else:
        return_status = -1

    return Response(json.dumps(return_status), mimetype = 'application/json')


def scrape(keyword):
    client = tw.Client("AAAAAAAAAAAAAAAAAAAAAOi1ZgEAAAAAKK5rqPAvqm9VFqjAxqqAXZ6xG%2BI%3Du08ykMFwrAF42bkhz0j1f2dujys4HeviR6DKz01GRSzKCUNtFJ")

    query = keyword

    tweets = client.search_recent_tweets(query=query,tweet_fields=['context_annotations', 'created_at'], max_results=10)

    found_tweets = []

    num_positive = 0
    num_negative = 0

    for tweet in tweets.data:
        if classify_tweet(tweet.text) == "Positive":
            num_positive = num_positive + 1
        else:
            num_negative = num_negative + 1

    return num_positive, num_negative