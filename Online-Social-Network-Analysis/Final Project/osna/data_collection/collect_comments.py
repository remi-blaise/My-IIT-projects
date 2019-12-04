#!/usr/bin/python3

import numpy as np
import pandas
import csv
import praw
from functools import reduce
import sys

PATH = './osna/data_collection/'
reddit = praw.Reddit(client_id='CDWqzxBMwHoFKw', client_secret="x95sILZingmAIt2FfLmccmp86qI", user_agent='MOZILLA')

def politifact_claims():
    def unique(l):
        used = set()
        return [ (i, x) for i, x in enumerate(l) if x not in used and (used.add(x) or True) ]

	# get politifact articles
    table = pandas.read_csv(PATH + 'politifact.csv')
	# get articles' headlines
    claims = table['claim']
	# Remove repeated healines
    claims = unique(claims)

    return claims

def collect_comments(claims, output_filename='comments_collected.csv'):
    claimWithSubmissionCount = 0
    submissionCount = 0

    resultingComments = {
        'claim_id': [],
        'author': [],
        'body': [],
        'submission_url': [],
        'submission_title': []
    }

    for claim_id, claim in claims:
        # search reddit comments with articles' headline
        allsub = reddit.subreddit("all")
        submissions = list(allsub.search(claim))

		# submissions are all threads for one article
        if submissions:
            claimWithSubmissionCount += 1
            submissionCount += len(submissions)

        for submission in submissions:
            sys.stdout.write('.')
            sys.stdout.flush()
            comments = list(submission.comments)

            for comment in comments:
			    # if comments are too numerous, some comments are in MoreComments object
                if isinstance(comment, praw.models.MoreComments):
                    comments.extend(comment.comments())
                    continue

				# put valuable information of comment in resultingComments
                resultingComments['claim_id'].append(claim_id)
                resultingComments['author'].append(comment.author)
                resultingComments['body'].append(comment.body)
                resultingComments['submission_url'].append(submission.url)
                resultingComments['submission_title'].append(submission.title)
        print()

    print('Ended')
	# put result in a csv file
    pandas.DataFrame(resultingComments).to_csv(PATH + output_filename)

    print("Submissions: " + str(submissionCount))
    print("Claims with submissions: " + str(claimWithSubmissionCount) + '/' + str(len(claims)))