import pandas as pd

sample_jokes = ['''
A man visits the doctor. The doctor says, "I have bad news for you. You have cancer and Alzheimer's disease".
The man replies, "Well, thank God I don't have cancer!"
''',
'''
This couple had an excellent relationship going until one day he came home from work to find his girlfriend packing. He asked her why she was leaving him and she told him that she had heard awful things about him.

"What could they possibly have said to make you move out?"
"They told me that you were a pedophile."
He replied, "That's an awfully big word for a ten year old."
''',
'''
Q. What's 200 feet long and has 4 teeth?<br />
A. The front row at a Willie Nelson concert.
''']

#sample_ratings = pd.read_csv('data/ratings.dat', sep="\t")

#sample_submission = pd.read_csv('data/sample_submission.csv')

#sample_submission.groupby(['user_id', 'joke_id']).sum()
#sample_ratings.groupby(['user_id', 'joke_id']).sum()

sample_ratings = pd.read_csv('data/dont_use.csv')

sample_ratings['rating'] = sample_ratings['rating'].apply(lambda x: (x<0)*(-10) or (x>0)*10)

sample_ratings.to_csv('data/worst_pred.csv')
