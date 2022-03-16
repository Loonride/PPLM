from transformers import pipeline


def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with open(file) as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return contents

pos_input = read_dialog('./../data/positive.txt')
neg_input = read_dialog('./../data/negative.txt')
neg2pos_input = read_dialog('./../data/negative_to_positive.txt')
pos2neg_input = read_dialog('./../data/positive_to_negative.txt')

sentiment_analysis = pipeline("sentiment-analysis")

num = 0
correct = 0
for sentence in pos_input:
	num += 1
	if sentiment_analysis(sentence)[0]['label'] == 'POSITIVE':
		correct += 1
print("pos")
print((correct/num)*100)

neg_num = 0
neg_correct = 0
for sentence in neg_input:
	num += 1
	neg_num += 1
	if sentiment_analysis(sentence)[0]['label'] == 'NEGATIVE':
		correct += 1
		neg_correct += 1
print("neg")
print((neg_correct/neg_num)*100)

print("total")
print((correct/num)*100)

for sentence in neg2pos_input:
	print(sentiment_analysis(sentence))
for sentence in pos2neg_input:
	print(sentiment_analysis(sentence))