import csv
import time
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# downloading needed nltk packages
nltk.download("stopwords")
# Getting the stop words corpus
stopwords_corpus = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()
# Reading csv file
bumble = pd.read_csv("bumble_google_play_reviews - bumble_google_play_reviews copy.csv")

# Getting rid of nan values in the content section of csv file
bumble = bumble[bumble['content'].notna()]

# determining whether we skip lowercase or not
skip = input("Do you want to skip Lower-Casing step?")
if skip == "Yes" or skip == "yes" or skip == "YES" or skip == "y" or skip == "Y":
    ignore = True
else:
    ignore = False


def classifier(ignore):
    start4 = time.time()
    # count var will be used as the training dictionary key to preserve uniqueness among entries
    count = 0
    # Initializing dictionary's
    training_samples = {}
    test_samples = {}
    if ignore:
        print("Ignored pre-processing step: LOWER CASING")
    else:
        print("Ignored pre-processing step: NONE")
    print("Completing pre-processing steps")
    # timer to analyze dataset processing time
    start = time.time()
    # Looping through csv file
    for row in bumble.index:
        # Putting the data in the content (review comments) and score (1-5 star score) columns into variables
        content = bumble.loc[row, "content"]
        score = bumble.loc[row, "score"]
        # allocating 80% of the dataset for classifier training
        training_bounds = int(len(bumble) * .8)
        # Checking if we are going to skip lower casing pre-processing step
        if ignore:
            # Tokenizing the current review we are looking at
            content1 = word_tokenize(content)
            # Removing stop words from the tokenized review
            content = [w for w in content1 if w not in stopwords_corpus]
            # Lemmatizing remaining words and adding them to a list
            content2 = [ps.stem(w, to_lowercase=False) for w in content]
            if count < training_bounds:
                # Adding the tokenized sentence and respective score to training dictionary with count as the key
                training_samples[count] = [content2, score]
                count += 1
            else:
                # allocating remaining 20% of the dataset for classifier testing
                test_samples[count] = [content2, score]
                count += 1
        # Repeating the same steps but this time no pre-processing steps are skipped
        else:
            content.lower()
            content1 = word_tokenize(content)
            content = [w for w in content1 if w not in stopwords_corpus]
            content2 = [ps.stem(w) for w in content]
            if count < training_bounds:
                training_samples[count] = [content2, score]
                count += 1
            else:
                test_samples[count] = [content2, score]
                count += 1
    end = time.time()
    total_time = end - start
    print("Pre-processing completed in: " + str(total_time))
    #
    word_counts1 = {}
    word_counts2 = {}
    word_counts3 = {}
    word_counts4 = {}
    word_counts5 = {}
    label1 = 0
    label2 = 0
    label3 = 0
    label4 = 0
    label5 = 0
    print("Training classifier…")
    start1 = time.time()
    # Looping through items in training dictionary
    for element in training_samples.items():
        # Gathering the word counts for each classifier based on score
        if element[1][1] == 1:
            # Keeping track of amount of reviews that are classified as 1 star
            label1 += 1
            # Looping through each word in tokenized sentence
            for i in element[1][0]:
                if i not in word_counts1:
                    # added +1 smoothing to all new words
                    word_counts1[i] = 2
                else:
                    # incrementing frequency if word already exists in the dictionary
                    word_counts1.update({i: word_counts1[i] + 1})
        elif element[1][1] == 2:
            label2 = label2 + 1
            for i in element[1][0]:
                if i not in word_counts2:
                    word_counts2[i] = 2
                else:
                    word_counts2.update({i: word_counts2[i] + 1})
        elif element[1][1] == 3:
            label3 = label3 + 1
            for i in element[1][0]:
                if i not in word_counts3:
                    word_counts3[i] = 2
                else:
                    word_counts3.update({i: word_counts3[i] + 1})
        elif element[1][1] == 4:
            label4 = label4 + 1
            for i in element[1][0]:
                if i not in word_counts4:
                    word_counts4[i] = 2
                else:
                    word_counts4.update({i: word_counts4[i] + 1})
        else:
            label5 = label5 + 1
            for i in element[1][0]:
                if i not in word_counts5:
                    word_counts5[i] = 2
                else:
                    word_counts5.update({i: word_counts5[i] + 1})
    # Getting the total size across all labels
    size_of_labels = label1 + label2 + label3 + label4 + label5
    # calculating the probability of each label
    probability_label1 = label1/size_of_labels
    probability_label2 = label2/size_of_labels
    probability_label3 = label3/size_of_labels
    probability_label4 = label4/size_of_labels
    probability_label5 = label5/size_of_labels
    # Getting the total amount of words per label
    words_in_label1 = sum(word_counts1.values())
    words_in_label2 = sum(word_counts2.values())
    words_in_label3 = sum(word_counts3.values())
    words_in_label4 = sum(word_counts4.values())
    words_in_label5 = sum(word_counts5.values())
    # looping through each label's dictionary
    for word in word_counts1.items():
        # getting the current word freq / total words in that label to get the words probability in that label
        prob_word_in1 = word[1]/words_in_label1
        # Updating the current item in the dictionary to include the probability of the word
        word_counts1.update({word[0]: [word[1], prob_word_in1]})
    # Repeating the same process for other labels
    for word in word_counts2.items():
        prob_word_in2 = word[1]/words_in_label2
        word_counts2.update({word[0]: [word[1], prob_word_in2]})
    for word in word_counts3.items():
        prob_word_in3 = word[1]/words_in_label3
        word_counts3.update({word[0]: [word[1], prob_word_in3]})
    for word in word_counts4.items():
        prob_word_in4 = word[1]/words_in_label4
        word_counts4.update({word[0]: [word[1], prob_word_in4]})
    for word in word_counts5.items():
        prob_word_in5 = word[1]/words_in_label5
        word_counts5.update({word[0]: [word[1], prob_word_in5]})
    end1 = time.time()
    total_time1 = end1 - start1
    print("Training completed in: " + str(total_time1))
    # Now we are going to test
    print("Testing classifier… \n")
    start2 = time.time()
    # Looping through all test data
    for element in test_samples.items():
        # Initializing the probability of each sentence with the probability of each label
        prob_of_sen_w_label1 = probability_label1
        prob_of_sen_w_label2 = probability_label2
        prob_of_sen_w_label3 = probability_label3
        prob_of_sen_w_label4 = probability_label4
        prob_of_sen_w_label5 = probability_label5
        # looping through the tokenized sentence
        for i in element[1][0]:
            # if the word is in our word_count dictionary
            if i in word_counts1:
                # multiply the probability of that word with the probability of the sentence being label 1
                prob_of_sen_w_label1 = prob_of_sen_w_label1 * word_counts1[i][1]
            else:
                # Ignore the word (no change to the probability of the sentence)
                prob_of_sen_w_label1 = prob_of_sen_w_label1 * 1
            # Repeat the same process to calculate the probability of the sentence for each label
            if i in word_counts2:
                prob_of_sen_w_label2 = prob_of_sen_w_label2 * word_counts2[i][1]
            else:
                prob_of_sen_w_label2 = prob_of_sen_w_label2 * 1

            if i in word_counts3:
                prob_of_sen_w_label3 = prob_of_sen_w_label3 * word_counts3[i][1]
            else:
                prob_of_sen_w_label3 = prob_of_sen_w_label3 * 1

            if i in word_counts4:
                prob_of_sen_w_label4 = prob_of_sen_w_label4 * word_counts4[i][1]
            else:
                prob_of_sen_w_label4 = prob_of_sen_w_label4 * 1

            if i in word_counts5:
                prob_of_sen_w_label5 = prob_of_sen_w_label5 * word_counts5[i][1]
            else:
                prob_of_sen_w_label5 = prob_of_sen_w_label5 * 1
        # Add all probabilities to a list to be able to get the index of the highest probability
        predicted_label1 = [prob_of_sen_w_label1, prob_of_sen_w_label2, prob_of_sen_w_label3,
                            prob_of_sen_w_label4, prob_of_sen_w_label5]
        # set predicted label as highest probability
        predicted_label = predicted_label1.index(max(predicted_label1))

        # Append the predicted label to the dictionary element
        if predicted_label == 0:
            predicted_label = 1
            test_samples.update({element[0]: [element[1][0], element[1][1], predicted_label]})
            predicted_label = 0
        if predicted_label == 1:
            predicted_label = 2
            test_samples.update({element[0]: [element[1][0], element[1][1], predicted_label]})
            predicted_label = 1
        if predicted_label == 2:
            predicted_label = 3
            test_samples.update({element[0]: [element[1][0], element[1][1], predicted_label]})
            predicted_label = 2
        if predicted_label == 3:
            predicted_label = 4
            test_samples.update({element[0]: [element[1][0], element[1][1], predicted_label]})
            predicted_label = 3
        if predicted_label == 4:
            predicted_label = 5
            test_samples.update({element[0]: [element[1][0], element[1][1], predicted_label]})
            predicted_label = 4

    current_label = 1
    # calculating the true positive, true negative, false positive, and false negative values per classifier
    for i in range(0, 5):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for item in test_samples.items():
            # checking if actual score is equal to current_label
            if item[1][1] == current_label:
                # increment true positive when predicted score = actual score
                if item[1][1] == item[1][2]:
                    true_positive += 1
                    # incrementing false negative when actual score is not equal to predicted score
                if item[1][1] != item[1][2]:
                    false_negative += 1
            else:
                # incrementing false positive when actual score is not current_label
                # but the predicted score is equal to the current label
                if item[1][2] == current_label:
                    false_positive += 1
                else:
                    # incrementing true negative when acutal score is not current label
                    # and predicted score is not current label
                    true_negative += 1
        print("Test results / metrics:")
        print("Classfier " + str(current_label))
        print("Number of true positives: " + str(true_positive))
        print("Number of true negative: " + str(true_negative))
        print("Number of false positives: " + str(false_positive))
        print("Number of false negative: " + str(false_negative))
        # calculating recall, specificity, precision, negative predictive value, accuracy, and F-score
        recall = true_positive/(true_positive+false_negative)
        specificity = true_negative/(true_negative+false_positive)
        precision = true_positive/(true_positive+false_positive)
        npv = true_negative/(true_negative+false_negative)
        accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
        f_score = (2*precision*recall)/(precision+recall)
        print("Sensitivity (recall): " + str(recall))
        print("Specificity: " + str(specificity))
        print("Precision: " + str(precision))
        print("Negative predictive value: " + str(npv))
        print("Accuracy: " + str(accuracy))
        print("F-score: " + str(f_score) + "\n")
        current_label += 1
    end2 = time.time()
    end3 = time.time()
    total_time2 = end2 - start2
    total_completion_time = end3 - start4
    print("Testing completed in: " + str(total_time2))
    print("Total time: " + str(total_completion_time))
    answer = "Y"
    while answer == "Y" or answer == "y" or answer == "yes" or answer == "YES" or answer == "Yes":
        # Asking for user to input a sentence they want classified
        sentence = input("Enter your sentence: ")
        print("Sentence S: \n" + sentence)
        # Completing necessary pre-processing steps
        if ignore:
            words1 = word_tokenize(sentence)
            words2 = [w for w in words1 if w not in stopwords_corpus]
            words3 = [ps.stem(w, to_lowercase=False) for w in words2]
        else:
            sentence = sentence.lower()
            words1 = word_tokenize(sentence)
            words2 = [w for w in words1 if w not in stopwords_corpus]
            words3 = [ps.stem(w) for w in words2]
        # re-initializing probabilities of each classifier
        prob_of_sen_w_label1 = probability_label1
        prob_of_sen_w_label2 = probability_label2
        prob_of_sen_w_label3 = probability_label3
        prob_of_sen_w_label4 = probability_label4
        prob_of_sen_w_label5 = probability_label5

        # calculating the probability of each sentence for each label using same method as did in training
        for word in words3:
            if word in word_counts1:
                prob_of_sen_w_label1 = prob_of_sen_w_label1 * word_counts1[word][1]
            else:
                prob_of_sen_w_label1 = prob_of_sen_w_label1 * 1

            if word in word_counts2:
                prob_of_sen_w_label2 = prob_of_sen_w_label2 * word_counts2[word][1]
            else:
                prob_of_sen_w_label2 = prob_of_sen_w_label2 * 1

            if word in word_counts3:
                prob_of_sen_w_label3 = prob_of_sen_w_label3 * word_counts3[word][1]
            else:
                prob_of_sen_w_label3 = prob_of_sen_w_label3 * 1

            if word in word_counts4:
                prob_of_sen_w_label4 = prob_of_sen_w_label4 * word_counts4[word][1]
            else:
                prob_of_sen_w_label4 = prob_of_sen_w_label4 * 1

            if word in word_counts5:
                prob_of_sen_w_label5 = prob_of_sen_w_label5 * word_counts5[word][1]
            else:
                prob_of_sen_w_label5 = prob_of_sen_w_label5 * 1
        predicted_label1 = [prob_of_sen_w_label1, prob_of_sen_w_label2, prob_of_sen_w_label3,
                            prob_of_sen_w_label4, prob_of_sen_w_label5]
        predicted_label = predicted_label1.index(max(predicted_label1))
        predicted_label += 1
        if predicted_label == 1:
            print("Classification/Rating: " + "⭐ ")
        if predicted_label == 2:
            print("Classification/Rating: " + "⭐ ⭐ ")
        if predicted_label == 3:
            print("Classification/Rating: " + "⭐ ⭐ ⭐ ")
        if predicted_label == 4:
            print("Classification/Rating: " + "⭐ ⭐ ⭐ ⭐ ")
        if predicted_label == 5:
            print("Classification/Rating: " + "⭐ ⭐ ⭐ ⭐ ⭐ ")
        # printing the probabilities of the sentence for each label
        print("P(1 star | " + sentence + "): " + str(prob_of_sen_w_label1))
        print("P(2 stars | " + sentence + "): " + str(prob_of_sen_w_label2))
        print("P(3 stars | " + sentence + "): " + str(prob_of_sen_w_label3))
        print("P(4 stars | " + sentence + "): " + str(prob_of_sen_w_label4))
        print("P(5 stars | " + sentence + "): " + str(prob_of_sen_w_label5))
        check_score = input("Was this the expected score? [Y/N] ")
        if check_score == "no" or check_score == "NO" or check_score == "N" or check_score == "n" or check_score == "No":
            predicted_label = input("Enter a score: [1-5] ")
            error = True
            while error:
                try:
                    predicted_label = int(predicted_label)
                    if 1 <= predicted_label < 6:
                        error = False
                except:
                    print("Sorry incorrect input please try again!")
                    predicted_label = input("Enter a score: [1-5]")
                    error = True
        with open('bumble_google_play_reviews copy.csv', 'a') as f_object:
            field_names = ['reviewId', 'userName', 'userImage', 'content', 'score', 'thumbsUpCount',
                           'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt']
            append_dict = {"reviewId": "", "userName": "", "userImage": "", "content": sentence,
                           "score": predicted_label,
                           "thumbsUpCount": "", "reviewCreatedVersion": "", "at": "", "replyContent": "",
                           "repliedAt": ""}
            dict_object = csv.DictWriter(f_object, fieldnames=field_names)
            dict_object.writerow(append_dict)
            f_object.close()
        answer = input(" Do you want to enter another sentence? [Y/N]? ")


classifier(ignore)
