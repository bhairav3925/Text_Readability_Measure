# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:06:23 2020

@author: Bhairav.Jain
"""

import nltk
import numpy as np
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

contractions = {"it’s":"it is","ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he shall", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how does", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as / so is", "that'd": "that had", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what shall have / what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when has", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "im": " i am", "isnt": "is not", "dont": "do not", "havent": "have not","its":"it is","won't":"will not","i'd":"i would","it’s":"it is"}
contractions = dict((k.lower(), v. lower()) for k,v in contractions.items())

# Counting Number of Syllables
def _countSyllablesEN(theText):

    cleanText = ""
    for ch in theText.lower():
        if ch in "abcdefghijklmnopqrstuvwxyz'’":
            cleanText += ch
        else:
            cleanText += " "

    asVow    = "aeiouy'’"
    dExep    = ("ei","ie","ua","ia","eo")
    theWords = cleanText.lower().split()
    allSylls = 0
    for inWord in theWords:
        nChar  = len(inWord)
        nSyll  = 0
        wasVow = False
        wasY   = False
        if nChar == 0:
            continue
        if inWord[0] in asVow:
            nSyll += 1
            wasVow = True
            wasY   = inWord[0] == "y"
        for c in range(1,nChar):
            isVow  = False
            if inWord[c] in asVow:
                nSyll += 1
                isVow = True
            if isVow and wasVow:
                nSyll -= 1
            if isVow and wasY:
                nSyll -= 1
            if inWord[c:c+2] in dExep:
                nSyll += 1
            wasVow = isVow
            wasY   = inWord[c] == "y"
        if inWord.endswith(("e")):
            nSyll -= 1
        if inWord.endswith(("le","ea","io")):
            nSyll += 1
        if nSyll < 1:
            nSyll = 1
        allSylls += nSyll

    return allSylls,len(theWords) 

def _structure_1(text_file):
    # Reading Text file
    with open(text_file,"r",encoding="utf-8") as f:
        data = f.read().split("\n")
    
    data = list(map(lambda x:re.sub("(\.\s){2,}",". ",x),data))
    
    # lists used to store the structural information of the text
    sentence_length = []
    syllable = []
    totalWords = []
    noCharacters = []
    avgSentenceLength = []
    avgSyllablesWords = []

    # Following steps are implemented in the loop below
    # Counting the syllables in each word, total words in the text,
    # total number of sentence in text, average sentence length in text and
    # average number of syllables in a word in the text
    # Counting total number of characters in each words in the text
    for i in range(len(data)):
        
        _temp_ = re.split(r"\.\s?|\?\”|\?",data[i])
        
        _temp_ = list(map(lambda x : re.sub("\”|\“|\""," ",x),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\""," ",x).strip(),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip(),_temp_))
        
        _temp_ = list(filter(lambda x : len(x)!=2,_temp_))
        
        _temp_ = list(map(lambda x : re.sub('!|,|’s'," ",x),_temp_))
        
        _temp_ = list(filter(lambda x: x!=' ',_temp_))
        
        _temp_ = list(filter(lambda x: x!='',_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip(),_temp_))
        
        _temp_
        
        noCharacters.append(sum([len(x) for x in _temp_]))
        
        _tempSyllableWords = 0
        _temp_totalWords_ = 0
        for x in range(len(_temp_)):
            _block_ = _countSyllablesEN(_temp_[x])
            _temp_totalWords_ = _temp_totalWords_ + _block_[1]
            _tempSyllableWords = _tempSyllableWords + _block_[0]
        
        syllable.append(_tempSyllableWords)
        totalWords.append(_temp_totalWords_)
        sentence_length.append(len(_temp_))
        avgSentenceLength.append(int(_temp_totalWords_ / len(_temp_)))
        avgSyllablesWords.append(round(_tempSyllableWords / _temp_totalWords_,2))
    
    # Lists to store the strings and numbers used to indicate the complexity level of the text
    readablitityMeasure_FleschReading = []
    gIndex_1 = []
    gIndex_2 = []
    gIndex = []
    readablitityMeasure_FleschKincaid = []
    for i in range(len(data)):
        
       asl = 1.015 * avgSentenceLength[i]
        
       asw = 84.6 * avgSyllablesWords[i]
       
       fEQ = 0.39*avgSentenceLength[i] 
       sEQ = 11.8*avgSyllablesWords[i]
       
       _tempMeasure = round(206.835 - (asl) - (asw),2)
       
       _tempMeasure_1 = round((fEQ + sEQ) - 15.59,2) 
       
       characterWord = 10 * (noCharacters[i] / totalWords[i])
       
       wordSentence = 300 * (sentence_length[i] / totalWords[i])
       
       gIndex.append(round(((89 - characterWord) + wordSentence),2))
       
       gIndex_1.append(_tempMeasure)
       if _tempMeasure > 0.0 and _tempMeasure <=30.0:
           readablitityMeasure_FleschReading.append("Between 0.0 and 30.0. Very difficult to read. Best understood by university graduates.")
       elif _tempMeasure > 30.0 and _tempMeasure <=50.0:
           readablitityMeasure_FleschReading.append("Between 30.0 and 50.0. Difficult to read.")
       elif _tempMeasure > 50.0 and _tempMeasure <= 60.0:
           readablitityMeasure_FleschReading.append("Between 50.0 and 60.0 Fairly difficult to read.")
       elif _tempMeasure > 60.0 and _tempMeasure <= 70.0:
           readablitityMeasure_FleschReading.append("Between 60.0 and 70.0 Plain English.")
       elif _tempMeasure > 70.0 and _tempMeasure <= 80.0:
           readablitityMeasure_FleschReading.append("Between 70.0 and 80.0 Fairly easy to read.")
       elif _tempMeasure > 80.0 and _tempMeasure <= 90.0:
           readablitityMeasure_FleschReading.append("Between 80.0 and 90.0 Easy to read. Conversational English for consumers.")
       elif _tempMeasure > 90.0 and _tempMeasure <= 100.0:
           readablitityMeasure_FleschReading.append("Between 90.0 and 100.0 Very easy to read. Easily understood.")
       
       gIndex_2.append(_tempMeasure_1) 
       if _tempMeasure_1 <= 5.0 :
            readablitityMeasure_FleschKincaid.append("Less than 5.0 for Pre-School")
       elif _tempMeasure_1 > 5.0 and _tempMeasure_1 <= 11.0:
            readablitityMeasure_FleschKincaid.append("Between 5.0 and 11.0 for Elementary / Primary School")
       elif _tempMeasure_1 > 11.0 and _tempMeasure_1 <= 14.0:
            readablitityMeasure_FleschKincaid.append("Between 11.0 and 14.0 for Junior / Secondary High School")
       elif _tempMeasure_1 > 14.0 and _tempMeasure_1 <= 18.0:
            readablitityMeasure_FleschKincaid.append("Between 14.0 and 18.0 for High School or 10th and 12th College")
       elif _tempMeasure_1 > 18.0 and _tempMeasure_1 <=22.0:
            readablitityMeasure_FleschKincaid.append("Between 18.0 and 22.0 for Undergraduate ")
       elif _tempMeasure_1 > 22.0:
            readablitityMeasure_FleschKincaid.append("Greater 22.0. Need to set threshold depends on Domain")
    
    return gIndex,readablitityMeasure_FleschReading,readablitityMeasure_FleschKincaid

# This function extracts the Semantic Information from the text
# We calculate the  Number of Noun Phrase, Verb Phrase, Preposition Phrase, Interjection Phrase
# We use formulas for calculating Chunk Index, Chunk Type & Readability Index
def _semantic_2(gIndex,text_file):
    
    with open(text_file,"r",encoding="utf-8") as f:
        data = f.read().split("\n")
      
    data = list(map(lambda x:re.sub("(\.\s){2,}",". ",x),data))
    _temp_sentence_pos = {}
    
    for i in range(len(data)):
        sample = data[i]
        
        sample = re.sub("\”|\“|\""," ",sample.lower()).strip()
        
        _temp_ = []
        for j in sample.split():
            j = j.replace("’","'")
            if contractions.get(j.lower()):
                _temp_.append(contractions.get(j.lower()))
            else:
                _temp_.append(j)
        
        text_1 = ' '.join(_temp_).lower()
        
        _temp_ = re.split(r"\.\s?|\?\”|\?",text_1.lower())
        
        _temp_ = list(map(lambda x : re.sub("\”|\“|\""," ",x),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\""," ",x).strip(),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\'s"," ",x).strip(),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip(),_temp_))
        
        _temp_ = list(filter(lambda x : len(x)!=2,_temp_))
        
        _temp_ = list(map(lambda x : re.sub('!|,|’s'," ",x),_temp_))
        
        _temp_ = list(filter(lambda x: x!=' ',_temp_))
        
        _temp_ = list(filter(lambda x: x!='',_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip().lower(),_temp_))
        
        _temp_
        
        gulpease_Index = gIndex[i]
        
        _temp_DS = data[i]
        
        np_count = 0
        np_cp = nltk.RegexpParser(r"""
          NP1: {<DT|PP\$>+<JJ.*>*<NN.*>+}    # noun phrase chunks 1 (Determiner or Possessive followed by Adjective with zero or more followed by Noun
          NP2: {<PRP.*>+<JJ>*}               # noun phrase chunks 2 (Personal/Possessive Pronoun followed by Adjective)
          NP3: {<NNS?>+}                     # noun phrase chunks 3 (1 or more Noun,plural)
          NP4: {<IN>+<NN>+}                  # noun phrase chunks 4 (Preposition followed by Noun)
          """)
        
        for i in range(len(_temp_)):
            tagged = nltk.pos_tag(_temp_[i].split())
            chunked = np_cp.parse(tagged)
            
            print("###################################################################")
            print("Original Sentence : ",_temp_[i],"\n")
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'NP1' or t.label() == 'NP2' or t.label() == 'NP3' or t.label() == 'NP4' or t.label() == 'PP'):
                print("-------------------------------------------------")
                print(subtree)
                print("-------------------------------------------------") 
                np_count = np_count + 1
            print("###################################################################\n")
            
        
        vp_cp = nltk.RegexpParser(r"""
          VP1: {<VB.*>+<RB|TO>*<VB>*}    # Verb phrase chunks 1 (Verb foolowed by Adverb with Zero or More followed Verb)
          VP2: {<TO|MD|NN>+<VB.*>+}      # Verb phrase chunks 2 
          """)
        vp_count = 0
        for i in range(len(_temp_)):
            tagged = nltk.pos_tag(_temp_[i].split())
            chunked = vp_cp.parse(tagged)
            
            print("###################################################################")
            print("Original Sentence : ",_temp_[i],"\n")
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'VP1' or t.label() == 'VP2'):
                print("-------------------------------------------------")
                print(subtree)
                print("-------------------------------------------------")  
                vp_count = vp_count + 1
            print("###################################################################\n")
        
        pp_cp = nltk.RegexpParser(r"""
                               PP: {<IN>}                 # prepositional phrase chunks
                               """)
        
        pp_count = 0
        for i in range(len(_temp_)):
            tagged = nltk.pos_tag(_temp_[i].split())
            chunked = pp_cp.parse(tagged)
            
            print("###################################################################")
            print("Original Sentence : ",_temp_[i],"\n")
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'PP'):
                print("-------------------------------------------------")
                print(subtree)
                print("-------------------------------------------------")  
                pp_count = pp_count + 1
            print("###################################################################\n")
    
    
     
        chunkIndex = 0
        if (np_count + vp_count + pp_count) == 1:
            chunkIndex = 100
        else:
            numChunk = (np_count + vp_count + pp_count) 
            numChunk = int(np.ceil(numChunk / len(_temp_)))
            
            # numChunk = (numChunk / len(_temp_))
            numSente = 1 * len(_temp_)
            
            numChunk = numChunk - numSente
            
            chunkIndex = round((100 / numChunk),2)
        
        # Noun Phrase 0.2048
        # Prepositional 0.1000
        # Verbal 0.2459
        count_Phrase = {"NP":np_count, "VP":vp_count, "PP":pp_count}
        weight_Phrase = {"NP":0.2048,"VP":0.1000,"PP":0.2459}
        _temp_Chunck_Index = 0
        _temp_Cal = [] 
        if (np_count + vp_count + pp_count) == 1:
            for i in weight_Phrase:
                _temp_Cal.append(weight_Phrase.get(i) * count_Phrase.get(i))
            
            _temp_Cal = sum(_temp_Cal)
            
            _temp_Chunck_Index = round(_temp_Cal / 0.2468,2)
            
            _temp_Chunck_Index = 100 * _temp_Chunck_Index
        else:
            for i in weight_Phrase:
                _temp_Cal.append(weight_Phrase.get(i) * count_Phrase.get(i))
            
            _temp_Cal = sum(_temp_Cal)
            
            _temp_Chunck_Index = round(_temp_Cal / sum(list(count_Phrase.values())),2)
             
            _temp_Chunck_Index = 544 * _temp_Chunck_Index
    
    
        final_Readability_Score = 0
        
        _temp_Value = 0.75
        
        final_Readability_Score = (_temp_Value * gulpease_Index) + (_temp_Value * chunkIndex) + (_temp_Value * _temp_Chunck_Index)
    
        _temp_sentence_pos[_temp_DS] = round(final_Readability_Score,2)        
    
    return _temp_sentence_pos

def _relative_POS_Percentage(text_file):
    with open(text_file,"r",encoding="utf-8") as f:
        data = f.read().split("\n")
    
    contractions = {"it’s":"it is","ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he shall", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how does", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as / so is", "that'd": "that had", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what shall have / what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when has", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "im": " i am", "isnt": "is not", "dont": "do not", "havent": "have not","its":"it is","won't":"will not","i'd":"i would","it’s":"it is"}
    contractions = dict((k.lower(), v. lower()) for k,v in contractions.items())
    
      
    data = list(map(lambda x:re.sub("(\.\s){2,}",". ",x),data))
    _final_df_per = pd.DataFrame()
    for i in range(len(data)):
        sample = data[i]
        
        sample = re.sub("\”|\“|\""," ",sample.lower()).strip()
        
        _temp_ = []
        for j in sample.split():
            j = j.replace("’","'")
            if contractions.get(j.lower()):
                _temp_.append(contractions.get(j.lower()))
            else:
                _temp_.append(j)
        
        text_1 = ' '.join(_temp_).lower()
        
        _temp_ = re.split(r"\.\s?|\?\”|\?",text_1.lower())
        
        _temp_ = list(map(lambda x : re.sub("\”|\“|\""," ",x),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\""," ",x).strip(),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\'s"," ",x).strip(),_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip(),_temp_))
        
        _temp_ = list(filter(lambda x : len(x)!=2,_temp_))
        
        _temp_ = list(map(lambda x : re.sub('!|,|’s'," ",x),_temp_))
        
        _temp_ = list(filter(lambda x: x!=' ',_temp_))
        
        _temp_ = list(filter(lambda x: x!='',_temp_))
        
        _temp_ = list(map(lambda x:re.sub("\s{2,}"," ",x).strip().lower(),_temp_))
        
        _temp_
        
        _temp_pos_sentence = []
        for i in _temp_:
            _temp_pos_sentence.append(nltk.pos_tag(i.split()))
        
        _temp_pos = {}
        for j in _temp_pos_sentence:
            for k in j:
                if _temp_pos.get(k[1]):
                    _temp_pos[k[1]] = _temp_pos[k[1]] + 1
                else:
                    _temp_pos[k[1]] = 1
        
        _total_occur = sum(list(_temp_pos.values()))
        _final_pos_percentage_ = {}
        for i in _temp_pos:
            _final_pos_percentage_[i] = round(_temp_pos[i] / _total_occur,3) * 100
        
        _temp_df_ = pd.DataFrame.from_dict(dict(sorted(_final_pos_percentage_.items(),key=lambda x:x[1],reverse=True)[:7]),orient="index")
    
        _temp_df_.columns = [["Percentage"]]
        
        _temp_df_ = _temp_df_.T.reset_index().drop(labels='level_0',axis=1)
        
        _temp_df_['Sentence'] = sample
        
        if _final_df_per.empty:
            _final_df_per = _temp_df_
        else:
            _final_df_per = pd.concat([_final_df_per,_temp_df_])
        
    return _final_df_per

# Provide full path of Text File
text_file =  "Testing_1.txt"   

# Pass Text File Name 
# The Function "_structure_1" will return Structral Based score,
# Measure of FleschReading and FleschKincaid scores,
# and some Other structure score

print("------1st Phase of Program(Calculating Structure INformation from Text File)------\n\t")
gIndex,FleschReading,FleschKincaid = _structure_1(text_file)
print("\tFleschReading Score : \n\t")
for i in FleschReading:
    print(i)
print("\n------End------\n\t")

# Below function will Calculate the Chunck Index, Chunk Type Index and Readability Index
print("------2nd Phase of Program (Calculating Semantics Information from Text File)------")
_text_readability_ = _semantic_2(gIndex,text_file)
print("\n\t-----Semantics Score Generated-----\n\t")
for i in list(_text_readability_.values()):
    print(i)
print("\n------End------\n\t")

# Below Function will return Relative Percentage of Part-of-Speech tags in Text
per_pos = _relative_POS_Percentage(text_file)
per_pos = per_pos.reset_index().drop(labels='index',axis=1)
per_pos = per_pos.fillna(0)
print("------3rd Phase of Program (Calculating Percentage of POS Tag from from Text File")
print(per_pos)
print("\n------End------\n\t")