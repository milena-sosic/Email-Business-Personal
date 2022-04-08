#!/bin/bash
#
# Description: The following script executes the pipeline.py file to trigger
#              a training of the email classification model exploration.
#
#######################################
#
# Interface
#  Modify the below variables prior the running the script.
# 
#  TRAIN_ID: Train dataset identifier.
#   Format: String ['columbia', 'berkeley']
#   Example: TRAIN_ID='columbia'
#   Default: 'columbia'
#
#  TEST_ID: Test dataset identifier.
#   Format: String ['columbia', 'berkeley']
#   Example: TEST_ID='columbia'
#   Default: 'columbia'
#
#  EXPERIMENT: Experiment identifier.
#   Format: String ['E', 'ED', 'EQ', 'EQD', 'B']
#   Example: EXPERIMENT='E'
#   Default: 'E'
#
#  PREPROCESS: Flag whether to preprocess the data
#   Format: Boolean (true || false)
#   Example: PREPROCESS=false
#   Default: false
#
#  VISUALIZE: Flag whether to visualize the data
#   Format: Boolean (true || false)
#   Example: VISUALIZE=false
#   Default: false
#
#  VECTOR_TYPE: Text vector identifier.
#   Format: String ['BOW', 'TFIDF', 'EMBD', 'META']
#   Example: VECTOR_TYPE='TFIDF'
#   Default: 'TFIDF'
#
#  FEATURE_TYPE: Additional features subset identifier.
#   Format: String ['LEX', 'CONV', 'RDY', 'EMO', 'MOR', 'EXP', 'NER', 'ALL']
#   Example: FEATURE_TYPE='ALL'
#   Default: 'ALL'
#
#  LEMMATIZE: Flag whether to lemmatize the data
#   Format: Boolean (true || false)
#   Example: LEMMATIZE=true
#   Default: true
#
#  STOP_WORDS: Flag whether to use stop_words
#   Format: Boolean (true || false)
#   Example: STOP_WORDS=true
#   Default: true
#
#######################################

TRAIN_ID='columbia'
TEST_ID='columbia'
EXPERIMENT='ED'
PREPROCESS=false
VISUALIZE=false
VECTOR_TYPE='EMBD'
FEATURE_TYPE='ALL'
LEMMATIZE=true
STOP_WORDS=true

#######################################
#
# Validating variables
#  Ensuring that variables are correctly set.
# 
#######################################

if  ([ $TRAIN_ID != 'columbia' ] && [ $TRAIN_ID != 'berkeley' ]) || \
    ([ $TEST_ID != 'columbia' ] && [ $TEST_ID != 'berkeley' ]) || \
    ([ $EXPERIMENT != 'E' ] && [ $EXPERIMENT != 'ED' ] \
      && [ $EXPERIMENT != 'EQ' ] && [ $EXPERIMENT != 'EQD' ] && [ $EXPERIMENT != 'B' ])
    ([ $PREPROCESS != false ] && [ $PREPROCESS != true ]) || \
    ([ $VISUALIZE != false ] && [ $VISUALIZE != true ]) || \
    ([ $VECTOR_TYPE != 'BOW' ] && [ $VECTOR_TYPE != 'TFIDF' ]
      && [ $VECTOR_TYPE != 'EMBD' ] && [ $VECTOR_TYPE != 'META' ]) || \
    ([ $FEATURE_TYPE != 'LEX' ] && [ $FEATURE_TYPE != 'CONV' ] \
      && [ $FEATURE_TYPE != 'RDY' ] && [ $FEATURE_TYPE != 'EMO' ]
      && [ $FEATURE_TYPE != 'MOR' ] && [ $FEATURE_TYPE != 'EXP' ]
      && [ $FEATURE_TYPE != 'NER' ] && [ $FEATURE_TYPE != 'ALL' ]) || \
    ([ $LEMMATIZE != false ] && [ $LEMMATIZE != true ]) || \
    ([ $STOP_WORDS != false ] && [ $STOP_WORDS != true ])
; then
    echo "Invalid variable value"
    exit 1
fi

#######################################
#
# Automated variable generation
# 
#######################################

PREPROCESS_CLAUSE=$([ "PREPROCESS" == true ] && echo "PRE" || echo "")
VISUALIZE_CLAUSE=$([ "VISUALIZE" == true ] && echo "VIS" || echo "")
LEMMA_CLAUSE=$([ "$LEMMATIZE" == true ] && echo "LEMM" || echo "")
STOP_WORDS_CLAUSE=$([ "$STOP_WORDS" == true ] && echo "STOP" || echo "")

FILE_NAME="EC_${TRAIN_ID}_${TEST_ID}_${$EXPERIMENT}_${PREPROCESS_CLAUSE}_${VISUALIZE_CLAUSE}_${VECTOR_TYPE}_${FEATURE_TYPE}_${LEMMA_CLAUSE}_${STOP_WORDS_CLAUSE}"

check_proc () {
    ps -p $(cat ${FILE_NAME}.pid)
}

kill_proc () {
    kill -9 $(cat ${FILE_NAME}.pid)
}

#######################################
#
# Script Execution
# 
#######################################

nohup python -u ../pipeline.py \
    --train_id $TRAIN_ID \
    --test_id $TEST_ID \
    --experiment $EXPERIMENT \
    --preprocess $PREPROCESS \
    --visualize $VISUALIZE \
    --vector_type $VECTOR_TYPE \
    --feature_type $FEATURE_TYPE \
    --lemmatize $LEMMATIZE \
    --stop_words $STOP_WORDS \
    >${FILE_NAME}.output \
    2>${FILE_NAME}.error \
    & echo $! > ${FILE_NAME}.pid


printf '%s\n' "Executing pipeline.py;" \
    "Process filename: ${FILE_NAME}.pid"
