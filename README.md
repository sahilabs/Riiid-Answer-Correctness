# Riiid-Answer-Correctness

## Table of Contents
* [Riiid-Answer-Correctness](#Riiid-Answer-Correctness)
* [Motivation](#Motivation)
* [Library](#Library)
* [DATA](#DATA)
  * [Train](#Train)
  * [Lecture](#Lecture)
  * [Question](#Question)
* [AIM](#AIM)
* [Requirement](#Requirement)
* [Sampler](#Sampler)
* [Feature_Engineering](#Feature_Engineering)
* [AUC_Metric](#AUC_Metric)
* [Model](#Model)
* [Loss](#Loss)
* [Model_Performance](#Model_Performance)


## Riiid-Answer-Correctness
Riiid is company which aims to empower global education players to rethink the traditional ways of learning via extending Riiid's AI competency Riiid's AI tutor solution replaces textbooks and traditional lectures with a personalized AI tutor outpacing human tutors at a fraction of the cost.
Riiid company introduces competition to find user's knowledge and can use to find which set of question is suitabale for user

## Motivation
In 2018, 260 million children weren't attending school. At the same time, more than half of these young students didn't meet minimum reading and math standards. Education was already in a tough place when COVID-19 forced most countries to temporarily close schools. This further delayed learning opportunities and intellectual development. The equity gaps in every country could grow wider. We need to re-think the current education system in terms of attendance, engagement, and individualized attention.

## AIM
Challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. so that we can make question according to their quality and very useful in targeting weak part student and help them to imporve, Just like our favorite teacher used to care us.

## DATA
Data is from kaggle(https://www.kaggle.com/c/riiid-test-answer-prediction/)<br/>
Data Set contains:
### Train 
train.csv is user_data and 
* list
 * sub-list
#### Attributes
 * **User_id** : unique ID corresponds each user.<br/>
 * **timestamp**: (int64) the time in milliseconds between this user interaction and the first event completion from that user.<br/>
 * **content_id**: (int16) ID code for the user interaction.<br/>
 * **content_type_id**: (int8) 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.<br/>
 * **task_container_id**: (int16) Id code for the batch of questions or lectures. For example, a user might see three questions in a row before seeing the explanations for any of them. Those three would all share a task_container_id.<br/>
 * **user_answer**: (int8) the user's answer to the question, if any. Read -1 as null, for lectures.<br/>
 * **answered_correctly**: (int8) if the user responded correctly. Read -1 as null, for lectures.<br/>
 * **prior_question_elapsed_time**: (float32) The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between. Is null for a user's first question bundle or lecture. Note that the time is the average time a user took to solve each question in the previous bundle.<br/>
 * **prior_question_had_explanation**: (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback<br/>
<br/>
<br/>

## Question
   questions.csv: metadata for the questions posed to users.<br/>
### Attributes
 * **question_id**: foreign key for the train/test content_id column, when the content type is question.<br/>

 * **bundle_id**: code for which questions are served together.<br/>

 * **correct_answer**: the answer to the question. Can be compared with the train user_answer column to check if the user was right.<br/>

 * **part**: the relevant section of the TOEIC test.<br/>

 * **tags**: one or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions              together.<br/>

## lectures
   lectures.csv: metadata for the lectures watched by users as they progress in their education.<br/>
### Attributes
 * **lecture_id**: foreign key for the train/test content_id column, when the content type is lecture.<br/>

 * **part**: top level category code for the lecture.<br/>

 * **tag**: one tag codes for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.<br/>
 asdasd
