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



## DATA
Data is from kaggle(https://www.kaggle.com/c/riiid-test-answer-prediction/)<br/>
Data Set contains:
### Train 
train.csv is user_data 
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

## AIM
Challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. so that we can make question according to their quality and very useful in targeting weak part student and help them to imporve, Just like our favorite teacher used to care us. 

## Feature Engineering
* **Average_Index** : it's Ratio of Prior_Elasped time and Average time to Answer Question Correctly.<br/>
* **Moving_Average_Index** : Moving Average of Average Index which helps to understand how the user used to Answer the question that is how far the question is                                      answered from the average.<br/>
* **No_of_Question_answered** : Total No of Question Answered by the user.<br/>
* **No_of_Question_answered_Correctly** : it's ratio of total_no_of_question_answered_correctly/total Question Attempted.<br/>
* **Percent_of_student_answered_Correctly** : As each question has it's own ID so taking ratio how many times the question answered correctly,total no of times                                                      questions attempted.<br/> 
* **Timestamp_diff** : This Features contains two steps:<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1) difference of timestamp.<br/>
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2) divide by the no of question in that bundle to get average timestamp taken by the user to answerd per question.<br/>
* **Day_Passed** : total no of day passed since user first interaction and it's obtained by dividing by 86400(24 hours).<br/>
* **Moving_average_prior_question_had_explaination/Moving_Explaination** : same procedure as Moving_average_index is genrated.<br/>

<p float="left">
  <img src="https://github.com/sahilabs/Riiid-Answer-Correctness/blob/main/Image/Feature%20Importance.png" />
</p>

# AUC_Metric
 AUC is Area Under Curve here Curve is ROC
 **ROC(Receiver operating characteristic)**: ROC explaination in simple Steps.<br/>
 1)Basically it's Curve drawn from TruePositive(as Y_axis) and False Positive(as X-axis).<br/>
 2)first sort all of the prediction and map to the actual value and start drawing  like when 1 comes go one step Horizontal and 0 comes fo one step one step
   Vertical<br/>
 3) Drawn curve is ROC.br/>
 Benfits of using AUC is that it doesn't depend on Threshold and the more AUC then the class is easily separable
