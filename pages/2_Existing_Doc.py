import streamlit as st
import os
import helper as help
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

llm = ChatOpenAI(model="gpt-3.5-turbo")

cosine_distance_evaluator = load_evaluator("embedding_distance")

geval_metric = GEval(
    name="Teacher",
    evaluation_steps=["Check whether ideas presented in 'actual output'\
                       are similar to those in 'input'"],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4"        
)
    
relevancy_metric = AnswerRelevancyMetric(
    model="gpt-4"
)

faithfulness_metric = FaithfulnessMetric(
    model="gpt-4"
)

st.set_page_config(page_title="existing doc")
st.sidebar.header("Query a stored document")

docs = os.scandir("vectorstores")
temp = []
for entry in docs:
    temp.append(entry.name)

option = st.selectbox(
    'Choose a stored file to query...',
    temp,
    index=None,
    placeholder="Select..."
)

if option:
    chain = help.fetch_db(option)

    question = st.text_input('Enter query here')
    reference_answer = st.text_input('Enter reference answer here')

    if question and reference_answer:
        answer, source = help.invoke(chain, question)
        st.write(answer)
      
        context = []
        for doc in source:
            context.append(doc.page_content)

        no_index = llm.invoke(question)

        relevancy_test = LLMTestCase(
            input=question,
            actual_output=str(answer),
        )

        faithfulness_test = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context
        )

        teacher_test_case = LLMTestCase(
            input=reference_answer,
            actual_output=answer
        )

        no_index_case = LLMTestCase(
            input=reference_answer,
            actual_output=no_index
        )

        cos_dist = cosine_distance_evaluator.evaluate_strings(
            prediction=answer,
            reference=reference_answer
        )
        
        st.write("cos_dist of rag answer and real answer:", cos_dist)

        relevancy_metric.measure(relevancy_test)
        st.write("Relevancy score:", relevancy_metric.score)
        st.write(relevancy_metric.reason)

        faithfulness_metric.measure(faithfulness_test)
        st.write("Faithfulness score:", faithfulness_metric.score)
        st.write(faithfulness_metric.reason)

        geval_metric.measure(teacher_test_case)
        st.write("G_Eval score with index:", geval_metric.score)
        st.write(geval_metric.reason)

        geval_metric.measure(no_index_case)
        st.write("G_Eval score without index:", geval_metric.score)
        st.write(geval_metric.reason)



        