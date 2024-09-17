# prompt_template_1 = """
#   You will be provide to compare multiple documentations in form of chunks, if asked to compare, then check the equivalent information from both documentation and provide the accurate results.
#   Answer the question as detailed as possible from the provided context that is provide all the required data provided in the article along with your own formed sentences, make sure to provide all the details, if the answer is not in provided context, if the question is about explaining or defining things then, explain in your own terms. The answer generated should be unique and not completely from the article.
#   Analyse the whole related document and answer it unique.
#   If the question is completely non related to the articles just say, "Ask different Question", don't provide the wrong answer\n\n
#   Context:\n {context}?\n
#   Question: \n{question}\n
#   """

prompt_template_1 = """
  You will be provided with multiple documentations in the form of chunks. If asked to compare, then analyze the equivalent information from both documentation sources and provide a detailed comparison.
  Answer the question by synthesizing the provided context and expanding upon it with original thought. Use the context as a reference, but ensure that your response includes your own interpretation, rephrasing, and additional insights.
  Make sure to cover all key details from the context, but avoid simply restating it. Instead, reword, explain in your own terms, and add unique contributions to the answer.
  If the answer is not found in the context or if the question requires definitions or explanations, provide those using your own knowledge.
  If the question is completely unrelated to the articles, respond with "Ask a different question" without providing an incorrect or misleading answer.
  
  Context:\n {context}?\n
  Question: \n{question}\n
  """
