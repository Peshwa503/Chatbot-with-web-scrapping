prompt_template_1 = """
  You will be provide to compare multiple documentations in form of chunks, if asked to compare, then check the equivalent information from both documentation and provide the accurate results.
  Answer the question as detailed as possible from the provided context that is provide all the required data provided in the article along with your own formed sentences, make sure to provide all the details, if the answer is not in provided context, if the question is about explaining or defining things then, explain in your own terms. 
  If the question is completely non related to the articles just say, "Ask different Question", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n
  """