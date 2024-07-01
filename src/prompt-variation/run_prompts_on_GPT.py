from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
from tqdm import tqdm

project_path_base = "/home/leczhang/research/llm-personas/"

lst = ['prompt_sensitivity/Truefalse_short-statement_double-bar-separated_colon-zero-space-ending_answer-asking', 
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-double-space-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_response-asking',
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-single-space-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-zero-space-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_question-mark-linebreak-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_single-space-separated_colon-zero-space-ending_answer-asking',
       'prompt_sensitivity/Truefalse_short-statement_triple-sharp-linebreak-separated_colon-linebreak-ending_answer-asking',]



API_KEY=''
max_length = 4000
llm = ChatOpenAI(
    #model_name='gpt-4', 
    model_name='gpt-3.5-turbo', 
                 openai_api_key=API_KEY, temperature=0,
                               max_retries=12, request_timeout=600) 

template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt,llm=llm)
llm_chain

for zlc in lst:
    prompt_path = project_path_base + "data/paraphrased-prompts-modified/" + zlc + ".json"

    with open(prompt_path) as f:
        prompt_dict = pd.read_json(f)

    temp = []

    for prompt_text in tqdm(prompt_dict['text']):
        output=llm_chain.run(prompt_text)
        first_word = output.split(" ")[0]
        
        result_dict = {'prompt': prompt_text}
        result_dict["p(" + first_word + ")"] = 1
        temp.append(result_dict)

    model_record_df = pd.DataFrame(temp).fillna(0)
    model_record_df.to_csv(project_path_base + "result/" + zlc + "_gpt35.csv")