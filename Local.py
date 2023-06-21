import streamlit as st
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

if torch.cuda.is_available():
        torch.cuda.empty_cache()

if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(model)
if 'pipeline' not in st.session_state:
    tokenizer = st.session_state['tokenizer']
    st.session_state['pipeline'] = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    temperature=0.1,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# tokenizer = AutoTokenizer.from_pretrained(model)

# pipeline = pipeline(
#     "text-generation", #task
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     temperature=0.1,
#     device_map="auto",
#     max_length=1000,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id
# )
if "llm" not in st.session_state:
    st.session_state["llm"] = HuggingFacePipeline(pipeline = st.session_state.pipeline, model_kwargs = {'temperature':0.1})

# llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.1})


def templatePrompt(context,query):
    from langchain import PromptTemplate, LLMChain

    template = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

    Context:
    {context}

    {query}""".strip()

    # context = '''Ezio Auditore da Firenze (Italian pronunciation: [ˈɛttsjo audiˈtoːre da (f)fiˈrɛntse]) is a fictional character in the video game series Assassin's Creed, an Italian master assassin who serves as the protagonist of the series' games set during the Italian Renaissance. His life and career as an assassin are chronicled in Assassin's Creed II, II: Discovery, Brotherhood, and Revelations, and the short films Assassin's Creed: Lineage and Assassin's Creed Embers. All games (excluding II: Discovery) and films he appears in were re-released as an enhanced bundle, The Ezio Collection, in 2016. Ezio has also been frequently referenced or made smaller appearances in other media within the franchise. Actor Roger Craig Smith has consistently provided the character's voice throughout his appearances, while Devon Bostick portrayed him in live-action in Lineage.
    #
    # Within the series' alternate historical setting, Ezio was born into Italian nobility from Florence in the year 1459. His family had long been loyal to the Assassin Brotherhood, a fictional organization inspired by the real-life Order of Assassins dedicated to protecting peace and freedom, but Ezio did not learn about his Assassin heritage until his late teens, after most of his immediate kin were killed during the Pazzi conspiracy. His quest to track down those responsible for killing his family eventually sets him up against the villainous Templar Order led by the House of Borgia. Spending years to fight against Rodrigo and Cesare Borgia and their henchmen, he eventually re-establishes the Brotherhood as the dominant force in Italy. His further adventures lead him to Spain and the Ottoman Empire, where he is also essential in overcoming Templar threats and restoring the Assassins. After his retirement from the Brotherhood, he lives a peaceful life in rural Tuscany until his eventual death from a heart attack in 1524.
    #
    # The character has received critical acclaim and is often named among the greatest video game characters of all time. While most of his praise focuses on his portrayal and growth throughout the series, as well as the unique chronicling of his entire life, he has also been noted as one of the most attractive video game characters of all time. Due to his reception and the fact that he is the only character in the series who is the protagonist of multiple major installments of the franchise,[a] he is usually considered the face of the franchise and its most popular character. Ezio's popularity has led to several crossover appearances outside of the Assassin's Creed series, notably Soulcalibur V, Fortnite, and Brawlhalla, as a guest character."'''

    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=template
    )

    # query = "Give summary of above context in 2 sentences"

    prompt = prompt_template.format(query=query, context=context)

    return prompt

def GetResponse(text):
    response = st.session_state.llm(text)
    print(response)
    return st.write(response)

with st.form('my_form'):
    context = st.text_area("Enter Context", placeholder="Enter Context Here")
    question = st.text_input("Enter Questions", placeholder="Enter Questions Here")
    submit = st.form_submit_button('Submit')
    prompt = templatePrompt(context, question)
    if submit:
        GetResponse(prompt)

