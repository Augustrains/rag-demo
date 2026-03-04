import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import bs4
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def load_config() -> None:
    """Load .env + set environment variables (LangSmith / HF mirror / UA)."""
    load_dotenv()

    # Optional: identify web requests
    os.environ.setdefault(
        "USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) rag-demo/1.0"
    )

    # Optional: LangSmith tracing
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def build_llm() -> ChatDeepSeek:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Put it in .env or environment variables.")

    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=api_key,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_docs(url: str, chunk_size: int, chunk_overlap: int):
    loader = WebBaseLoader(
        web_paths=(url,),
        # bs_kwargs=dict(
        #     parse_only=bs4.SoupStrainer(
        #         class_=("post-content", "post-title", "post-header")
        #     )
        # ),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    # Add chunk_id for debugging / citations
    for i, d in enumerate(splits):
        d.metadata.setdefault("source", url)
        d.metadata["chunk_id"] = i
    
    #------------------------判断当前过滤后的网页内容-----------------------
    # docs = loader.load()
    # print("URL:", url, "docs:", len(docs), "content_len:", [len(d.page_content or "") for d in docs])

    # splits = splitter.split_documents(docs)
    # print("URL:", url, "splits:", len(splits))

    return splits


def build_retriever(splits, embeddings, k: int):
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(
            f"[source={d.metadata.get('source')} chunk={d.metadata.get('chunk_id')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_Multi_query_chain(retriever,llm,*,question_num=4,K=6):
    gen_prompt=ChatPromptTemplate.from_template(
        """你是一个检索增强问答系统的Query扩展器。请根据用户问题，生成{n}个近似问题(和原问题语义相似)，用于提高召回率。
        要求: 1.每行输出一个问题 2.不要编号，不要多余的解释 3.与原问题保持风格一致
        用户问题:{question}
        """
    )

    def parse_queries(text:str)->List[str]:
        """
        将大模型生成的问题解析出来，存放在List中
        """

        #按换行符拆成多行,然后去掉前后空格和过滤空行
        lines=[ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned=[]
        for ln in lines:
            #去掉例如-xx ，*xx的形式
            ln2=ln.lstrip("-.*").strip()
            #去掉例如1. 1)的形式
            if len(ln2)>=3 and ln2[0].isdigit() and ln2[1] in [".",")"]:
                ln2=ln2[2:].strip()
            cleaned.append(ln2)
        return cleaned
    
    #合并多query检索并且去重
    def multi_retriever(inputs:Dict[str,Any]):
        question =inputs["question"]

        #首先生成多个问题
        q_text=gen_prompt.format_messages(n=question_num,question=question )
        q_out=llm.invoke(q_text)
        queries=parse_queries(q_out.content)
        print("-------------queries--------------\n","Generated queries:", queries)

        if question not in queries:
            queries=[question]+queries
        
        #对每个问题进行检索
        all_lists=[]
        for q in queries:
            docs=retriever.invoke(q)
            all_lists.append(docs)
        #去重排序与Top-K
        seen=set()
        uniq_docs=[]
        max_len=max((len(docs) for docs in all_lists),default=0)
        for i in range(max_len):
            for docs in all_lists:
                if i>=len(docs):
                    continue
                d=docs[i]
                key=(d.metadata.get('source'),d.metadata.get('chunk_id'),d.page_content[:200])
                if key in seen:
                    continue
                seen.add(key)
                uniq_docs.append(d)
                if len(uniq_docs)>=K:
                    return uniq_docs
        return uniq_docs
    
    def format_docs(docs):
        return "\n\n".join(
            f"[source={d.metadata.get('source')},chunk={d.metadata.get('chunk_id')}]\n{d.page_content}"
            for d in docs
        )
    
    #最终问题
    answer_prompt=ChatPromptTemplate.from_template(
        """你需要根据给定的上下文回答问题。如果上下文中没有答案，请回答:"不知道"
        上下文:{context}
        问题:{question}
    """
    )

    chain=(
        {
            "question":RunnablePassthrough(),
            "context":RunnableLambda(lambda q:{"question":q})
            |RunnableLambda(multi_retriever)
            |RunnableLambda(format_docs)
        }
        |answer_prompt
        |llm
        |StrOutputParser()
    )
    return chain



# ---------- Routing components ----------

def build_domain_retriever(url:str,embeddings,k:int,chunk_size:int,chunk_overlap:int):
    """
    根据网址生成对应检索器
    """
    splits = load_and_split_docs(url, chunk_size, chunk_overlap)
    retriever = build_retriever(splits, embeddings, k)
    return retriever

def select_retriver(embeddings,k,chunk_size,chunk_overlap,llm):
    """
    生成多个检索器，并且选择适合的检索器
    """
    urls=[
       "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://www.geeksforgeeks.org/dynamic-programming/",
       "https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx"]
    
    retriever_infos = {
        "agents":{
            "retriever":build_domain_retriever(urls[0],embeddings,k,chunk_size,chunk_overlap),
            "description":"""这一路检索器覆盖 LLM Agent/智能体 相关内容：Agent 架构、规划与分解（task decomposition）、工具使用（tool use / function calling）、反思与自我改进（reflection）、记忆（memory）、ReAct/Plan-and-Execute 等范式，以及 agent 评估与常见设计模式。
适合问题： “什么是 task decomposition / planning？”“agent 如何调用工具？”“ReAct 是什么？”“reflection 如何做？”“agent memory 怎么设计？”
不适合： 纯算法/动态规划教程；微积分知识与公式推导。
关键词提示： agent, planning, task decomposition, tool use, ReAct, reflection, memory, autonomy.
"""
        },
        "dp": {
            "retriever": build_domain_retriever(urls[1],embeddings,k,chunk_size,chunk_overlap),
            "description": """这一路检索器覆盖 动态规划（Dynamic Programming） 入门与常见套路：最优子结构、重叠子问题、状态定义、转移方程、记忆化搜索 vs 迭代 DP、时间/空间复杂度分析，以及典型题型（背包、LCS、LIS、矩阵链、区间 DP 等）。内容偏 算法学习与刷题 风格。
适合问题： “DP 的定义/思想是什么？”“如何设计状态与转移？”“0/1 背包怎么写？”“记忆化和递推区别？”“某题怎么用 DP 解？”
不适合： LLM agent 架构与 RAG；微积分定理、导数/积分技巧与证明。
关键词提示： dynamic programming, memoization, state, transition, knapsack, LCS, LIS, recursion vs iteration.
"""
        },
        "calculus": {
            "retriever": build_domain_retriever(urls[2],embeddings,k,chunk_size,chunk_overlap),
            "description":"""这一路检索器覆盖 微积分 I（Calculus I） 系列课程笔记：极限、连续、导数定义与求导法则、导数应用（单调性、极值、优化）、积分与基本积分技巧、微分/积分的核心定理等。内容偏 数学推导与例题讲解。
适合问题： “极限怎么求？”“导数法则/链式法则？”“求极值/最优化？”“积分基本技巧？”“微积分基本定理是什么？”
不适合： 动态规划刷题；LLM agent/RAG 设计。
关键词提示： limits, continuity, derivative, chain rule, optimization, integral, Fundamental Theorem of Calculus.
"""
        }}
    
    candidates_text="\n\n".join(
        [f"{name}:{info['description'].strip()}" for name,info in retriever_infos.items()]
    )

    route_prompt=ChatPromptTemplate.from_template("""你是一个路由器。请根据用户提出的问题选择适合的路由器。注意，只能从候选项中选择路由器，并且只能输出一个key(不需要任何额外的解释)
                                                   用户问题：{question}
                                                   路由器候选项：{candidates}
                                                   输出:只返回一个key
""")
    
    valid_keys=set(retriever_infos.keys())
    
    def route(question:str)->str:
        """
        返回一个路由器的key，例如 agents、dp、calculus
        """
        msgs=route_prompt.format_messages(question=question,candidates=candidates_text)
        out=llm.invoke(msgs).content
        out=out.strip().strip('"').strip("'")

        if out not in valid_keys:
            out='agents'
        return out
    return retriever_infos,route


def rag_build_chain(retriever_infos,route,llm):
    """
    route->retriever->llm
    """
    def format_docs(docs):
        return "\n\n".join(
            f"[source={d.metadata.get('source','unknown')} chunk={d.metadata.get('chunk_id')}]\n{d.page_content}" for d in docs
        )
    def retrieve_by_route(inputs:Dict[str,Any]):
        question=inputs['question']
        key=route(question)
        docs=retriever_infos[key]['retriever'].invoke(question)
        print("ROUTE TO: " ,key)
        return {"question":question,"domain":key,"docs":docs}

    answer_prompt=ChatPromptTemplate.from_template(
        """你需要根据给定的上下文回答问题。如果上下文中没有答案，请回答:"不知道"
（路由选择：{domain}）

上下文：
{context}

问题：{question}
""")
    
    chain=(
        {"question":RunnablePassthrough()}
        |RunnableLambda(retrieve_by_route)
        |RunnableLambda(lambda x: {
            "question": x["question"],
            "domain": x["domain"],
            "context": format_docs(x["docs"]),
        })
        |answer_prompt
        |llm
        |StrOutputParser()
    )
    return chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://lilianweng.github.io/posts/2023-06-23-agent/")
    parser.add_argument("--query", default="What is Task Decomposition?")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--task",default="baserag",choices=["baserag","multirag","routing"])
    args = parser.parse_args()

    load_config()
    llm = build_llm()
    embeddings = build_embeddings()
    splits = load_and_split_docs(args.url, args.chunk_size, args.chunk_overlap)
    retriever = build_retriever(splits, embeddings, args.k)
    retriever_infos, route = select_retriver(
        embeddings, args.k, args.chunk_size, args.chunk_overlap, llm
    )
   
    TASK_REGISTRY = {
    "baserag": build_chain,
    "multirag": build_Multi_query_chain,
    "routing": lambda _retriever,_llm:rag_build_chain(retriever_infos,route,_llm)
    # "corrective": build_corrective_chain,
}  
    rag_chain = TASK_REGISTRY[args.task](retriever, llm)
    
    ans = rag_chain.invoke(args.query)

    print("\n" + "-" * 30 + " ANSWER " + "-" * 30)
    print(ans)


if __name__ == "__main__":
    main()