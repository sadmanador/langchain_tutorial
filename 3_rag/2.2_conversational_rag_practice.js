import dotenv from "dotenv";
dotenv.config();

const openApiKey = process.env.OPENAI_API_KEY;

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({ apiKey: openApiKey });

const vectorStore = await MemoryVectorStore.fromTexts(
    [
      "The golden key is in the Mountains of Ilsodor",
      "To reach the Mountains of Ilsodor, you must travel northwest from the Forest of Forloson",
    ],
    [{ id: 1 }, { id: 2 }],
    new OpenAIEmbeddings(),
  );
  
  const retriever = vectorStore.asRetriever();
  
  const condenseQuestionTemplate = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
  
  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:
  `;
  
  const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(
    condenseQuestionTemplate
  );
  
  const answerTemplate = `Answer the question based only on the following context:
  {context}
  
  Question: {question}
  `;
  
  const ANSWER_PROMPT = PromptTemplate.fromTemplate(answerTemplate);
  
  const combineDocumentsFn = (docs, separator = "\n\n") => {
    const serializedDocs = docs.map((doc) => doc.pageContent);
    return serializedDocs.join(separator);
  };
  
  const formatChatHistory = (chatHistory) => {
    const formattedDialogueTurns = chatHistory.map(
      (dialogueTurn) => `Human: ${dialogueTurn[0]}\nAssistant: ${dialogueTurn[1]}`
    );
    return formattedDialogueTurns.join("\n");
  };
  
  const standaloneQuestionChain = RunnableSequence.from([
    {
      chat_history: (input) => formatChatHistory(input.chat_history),
      question: (input) => input.question
    },
    CONDENSE_QUESTION_PROMPT,
    model,
    new StringOutputParser()
  ]) 
  
  const answerChain = RunnableSequence.from([
    {
      context: retriever.pipe(combineDocumentsFn),
      question: new RunnablePassthrough()
    },
    ANSWER_PROMPT,
    model,
    new StringOutputParser()
  ])
  
  const conversationalRetrievalQAChain = standaloneQuestionChain.pipe(answerChain);
  
  // const result1 = await conversationalRetrievalQAChain.invoke({
  //   question: "Where is the golden key?",
  //   chat_history: [],
  // });
  // console.log(result1);
  /*
    AIMessage { content: "The golden key is in the Mountains of Ilsodor. }
  */
  
  const result2 = await conversationalRetrievalQAChain.invoke({
    question: "How do I get there?",
    chat_history: [
      [
        "Where is the golden key?",
        "The golden key is in the Mountains of Ilsodor.",
      ],
    ],
  });
  console.log(result2);
  /*
    AIMessage { content: "To reach the Mountains of Ilsodor, you should travel northwest from the Forest of Forloson." }
  */