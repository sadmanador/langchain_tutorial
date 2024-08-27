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
    "mitochondria is the powerhouse of the cell",
    "mitochondria is made of lipids",
  ],
  [{ id: 1 }, { id: 2 }],
  new OpenAIEmbeddings()
);

const retriever = vectorStore.asRetriever();

const condenseQuestionTemplate = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
  
  Chat History: {chat_history}
    
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

const formatChatHistory = (chatHistory) => {
  const formattedDialogueTurns = chatHistory.map(
    (dialogueTurn) => `Human: ${dialogueTurn[0]}\nAssistant: ${dialogueTurn[1]}`
  );
  return formattedDialogueTurns.join("\n");
};

const standaloneQuestionChain = RunnableSequence.from([
  {
    chat_history: (input) => formatChatHistory(input.chat_history),
    question: (input) => input.question,
  },
  CONDENSE_QUESTION_PROMPT,
  model,
  new StringOutputParser(),
]);

const combineDocumentsFn = (docs, separator = "\n\n") => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

const answerChain = RunnableSequence.from([
  {
    context: retriever.pipe(combineDocumentsFn),
    question: new RunnablePassthrough(),
  },
  ANSWER_PROMPT,
  model,
]);

const conversationalRetrievalQAChain =
  standaloneQuestionChain.pipe(answerChain);

// const result1 = await conversationalRetrievalQAChain.invoke({
//   question: "What is the powerhouse of the cell?",
//   chat_history: [],
// });
// console.log(result1);
/*
    AIMessage { content: "The powerhouse of the cell is the mitochondria." }
  */

const result2 = await conversationalRetrievalQAChain.invoke({
  question: "What are they made out of?",
  chat_history: [
    [
      "What is the powerhouse of the cell?",
      "The powerhouse of the cell is the mitochondria.",
    ],
  ],
});
console.log(result2);
