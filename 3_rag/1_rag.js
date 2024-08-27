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

//this is the vector store which have the data for the context
const vectorStore = await MemoryVectorStore.fromTexts(
  [
    "The yabadabadoo is the powerhouse of the cell",
    "lysosomes are the garbage disposal of the cell",
    "the nucleus is the control center of the cell",
  ],
  [{ id: 1 }, { id: 2 }, { id: 3 }],
  new OpenAIEmbeddings()
);


//A retriever is made to provide the given context
const retriever = vectorStore.asRetriever();

const prompt =
  PromptTemplate.fromTemplate(`Answer the question based only on the following context:
  {context}
  
  Question: {question}`);


//this function make the data in a single string
const serializeDocs = (docs) => docs.map((doc) => doc.pageContent).join("\n");



const chain = RunnableSequence.from([
  {
    // retriever make the data as string
    context: retriever.pipe(serializeDocs),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);



const result = await chain.invoke("What is the powerhouse of the cell?");

console.log(result);
