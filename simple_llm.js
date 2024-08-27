import dotenv from "dotenv";
dotenv.config();
const openApiKey = process.env.OPENAI_API_KEY;
import { OpenAI } from "@langchain/openai";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import {
//   RunnableSequence,
//   RunnablePassthrough,
// } from "langchain/schema/runnable";
// import { MemoryVectorStore } from "langchain/vectorstores/memory";
// import { StringOutputParser } from "langchain/schema/output_parser";



const model = new OpenAI({
  openAIApiKey: openApiKey,
  temperature: 0.7,
});

// const vectorStore = await MemoryVectorStore.fromTexts(
//     [
//       "mitochondria is the powerhouse of the cell",
//       "lysosomes are the garbage disposal of the cell",
//       "the nucleus is the control center of the cell",
//     ],
//     [{ id: 1 }, { id: 2 }, { id: 3 }],
//     new OpenAIEmbeddings(),
//   );

//   const retriever = vectorStore.asRetriever();

const prompt = new PromptTemplate({
  template: "What is {thing}?",
  inputVariables: ["thing"],
});


const chain = new LLMChain({ llm: model, prompt });


const response = await chain.invoke({ thing: "port" });

console.log(response);
