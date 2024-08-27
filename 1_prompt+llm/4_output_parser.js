import dotenv from "dotenv";
dotenv.config();
const openApiKey = process.env.OPENAI_API_KEY;
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

const model = new ChatOpenAI({ apiKey: openApiKey });

const promptTemplate = PromptTemplate.fromTemplate(
  "tell me a joke about {topic}"
);

// make the output short and output can be modified based on any requirement
const outputParser = new StringOutputParser();

/* 
way no. 1. to bind outputParser with chain
const chain = promptTemplate.pipe(model).pipe(outputParser);
*/

/*
way no. 2 involve to import runnableSequence
const chain = RunnableSequence.from([promptTemplate, model, outputParser]);
*/
const chain = RunnableSequence.from([promptTemplate, model, outputParser]);

const result = await chain.invoke({ topic: "bears" });

console.log(result);
