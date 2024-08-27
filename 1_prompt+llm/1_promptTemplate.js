import dotenv from "dotenv";
dotenv.config();
const openApiKey = process.env.OPENAI_API_KEY;
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";


const model = new ChatOpenAI({ apiKey: openApiKey});

const promptTemplate = PromptTemplate.fromTemplate(
  "what is the {thing} of {country}"
);

const chain = promptTemplate.pipe(model);

const result = await chain.invoke({ thing: "capital", country: 'Bangladesh' });

console.log(result);

/*
1. use a model
2. make a prompt using prompt template
3. chain will get the prompt and pass the value to a model
4. the chain call or invoke

the flow: input variable --> prompt template --> prompt --> model --> result
*/