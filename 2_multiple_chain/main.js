import dotenv from "dotenv";
dotenv.config();

const openApiKey = process.env.OPENAI_API_KEY;

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({ apiKey: openApiKey });

const prompt1 = PromptTemplate.fromTemplate(
  `What is the city {person} is from? Only respond with the name of the city.`
);
const prompt2 = PromptTemplate.fromTemplate(
  `What country is the city {city} in? Respond in {language}.`
);

const chain1 = prompt1.pipe(model).pipe(new StringOutputParser());

// Linking the output of chain1 to the input of prompt2
const combinedChain = RunnableSequence.from([
  // this invokes chain1 to get the city name
  {
    city: chain1,
    //passing the direct language with not work
    // this second parameter which is not part of chain1 is a runnable program rather than a string
    language: (input) => input.language,
  },
  prompt2,
  model,
  new StringOutputParser(),
]);

const result = await combinedChain.invoke({
  person: "Obama",
  language: "German",
});

console.log(result);
