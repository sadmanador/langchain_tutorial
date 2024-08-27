import dotenv from "dotenv";
dotenv.config();
const openApiKey = process.env.OPENAI_API_KEY;
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

const model = new ChatOpenAI({ apiKey: openApiKey });

const promptTemplate = PromptTemplate.fromTemplate(
  "Give me a list of facts about {subject}"
);

// a stop sequence is use to get only 3 facts as the results are coming as 1. 2. 3. 4. 5. with a line break at the beginning
const chain = promptTemplate.pipe(model.bind(
    { stop: ["\n4."] }
));

const result = await chain.invoke({ subject: "capybaras" });

console.log(result);
