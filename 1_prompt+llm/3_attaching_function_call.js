import dotenv from "dotenv";
dotenv.config();
const openApiKey = process.env.OPENAI_API_KEY;
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

const prompt = PromptTemplate.fromTemplate(
  // speechUnit receives what to do with the subject
  `Tell me {speechUnit} about {subject}`
);

const model = new ChatOpenAI({ apiKey: openApiKey });

const functionSchema = [
  {
    name: "joke",
    description: "A joke",
    parameters: {
      type: "object",
      properties: {
        setup: {
          type: "string",
          description: "The setup for the joke",
        },
        punchline: {
          type: "string",
          description: "The punchline for the joke",
        },
      },
      required: ["setup", "punchline"],
    },
  },
  {
    name: "anecdote",
    description: "An anecdote",
    parameters: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description:
            "A question setting up the anecodte, starting with the words 'Did you know that'",
        },
        answer: {
          type: "string",
          description:
            "A follow-up to the question, elaborating the anecdote, starting with the words 'It's true!'",
        },
      },
      required: ["question", "answer"],
    },
  },
];

const chain = prompt.pipe(
  model.bind({
    functions: functionSchema,
  })
);

// invoking different functions
const result1 = await chain.invoke({
  speechUnit: "a joke",
  subject: "bears",
});

const result2 = await chain.invoke({
  speechUnit: "an anecdote",
  subject: "bears",
});

console.log(result1);
console.log(result2);
