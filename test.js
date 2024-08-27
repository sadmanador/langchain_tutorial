import dotenv from "dotenv";
dotenv.config();

import fs from "fs/promises";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";

// Initialize the model with the OpenAI API key
const model = new ChatOpenAI({model: "gpt-4o-mini", apiKey: process.env.OPENAI_API_KEY });

// Function to read the file and create vector store
async function createVectorStoreFromTextFile(filePath) {
  try {
    const fileContent = await fs.readFile(filePath, "utf-8");
    const documents = fileContent.split("\n\n"); // Adjust the split based on your file's formatting
    const vectorStore = await MemoryVectorStore.fromTexts(
      documents,
      documents.map((_, index) => ({ id: index + 1 })),
      new OpenAIEmbeddings()
    );
    return vectorStore;
  } catch (error) {
    console.error("Error reading or processing the text file:", error);
  }
}

// Main function to set up and run the chains
async function runConversationalRetrievalQA() {
  const vectorStore = await createVectorStoreFromTextFile("./all.txt");
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
      (dialogueTurn) =>
        `Human: ${dialogueTurn[0]}\nAssistant: ${dialogueTurn[1]}`
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

  const answerChain = RunnableSequence.from([
    {
      context: retriever.pipe(combineDocumentsFn),
      question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
    new StringOutputParser(),
  ]);

  const conversationalRetrievalQAChain = standaloneQuestionChain.pipe(
    answerChain
  );

  // First invoker: Asking about the urinary tract
  const result1 = await conversationalRetrievalQAChain.invoke({
    question: "What does the urinary tract include?",
    chat_history: [],
  });
  console.log(result1);
  /*
    AIMessage { content: "The urinary tract includes the kidneys, the bladder, the tubes that carry urine from the kidneys to the bladder (ureters), and the tube that carries urine from the bladder to outside of the body (urethra)." }
  */

  // Second invoker: Asking about the method of diagnosis for UTIs
  const result2 = await conversationalRetrievalQAChain.invoke({
    question: "How is a urinary tract infection diagnosed in children?",
    chat_history: [
      [
        "What does the urinary tract include?",
        "The urinary tract includes the kidneys, the bladder, the tubes that carry urine from the kidneys to the bladder (ureters), and the tube that carries urine from the bladder to outside of the body (urethra).",
      ],
    ],
  });
  console.log(result2);
  /*
    AIMessage { content: "The doctor will need to perform a urine culture on sterile urine to determine if the child does indeed have a UTI. To do this on a young baby, the doctor may place a plastic bag over the genitals to collect the urine, although this method isn't very accurate. A better way of collecting a urine sample for culture is by inserting a catheter up the urethra and retrieving urine directly from the bladder." }
  */
}

// Run the main function
runConversationalRetrievalQA();
