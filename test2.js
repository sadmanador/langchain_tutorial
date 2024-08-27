import dotenv from "dotenv";
dotenv.config();

import fs from "fs/promises";
import { MongoClient } from "mongodb";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";

// Initialize the model with the OpenAI API key
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  apiKey: process.env.OPENAI_API_KEY,
});

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

// Function to create a vector store from MongoDB documents
async function createVectorStoreFromMongoDB(uri) {
    const client = new MongoClient(uri);
  
    try {
      await client.connect();
      console.log("Connected to MongoDB");
  
      const collection = client.db("test").collection("devices");
      const documents = await collection.find({}).toArray();
      const textDocuments = documents.map((doc) => JSON.stringify(doc)); // Convert each document to a string format
  
      const vectorStore = await MemoryVectorStore.fromTexts(
        textDocuments,
        textDocuments.map((_, index) => ({ id: index + 1 })),
        new OpenAIEmbeddings()
      );
  
      return vectorStore;
    } catch (err) {
      console.error("Error connecting to MongoDB or processing documents:", err);
    } finally {
      await client.close();
      console.log("MongoDB connection closed");
    }
  }
  

// Main function to set up and run the chains
async function runConversationalRetrievalQA() {
  const vectorStoreTextFile = await createVectorStoreFromTextFile("./all.txt");
  const vectorStoreMongoDB = await createVectorStoreFromMongoDB(process.env.MONGODB_URI);

  // Combine vector stores or use separately
  const retrieverTextFile = vectorStoreTextFile.asRetriever();
  const retrieverMongoDB = vectorStoreMongoDB.asRetriever();

  const condenseQuestionTemplate = `Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow-Up Input: {question}
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
      context: retrieverTextFile.pipe(combineDocumentsFn), // Use text file retriever
      question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
    new StringOutputParser(),
  ]);

  const answerChainMongoDB = RunnableSequence.from([
    {
      context: retrieverMongoDB.pipe(combineDocumentsFn), // Use MongoDB retriever
      question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
    new StringOutputParser(),
  ]);

  const conversationalRetrievalQAChain = standaloneQuestionChain.pipe(answerChain);
  const conversationalRetrievalQAChainMongoDB = standaloneQuestionChain.pipe(answerChainMongoDB);

  // First invoker: Asking about the urinary tract
  const result1 = await conversationalRetrievalQAChain.invoke({
    question: "What does the urinary tract include?",
    chat_history: [],
  });
  console.log("Text file result:", result1);

  // Second invoker: Asking a MongoDB based question
  const result2 = await conversationalRetrievalQAChainMongoDB.invoke({
    question: "Show me all devices registered last week.",
    chat_history: [],
  });
  console.log("MongoDB result:", result2);
}

// Run the main function
runConversationalRetrievalQA();
