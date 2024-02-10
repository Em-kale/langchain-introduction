import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

//in memory vector store that works in a node environment
import { CloseVectorNode } from "@langchain/community/vectorstores/closevector/node";

console.log("starting the program");
console.log("loading the documents from the vitalpoint.ai website...");

const loader = new CheerioWebBaseLoader(
  //load in information about langchain
  "https://vitalpoint.ai/"
);
const docs = await loader.load();

console.log("finished loading the documents from the vitalpoint.ai website");

//split the text into more manageable chunks
console.log("splitting the documents into smaller chunks...");
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

console.log("creating the embeddings...");
const embeddings = new OllamaEmbeddings({
  model: "mistral",
});
console.log("embeddings created");

console.log("creating the vector store...");

//this will take our embeddings model, which uses mixtral embeddings, and apply that to the splitDocs, generating
//the relationships of the tokens in vector space
const vectorstore = await CloseVectorNode.fromDocuments(
  splitDocs,
  new OllamaEmbeddings()
);

console.log("vector store created");

console.log("initializing the chat model...");
//this is the chat model that we will use to generate the responses
const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "mistral",
});

//essentially sets what messages have happened alrady
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  ["user", "{input}"],
]);

async function doPrompt() {
  console.log("`creating document chain...");

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  console.log("creating retriever...");
  //wraps the vector store so it conforms to the retriever interface
  //it in turn will use vector store methods to query the data , but lets you treat all retrievers the same
  const retriever = vectorstore.asRetriever();

  console.log("creating retrieval chain...");
  //this should just handle adding the context to the input but I'm not confident how - need to investigate
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  console.log("invoking retrieval chain...");
  const result = await retrievalChain.invoke({
    input: "what is vital point ai",
  });
  console.log("finished.");
  console.log(result);
}

doPrompt();
